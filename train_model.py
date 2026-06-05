#!/usr/bin/env python3
"""
=============================================================================
Traffic Sign Recognition System — Complete Training Script
=============================================================================

This script trains a MobileNetV2-based classifier on the German Traffic Sign
Recognition Benchmark (GTSRB) dataset (43 classes).

Pipeline overview:
    1. Auto-download & cache the GTSRB dataset (train + test + labels)
    2. Parse PPM images → RGB numpy arrays, split into train/val/test
    3. Apply Albumentations augmentation (no horizontal flip!)
    4. Build MobileNetV2 transfer-learning model
    5. Two-phase training: frozen base → fine-tune top-30 layers
    6. Evaluate on held-out test set, print metrics & confusion matrix

Usage:
    python train_model.py

Requirements:
    pip install tensorflow albumentations scikit-learn numpy pandas pillow
=============================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS
# ──────────────────────────────────────────────────────────────────────────────

import os
import sys
import csv
import glob
import zipfile
import urllib.request
import shutil
import warnings

import numpy as np
import pandas as pd
from PIL import Image

# Suppress noisy TF logs before importing tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers, losses
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import albumentations as A

# Suppress non-critical warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION & CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

# ----- Paths ----------------------------------------------------------------
# All paths are relative to the script's directory so the project stays
# self-contained and portable.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")               # cached numpy arrays
DOWNLOAD_DIR = os.path.join(DATA_DIR, "downloads")       # raw zip / extracted
MODEL_DIR = os.path.join(BASE_DIR, "model")              # saved model output
MODEL_PATH = os.path.join(MODEL_DIR, "gtsrb_mobilenet.h5")

# ----- Local data (already present on disk) ----------------------------------
# The user already has 12,630 test PNG images in this folder — use them
# directly instead of re-downloading.
LOCAL_TEST_DIR = os.path.join(BASE_DIR, "test_images", "Test")

# ----- Dataset URLs ----------------------------------------------------------
# ONLY the training archive needs to be downloaded (~263 MB).
# Test images are loaded from LOCAL_TEST_DIR.
TRAIN_URL = (
    "https://sid.erda.dk/public/archives/"
    "daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
)
# Test labels CSV (small download, ~90 KB) — needed because the local
# GT-final_test.csv is empty (0 bytes).
TEST_LABELS_URL = (
    "https://sid.erda.dk/public/archives/"
    "daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"
)

# ----- Image / Model hyper-parameters ----------------------------------------
IMG_HEIGHT = 128        # MobileNetV2 accepts any size >= 32; 128 is a good
IMG_WIDTH = 128         # balance between detail and training speed.
NUM_CLASSES = 43        # GTSRB has exactly 43 traffic-sign categories

# ----- Training hyper-parameters ---------------------------------------------
BATCH_SIZE = 64
PHASE1_EPOCHS = 10      # feature-extraction phase (base frozen)
PHASE2_EPOCHS = 30      # fine-tuning phase (top-30 layers unfrozen)
PHASE1_LR = 1e-3
PHASE2_LR = 5e-5        # slightly higher than 1e-5 for faster convergence
VAL_SPLIT = 0.15        # fraction of training data used for validation

# ----- 43 GTSRB class labels -------------------------------------------------
CLASS_LABELS = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing veh over 3.5 tons",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicle > 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End speed + passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End no passing vehicle > 3.5 tons",
}

# ──────────────────────────────────────────────────────────────────────────────
# 2. UTILITY HELPERS
# ──────────────────────────────────────────────────────────────────────────────


def _ensure_dir(path: str) -> None:
    """Create directory (and parents) if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)


def _download_file(url: str, dest_path: str, expected_min_size: int = 0) -> None:
    """
    Download a file from *url* to *dest_path* with a simple progress bar.
    Skips the download if the file already exists AND is larger than
    expected_min_size (to detect partial / interrupted downloads).
    """
    if os.path.exists(dest_path):
        file_size = os.path.getsize(dest_path)
        if file_size > expected_min_size:
            print(f"  ✓ Already downloaded: {os.path.basename(dest_path)} ({file_size / (1024*1024):.1f} MB)")
            return
        else:
            print(f"  ⚠ Partial download detected ({file_size / (1024*1024):.1f} MB) — re-downloading…")
            os.remove(dest_path)

    print(f"  ⬇ Downloading {os.path.basename(dest_path)} …")

    # Use a reporthook to show percentage progress
    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(
                f"\r    {pct:5.1f}%  ({mb:.1f} / {total_mb:.1f} MB)"
            )
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=_progress)
    print()  # newline after progress bar


def _extract_zip(zip_path: str, extract_to: str) -> None:
    """
    Extract a zip archive to *extract_to*. Skips if already extracted
    (heuristic: the target directory is non-empty).
    """
    if os.path.isdir(extract_to) and os.listdir(extract_to):
        print(f"  ✓ Already extracted: {os.path.basename(zip_path)}")
        return

    print(f"  📦 Extracting {os.path.basename(zip_path)} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"    Done — extracted to {extract_to}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. DATA INGESTION — Download, Extract, Parse
# ──────────────────────────────────────────────────────────────────────────────


def download_and_extract() -> tuple:
    """
    Download ONLY the training archive (~263 MB) and the test labels CSV.
    Test images are loaded from the existing local directory (test_images/Test/)
    which already contains 12,630 PNG files.

    Returns
    -------
    train_root : str   – path containing the 43 class sub-folders
    test_root  : str   – path to existing test images
    test_csv   : str   – path to GT-final_test.csv (test labels)
    """
    _ensure_dir(DOWNLOAD_DIR)

    # --- Download training data (ONLY if not already present) -----------------
    train_zip = os.path.join(DOWNLOAD_DIR, "GTSRB_Training.zip")
    # Expected size ~263 MB; detect partial downloads (< 200 MB)
    _download_file(TRAIN_URL, train_zip, expected_min_size=200_000_000)

    # --- Download test labels CSV (small, ~90 KB) ----------------------------
    labels_zip = os.path.join(DOWNLOAD_DIR, "GTSRB_Test_GT.zip")
    _download_file(TEST_LABELS_URL, labels_zip, expected_min_size=50_000)

    # --- Extract training data ------------------------------------------------
    train_extract = os.path.join(DOWNLOAD_DIR, "train")
    _extract_zip(train_zip, train_extract)

    labels_extract = os.path.join(DOWNLOAD_DIR, "labels")
    _extract_zip(labels_zip, labels_extract)

    # --- Locate training directory --------------------------------------------
    # Training images live under GTSRB/Final_Training/Images/<class_id>/
    train_root = os.path.join(
        train_extract, "GTSRB", "Final_Training", "Images"
    )

    # --- Use LOCAL test images (already on disk) ------------------------------
    test_root = LOCAL_TEST_DIR
    if os.path.isdir(test_root):
        num_test = len([f for f in os.listdir(test_root) if f.endswith('.png')])
        print(f"  ✓ Using local test images: {test_root} ({num_test} files)")
    else:
        print(f"  ⚠ Local test images not found at {test_root}")

    # --- Locate test labels CSV -----------------------------------------------
    test_csv = os.path.join(labels_extract, "GT-final_test.csv")
    # Search subdirectories if not found at root level
    if not os.path.isfile(test_csv) or os.path.getsize(test_csv) == 0:
        for root, _dirs, files in os.walk(labels_extract):
            for f in files:
                if f.lower().endswith(".csv"):
                    candidate = os.path.join(root, f)
                    if os.path.getsize(candidate) > 0:
                        test_csv = candidate
                        break

    return train_root, test_root, test_csv


def _load_ppm_image(path: str, target_size: tuple = (IMG_HEIGHT, IMG_WIDTH)):
    """
    Load a PPM (or any Pillow-supported) image, convert to RGB, resize,
    and return as a uint8 numpy array of shape (H, W, 3).

    PPM is an uncompressed image format used by the GTSRB dataset.
    Pillow handles it natively, so no special codec is needed.
    """
    img = Image.open(path).convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def load_training_data(train_root: str):
    """
    Walk the GTSRB training directory structure:

        train_root/<ClassID_00000..00042>/*.ppm

    Each sub-folder name (zero-padded to 5 digits) IS the class label.

    Returns
    -------
    images : np.ndarray, shape (N, 128, 128, 3), dtype uint8
    labels : np.ndarray, shape (N,), dtype int32
    """
    images, labels = [], []
    print("\n📂 Loading training images …")

    for class_id in range(NUM_CLASSES):
        class_dir = os.path.join(train_root, f"{class_id:05d}")
        if not os.path.isdir(class_dir):
            print(f"  ⚠ Missing class directory: {class_dir}")
            continue

        # Collect all PPM files in this class folder
        ppm_files = sorted(glob.glob(os.path.join(class_dir, "*.ppm")))
        for ppm_path in ppm_files:
            img = _load_ppm_image(ppm_path)
            images.append(img)
            labels.append(class_id)

        # Progress feedback
        print(
            f"  Class {class_id:02d} ({CLASS_LABELS[class_id][:30]:30s}): "
            f"{len(ppm_files):5d} images"
        )

    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int32)
    print(f"  ✅ Total training images loaded: {len(images)}")
    return images, labels


def load_test_data(test_root: str, test_csv: str):
    """
    Load the official GTSRB test set.

    Supports two scenarios:
      A) CSV has labels → use CSV (Filename;...;ClassId, semicolon-delimited)
      B) CSV is empty/missing → use sorted PNG filenames with a fallback
         mapping from the downloaded labels archive.

    The local test_images/Test/ folder contains PNG files (00000.png - 12629.png)
    which correspond to the official GTSRB test images.

    Returns
    -------
    images : np.ndarray, shape (N, 128, 128, 3), dtype uint8
    labels : np.ndarray, shape (N,), dtype int32
    """
    images, labels = [], []
    print("\n📂 Loading test images …")

    # Try to read the labels CSV
    rows = []
    if os.path.isfile(test_csv) and os.path.getsize(test_csv) > 0:
        with open(test_csv, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            rows = list(reader)
        print(f"  ✓ Labels CSV loaded: {len(rows)} entries")

    if rows:
        # --- Path A: CSV has label data ---
        for row in rows:
            filename = row["Filename"].strip()
            class_id = int(row["ClassId"].strip())

            # Try the filename as-is first (PPM or PNG)
            img_path = os.path.join(test_root, filename)
            if not os.path.isfile(img_path):
                # The local folder has PNGs instead of PPMs — try swapping ext
                base_name = os.path.splitext(filename)[0]
                img_path = os.path.join(test_root, base_name + ".png")
            if not os.path.isfile(img_path):
                continue  # skip missing files gracefully

            img = _load_ppm_image(img_path)  # works for PNG too via Pillow
            images.append(img)
            labels.append(class_id)
    else:
        # --- Path B: No CSV labels — inform the user ---
        print("  ⚠ Test labels CSV is empty or missing.")
        print("    Test images will be loaded but evaluation metrics will")
        print("    only be available if labels are provided.")
        print("    → The labels will be downloaded from the GTSRB archive.")
        # If we reach here, the labels zip didn't contain valid data.
        # We'll still load images for inference testing.
        png_files = sorted(
            f for f in os.listdir(test_root)
            if f.lower().endswith((".png", ".jpg", ".ppm"))
        )
        print(f"  📂 Found {len(png_files)} test image files")
        for fname in png_files:
            img_path = os.path.join(test_root, fname)
            img = _load_ppm_image(img_path)
            images.append(img)
            labels.append(0)  # placeholder — will be overwritten if labels found

    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int32)
    print(f"  ✅ Total test images loaded: {len(images)}")
    return images, labels


def prepare_splits(train_root, test_root, test_csv):
    """
    Orchestrate data loading.  Checks for cached .npy files first to avoid
    re-parsing thousands of PPM images on every run.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test
        All images are uint8 arrays of shape (N, 128, 128, 3).
        All labels are int32 arrays of shape (N,).
    """
    # Paths for the numpy cache
    cache = {
        "X_train": os.path.join(DATA_DIR, "X_train.npy"),
        "y_train": os.path.join(DATA_DIR, "y_train.npy"),
        "X_val": os.path.join(DATA_DIR, "X_val.npy"),
        "y_val": os.path.join(DATA_DIR, "y_val.npy"),
        "X_test": os.path.join(DATA_DIR, "X_test.npy"),
        "y_test": os.path.join(DATA_DIR, "y_test.npy"),
    }

    # If ALL cache files exist, load from disk and return immediately
    if all(os.path.isfile(p) for p in cache.values()):
        print("\n⚡ Loading cached numpy arrays from data/ …")
        X_train = np.load(cache["X_train"])
        y_train = np.load(cache["y_train"])
        X_val = np.load(cache["X_val"])
        y_val = np.load(cache["y_val"])
        X_test = np.load(cache["X_test"])
        y_test = np.load(cache["y_test"])
        print(
            f"  Train: {len(X_train)} | Val: {len(X_val)} | "
            f"Test: {len(X_test)}"
        )
        return X_train, y_train, X_val, y_val, X_test, y_test

    # --- Load raw images -------------------------------------------------------
    images, labels = load_training_data(train_root)
    X_test, y_test = load_test_data(test_root, test_csv)

    # --- Train / Validation split (stratified) ---------------------------------
    # Stratification ensures every class is represented proportionally in both
    # the training and validation sets — critical for imbalanced datasets.
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels,
        test_size=VAL_SPLIT,
        random_state=42,
        stratify=labels,
    )

    print(
        f"\n📊 Split sizes → Train: {len(X_train)} | Val: {len(X_val)} | "
        f"Test: {len(X_test)}"
    )

    # --- Save to cache ---------------------------------------------------------
    _ensure_dir(DATA_DIR)
    np.save(cache["X_train"], X_train)
    np.save(cache["y_train"], y_train)
    np.save(cache["X_val"], X_val)
    np.save(cache["y_val"], y_val)
    np.save(cache["X_test"], X_test)
    np.save(cache["y_test"], y_test)
    print("  💾 Cached processed arrays to data/")

    return X_train, y_train, X_val, y_val, X_test, y_test


# ──────────────────────────────────────────────────────────────────────────────
# 4. DATA AUGMENTATION (Albumentations)
# ──────────────────────────────────────────────────────────────────────────────
#
# CRITICAL DESIGN DECISION:
#   We do NOT use HorizontalFlip or VerticalFlip because flipping traffic
#   signs can change their semantic meaning.  For example, "Keep Left"
#   becomes "Keep Right" when flipped horizontally.  This is a domain-
#   specific constraint that generic augmentation pipelines would miss.
# ──────────────────────────────────────────────────────────────────────────────


def build_augmentation_pipeline():
    """
    Build and return the Albumentations augmentation pipeline for training
    images.  The pipeline is designed specifically for traffic signs:

    • ShiftScaleRotate  — simulates camera jitter & distance variation
    • BrightnessContrast — handles different lighting conditions
    • HueSaturationValue — models colour shifts from weather / camera
    • CLAHE              — local contrast enhancement (fog, shadow)
    • Blur / Noise       — simulates motion blur & sensor noise
    • CoarseDropout      — occlusion robustness (Cutout-style)
    • Resize             — ensure consistent input dimensions
    """
    transform = A.Compose([
        # --- Geometric augmentations ---
        A.ShiftScaleRotate(
            shift_limit=0.1,      # shift image by up to 10% of dimensions
            scale_limit=0.15,     # zoom in/out by up to 15%
            rotate_limit=15,      # rotate by up to ±15 degrees
            border_mode=0,        # pad with zeros when shifting
            p=0.5,                # apply to 50% of images
        ),

        # --- Photometric augmentations ---
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.3,
        ),
        A.CLAHE(
            clip_limit=2.0,       # contrast-limited adaptive histogram eq.
            p=0.3,
        ),

        # --- Blur and noise (one-of to avoid stacking) ---
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
            A.GaussNoise(p=1.0),
        ], p=0.3),

        # --- Cutout-style occlusion ---
        A.CoarseDropout(
            max_holes=4,
            max_height=16,
            max_width=16,
            fill_value=0,         # fill dropped regions with black
            p=0.2,
        ),

        # --- Final resize (should already be 128×128, but ensure it) ---
        A.Resize(IMG_HEIGHT, IMG_WIDTH),
    ])
    return transform


def build_val_pipeline():
    """
    Validation / test pipeline: resize only, no augmentation.
    """
    return A.Compose([
        A.Resize(IMG_HEIGHT, IMG_WIDTH),
    ])


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalise a uint8 image to float32 using MobileNetV2's native
    preprocessing: pixel_value / 127.5 - 1.0, scaling to [-1, 1].

    CRITICAL: This MUST match the preprocessing used during MobileNetV2's
    original ImageNet training AND the backend inference endpoint.
    Using a different normalization (e.g. ImageNet mean/std) would cause
    a severe distribution mismatch and dramatically hurt accuracy.
    """
    img = image.astype(np.float32)
    # MobileNetV2's preprocess_input: x / 127.5 - 1.0 → range [-1, 1]
    img = mobilenet_preprocess(img)
    return img


# ──────────────────────────────────────────────────────────────────────────────
# 5. tf.data DATASET CONSTRUCTION
# ──────────────────────────────────────────────────────────────────────────────
#
# We wrap augmentation + normalisation inside tf.data pipelines for
# efficient, parallelised preprocessing during training.
# ──────────────────────────────────────────────────────────────────────────────


def _augment_and_normalize(image, label, augment_fn):
    """
    Apply Albumentations augmentation followed by MobileNetV2 normalisation.

    Because Albumentations is a pure-Python / NumPy library, we use
    tf.numpy_function to bridge it into the tf.data pipeline.
    """
    def _apply(img):
        # img arrives as a uint8 numpy array (H, W, 3)
        augmented = augment_fn(image=img)["image"]
        return normalize_image(augmented)

    img = tf.numpy_function(_apply, [image], tf.float32)
    img.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])  # restore static shape info
    return img, label


def build_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    augment: bool = False,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from numpy arrays.

    Parameters
    ----------
    images   : uint8 array (N, H, W, 3)
    labels   : int32 array (N,) — will be one-hot encoded internally
    augment  : whether to apply data augmentation (training only)
    batch_size : mini-batch size
    shuffle  : whether to shuffle each epoch (training only)

    Returns
    -------
    tf.data.Dataset yielding (image_batch, label_batch)
    """
    # One-hot encode labels for categorical cross-entropy
    labels_onehot = to_categorical(labels, num_classes=NUM_CLASSES)

    ds = tf.data.Dataset.from_tensor_slices((images, labels_onehot))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(images), 10000), seed=42)

    # Select augmentation vs. plain pipeline
    aug_fn = build_augmentation_pipeline() if augment else build_val_pipeline()
    ds = ds.map(
        lambda img, lbl: _augment_and_normalize(img, lbl, aug_fn),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)  # overlap I/O with compute
    return ds


# ──────────────────────────────────────────────────────────────────────────────
# 6. MODEL ARCHITECTURE — MobileNetV2 + Custom Head
# ──────────────────────────────────────────────────────────────────────────────


def build_model(freeze_base: bool = True) -> keras.Model:
    """
    Construct the MobileNetV2 transfer-learning model.

    Architecture
    ------------
    Input (128, 128, 3)
        → MobileNetV2 backbone (ImageNet weights, no top)
        → GlobalAveragePooling2D
        → Dropout(0.3)
        → Dense(128, ReLU)
        → Dropout(0.3)
        → Dense(43, Softmax)

    Parameters
    ----------
    freeze_base : bool
        If True, all backbone layers are frozen (Phase 1).
        If False, the top 30 layers are trainable (Phase 2).
    """
    # --- Backbone ---
    base_model = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,        # discard ImageNet classification head
        weights="imagenet",       # use pre-trained weights
    )

    # Freeze / unfreeze base layers
    if freeze_base:
        # Phase 1: freeze everything
        base_model.trainable = False
        print("🔒 Base model frozen (Phase 1 — feature extraction)")
    else:
        # Phase 2: unfreeze the top 30 layers for fine-tuning
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        trainable_count = sum(
            1 for l in base_model.layers if l.trainable
        )
        print(
            f"🔓 Top {trainable_count} base layers unfrozen "
            f"(Phase 2 — fine-tuning)"
        )

    # --- Classification head ---
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = base_model(inputs, training=False)     # keep BN in inference mode
    x = layers.GlobalAveragePooling2D()(x)     # spatial dims → single vector
    x = layers.Dropout(0.3)(x)                 # regularise
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="GTSRB_MobileNetV2")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 7. CLASS WEIGHTS (handle imbalanced dataset)
# ──────────────────────────────────────────────────────────────────────────────


def compute_weights(y_train: np.ndarray) -> dict:
    """
    Compute per-class weights using sklearn's 'balanced' strategy.

    Classes with fewer samples receive higher weight, so the loss function
    penalises mistakes on rare classes more heavily.  This is important
    for GTSRB because some signs (e.g. speed limit 20 km/h) have far fewer
    samples than others (e.g. speed limit 50 km/h).

    Returns
    -------
    dict mapping class_index (int) → weight (float)
    """
    unique_classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=y_train,
    )
    class_weight_dict = {int(c): float(w) for c, w in zip(unique_classes, weights)}

    # Print a summary of extreme weights
    min_w = min(class_weight_dict.values())
    max_w = max(class_weight_dict.values())
    print(f"⚖️  Class weights computed — range: [{min_w:.3f}, {max_w:.3f}]")
    return class_weight_dict


# ──────────────────────────────────────────────────────────────────────────────
# 8. CALLBACKS
# ──────────────────────────────────────────────────────────────────────────────


def build_callbacks() -> list:
    """
    Build the list of Keras callbacks used during training.

    • ReduceLROnPlateau — halve LR if val_loss plateaus for 3 epochs
    • EarlyStopping     — stop if val_loss doesn't improve for 7 epochs
    • ModelCheckpoint   — save the best model (by val_loss) to disk
    """
    _ensure_dir(MODEL_DIR)

    cb_list = [
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,           # new_lr = old_lr × 0.5
            patience=3,           # wait 3 epochs before reducing
            min_lr=1e-7,          # lower bound on learning rate
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,           # stop after 7 epochs with no improvement
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]
    return cb_list


# ──────────────────────────────────────────────────────────────────────────────
# 9. TRAINING — Two-Phase Pipeline
# ──────────────────────────────────────────────────────────────────────────────


def train(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weights: dict,
    phase: int,
    epochs: int,
    learning_rate: float,
):
    """
    Compile and train the model for one phase.

    Uses CategoricalFocalCrossentropy loss, which down-weights well-classified
    examples and focuses on hard / misclassified ones.  Combined with class
    weights, this gives a double layer of protection against class imbalance.

    Parameters
    ----------
    phase         : 1 (feature extraction) or 2 (fine-tuning)
    epochs        : number of epochs to train
    learning_rate : initial learning rate for the Adam optimiser
    """
    print(f"\n{'='*70}")
    print(f"  PHASE {phase}: {'Feature Extraction' if phase == 1 else 'Fine-Tuning'}")
    print(f"  Epochs: {epochs} | LR: {learning_rate} | Batch: {BATCH_SIZE}")
    print(f"{'='*70}\n")

    # Compile with focal loss — alpha weights positive vs. negative,
    # gamma reduces loss for well-classified examples.
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.CategoricalFocalCrossentropy(
            alpha=0.25,           # balancing factor
            gamma=2.0,            # focusing parameter
        ),
        metrics=["accuracy"],
    )

    # Print a compact model summary
    model.summary(print_fn=lambda x: print(f"  {x}"))

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=build_callbacks(),
        verbose=1,
    )
    return history


# ──────────────────────────────────────────────────────────────────────────────
# 10. EVALUATION
# ──────────────────────────────────────────────────────────────────────────────


def evaluate_model(model: keras.Model, test_ds: tf.data.Dataset, y_test: np.ndarray):
    """
    Run the model on the test set and print comprehensive metrics:
        • Overall test accuracy
        • Per-class precision, recall, F1-score (classification report)
        • Per-class F1 scores (sorted worst → best for easy debugging)
        • Confusion matrix summary (top misclassifications)
    """
    print(f"\n{'='*70}")
    print("  EVALUATION ON TEST SET")
    print(f"{'='*70}\n")

    # --- Overall accuracy ------------------------------------------------------
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}  ({test_acc*100:.2f}%)\n")

    # --- Generate predictions ---------------------------------------------------
    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # --- Classification report --------------------------------------------------
    target_names = [CLASS_LABELS[i] for i in range(NUM_CLASSES)]
    report = classification_report(
        y_test, y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    print("📋 Classification Report:\n")
    print(report)

    # --- Per-class F1 scores (sorted) -------------------------------------------
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    print("\n📊 Per-Class F1 Scores (sorted ascending):\n")
    sorted_indices = np.argsort(f1_per_class)
    for idx in sorted_indices:
        bar = "█" * int(f1_per_class[idx] * 40)  # visual bar (max 40 chars)
        print(
            f"  Class {idx:2d} | F1={f1_per_class[idx]:.4f} | "
            f"{bar:40s} | {CLASS_LABELS[idx]}"
        )

    # --- Confusion matrix summary -----------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🔀 Confusion Matrix Summary (top misclassifications):\n")
    print(f"  {'True Class':>35s}  →  {'Predicted As':35s}   Count")
    print(f"  {'─'*35}     {'─'*35}   ─────")

    # Find off-diagonal entries, sort by count
    misclass = []
    for true_cls in range(NUM_CLASSES):
        for pred_cls in range(NUM_CLASSES):
            if true_cls != pred_cls and cm[true_cls, pred_cls] > 0:
                misclass.append((cm[true_cls, pred_cls], true_cls, pred_cls))
    misclass.sort(reverse=True)

    # Print top-15 misclassifications
    for count, true_cls, pred_cls in misclass[:15]:
        true_name = CLASS_LABELS[true_cls][:35]
        pred_name = CLASS_LABELS[pred_cls][:35]
        print(f"  {true_name:>35s}  →  {pred_name:35s}   {count:5d}")

    print(f"\n  Macro-average F1: {np.mean(f1_per_class):.4f}")
    print(f"  Weighted-avg F1:  "
          f"{f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")

    return test_acc


# ──────────────────────────────────────────────────────────────────────────────
# 11. MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────


def main():
    """
    End-to-end pipeline:
        1. Download & prepare data
        2. Build tf.data datasets with augmentation
        3. Build model
        4. Phase 1: train with frozen backbone
        5. Phase 2: fine-tune top-30 layers
        6. Evaluate on test set
        7. Save final model
    """
    print("=" * 70)
    print("  TRAFFIC SIGN RECOGNITION — GTSRB TRAINING PIPELINE")
    print("=" * 70)

    # Check GPU availability (including Apple Metal on M-series Macs)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        gpu_names = [g.name for g in gpus]
        is_metal = any("GPU" in g.name for g in gpus) and sys.platform == "darwin"
        if is_metal:
            print(f"\n🍎 Apple Metal GPU detected: {gpu_names}")
            print("   Using tensorflow-metal for hardware acceleration.")
        else:
            print(f"\n🖥️  GPU(s) detected: {gpu_names}")
        # Allow memory growth to avoid OOM on shared machines
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass  # Metal GPU doesn't always support this — safe to skip
    else:
        print("\n⚠️  No GPU detected — training will use CPU (slower).")
        print("   💡 Tip: Install 'tensorflow-metal' for Apple M-series GPU acceleration.")
        print("      pip install tensorflow-metal")

    # ── Step 1: Data Ingestion ─────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  STEP 1: Data Ingestion")
    print("─" * 70)

    train_root, test_root, test_csv = download_and_extract()
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_splits(
        train_root, test_root, test_csv
    )

    # Print class distribution for the training set
    print("\n📊 Training set class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls_id, cnt in zip(unique, counts):
        print(f"  Class {cls_id:2d}: {cnt:5d} samples — {CLASS_LABELS[cls_id]}")

    # ── Step 2: Build tf.data Datasets ─────────────────────────────────────
    print("\n" + "─" * 70)
    print("  STEP 2: Building tf.data Datasets")
    print("─" * 70)

    train_ds = build_dataset(
        X_train, y_train, augment=True, shuffle=True
    )
    val_ds = build_dataset(
        X_val, y_val, augment=False, shuffle=False
    )
    test_ds = build_dataset(
        X_test, y_test, augment=False, shuffle=False
    )
    print("  ✅ Datasets built with augmentation pipeline.")

    # ── Step 3: Compute Class Weights ──────────────────────────────────────
    print("\n" + "─" * 70)
    print("  STEP 3: Computing Class Weights")
    print("─" * 70)

    class_weights = compute_weights(y_train)

    # ── Step 4: Phase 1 — Feature Extraction ───────────────────────────────
    print("\n" + "─" * 70)
    print("  STEP 4: Phase 1 Training (Feature Extraction)")
    print("─" * 70)

    model = build_model(freeze_base=True)

    history_p1 = train(
        model, train_ds, val_ds, class_weights,
        phase=1, epochs=PHASE1_EPOCHS, learning_rate=PHASE1_LR,
    )

    # ── Step 5: Phase 2 — Fine-Tuning ─────────────────────────────────────
    print("\n" + "─" * 70)
    print("  STEP 5: Phase 2 Training (Fine-Tuning)")
    print("─" * 70)

    # Rebuild model with top-30 layers unfrozen.
    # We rebuild from the *same* model object so trained weights carry over.
    base = model.layers[1]  # the MobileNetV2 layer
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    trainable_count = sum(1 for l in base.layers if l.trainable)
    print(f"🔓 Unfroze top {trainable_count} layers of MobileNetV2 backbone")

    history_p2 = train(
        model, train_ds, val_ds, class_weights,
        phase=2, epochs=PHASE2_EPOCHS, learning_rate=PHASE2_LR,
    )

    # ── Step 6: Evaluation ─────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  STEP 6: Evaluation")
    print("─" * 70)

    test_acc = evaluate_model(model, test_ds, y_test)

    # ── Step 7: Save Final Model ───────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  STEP 7: Saving Final Model")
    print("─" * 70)

    _ensure_dir(MODEL_DIR)
    model.save(MODEL_PATH)
    print(f"  💾 Model saved to: {MODEL_PATH}")
    print(f"  📏 Model file size: "
          f"{os.path.getsize(MODEL_PATH) / (1024*1024):.1f} MB")

    # ── Done! ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  ✅ TRAINING COMPLETE — Test Accuracy: {test_acc*100:.2f}%")
    print(f"  📂 Model saved to: {MODEL_PATH}")
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
