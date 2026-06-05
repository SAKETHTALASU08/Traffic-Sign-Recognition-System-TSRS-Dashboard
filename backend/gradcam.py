"""
gradcam.py — Gradient-weighted Class Activation Mapping (Grad-CAM)
===================================================================

Grad-CAM produces a "visual explanation" for decisions from CNN-based models.
It uses the gradients flowing into the last convolutional layer to produce a
coarse localisation map highlighting the important regions in the image.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017.

This module provides three functions:
  1. make_gradcam_heatmap  — raw heatmap generation (numpy array)
  2. overlay_gradcam       — blend heatmap onto the original image
  3. generate_gradcam_base64 — end-to-end pipeline returning a base64 PNG

Architecture Note:
  Our model is a tf.keras.Sequential that wraps a MobileNetV2 base with a
  custom classification head. The last convolutional layer we target is
  'Conv_1' inside the MobileNetV2 base model, NOT in the Sequential wrapper.
  The code handles both cases (direct Functional model and Sequential wrapper).
"""

import base64
import io
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ---------------------------------------------------------------------------
# 1. Raw Heatmap Generation
# ---------------------------------------------------------------------------

def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str = "Conv_1",
    pred_index: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for the given input image.

    Parameters
    ----------
    img_array : np.ndarray
        Preprocessed image tensor of shape (1, H, W, 3).
    model : tf.keras.Model
        The trained classification model.
    last_conv_layer_name : str
        Name of the last convolutional layer to use for Grad-CAM.
        For MobileNetV2, this is typically 'Conv_1' (the final 1×1 conv
        that outputs 1280 channels before global average pooling).
    pred_index : int or None
        Index of the target class. If None, uses the argmax (top prediction).

    Returns
    -------
    np.ndarray
        2D heatmap array of shape (conv_height, conv_width), values in [0, 1].

    Implementation Details
    ----------------------
    The key idea is:
      1. Build a sub-model that outputs BOTH the last conv layer's activations
         AND the final prediction.
      2. Use tf.GradientTape to compute the gradient of the target class score
         with respect to the last conv layer's output feature maps.
      3. Global-average-pool those gradients to get per-channel importance weights.
      4. Compute a weighted sum of the feature maps → the raw heatmap.
      5. Apply ReLU (we only care about features that have a POSITIVE influence
         on the predicted class) and normalise to [0, 1].
    """
    # -----------------------------------------------------------------------
    # Step 0: Locate the feature map layer.
    # -----------------------------------------------------------------------
    # Our model wraps MobileNetV2. The output of the MobileNetV2 base model
    # (which is layer index 1) is a 4x4x1280 feature map. This is exactly what
    # we need for Grad-CAM. Digging inside the base model breaks the TensorFlow
    # graph, so we just use the base model's output tensor directly.
    base_model_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or layer.name.startswith("mobilenetv2"):
            base_model_layer = layer
            break

    if base_model_layer is None:
        raise ValueError("Could not find the MobileNetV2 base model layer.")

    # -----------------------------------------------------------------------
    # Step 1: Build a "grad model" that outputs the conv activations + preds.
    # -----------------------------------------------------------------------
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[base_model_layer.output, model.output],
    )

    # -----------------------------------------------------------------------
    # Step 2: Compute gradients of the target class w.r.t. conv outputs.
    # -----------------------------------------------------------------------
    with tf.GradientTape() as tape:
        # Forward pass through the grad model.
        conv_outputs, predictions = grad_model(img_array)

        # If no specific class is targeted, use the model's top prediction.
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        # Extract the score for the target class.
        # This is the value we'll differentiate with respect to conv_outputs.
        class_score = predictions[:, pred_index]

    # Backpropagate to get gradients of class_score w.r.t. conv feature maps.
    grads = tape.gradient(class_score, conv_outputs)

    # -----------------------------------------------------------------------
    # Step 3: Global Average Pooling of gradients → per-channel weights.
    # -----------------------------------------------------------------------
    # grads shape: (1, conv_h, conv_w, num_channels)
    # We average over the spatial dimensions (axes 0, 1, 2 — batch, height, width)
    # to get a single importance weight per channel.
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # -----------------------------------------------------------------------
    # Step 4: Weighted combination of feature maps.
    # -----------------------------------------------------------------------
    # Multiply each channel's feature map by its importance weight,
    # then sum across channels to get the 2D heatmap.
    conv_outputs = conv_outputs[0]  # Remove batch dimension → (h, w, channels)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (h, w, 1)
    heatmap = tf.squeeze(heatmap)  # (h, w)

    # -----------------------------------------------------------------------
    # Step 5: ReLU + normalise to [0, 1].
    # -----------------------------------------------------------------------
    # ReLU: keep only positive contributions (features that increase the
    # target class score). Negative values would indicate features that
    # *decrease* the class score — not useful for visualisation.
    heatmap = tf.maximum(heatmap, 0)

    # Normalise to [0, 1] for visualisation. Guard against division by zero
    # (can happen if the model is very confident and gradients are tiny).
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


# ---------------------------------------------------------------------------
# 2. Heatmap Overlay
# ---------------------------------------------------------------------------

def overlay_gradcam(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay the Grad-CAM heatmap on the original image.

    Parameters
    ----------
    original_image : np.ndarray
        Original RGB image as uint8 array, shape (H, W, 3).
    heatmap : np.ndarray
        2D heatmap from make_gradcam_heatmap, values in [0, 1].
    alpha : float
        Blending factor. 0.0 = only original, 1.0 = only heatmap.
        0.4 is a good default that lets you see both the sign and the
        activation regions clearly.

    Returns
    -------
    np.ndarray
        Blended RGB image as uint8 array, shape (H, W, 3).

    Process
    -------
    1. Rescale heatmap to 0–255 and convert to uint8.
    2. Resize heatmap to match the original image dimensions.
    3. Apply a JET colourmap (blue → green → red) for intuitive
       visualisation (red = high activation = important region).
    4. Convert JET image from BGR (OpenCV default) to RGB.
    5. Alpha-blend the colourised heatmap with the original image.
    """
    # Scale heatmap to 0–255 for colourmap application.
    heatmap_uint8 = np.uint8(255 * heatmap)

    # Resize the heatmap to match the original image's spatial dimensions.
    # INTER_LINEAR gives smooth upscaling (the heatmap is typically 4×4 or 7×7).
    heatmap_resized = cv2.resize(
        heatmap_uint8,
        (original_image.shape[1], original_image.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    # Apply JET colourmap: cold (blue) → warm (red).
    # OpenCV returns BGR, so we convert to RGB to match our image format.
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Alpha-blend: result = alpha * heatmap + (1 - alpha) * original.
    # cv2.addWeighted handles clipping to [0, 255] automatically.
    superimposed = cv2.addWeighted(
        heatmap_colored, alpha,
        original_image, 1 - alpha,
        0,  # gamma (brightness offset) — we don't need it
    )

    return superimposed


# ---------------------------------------------------------------------------
# 3. End-to-End Pipeline
# ---------------------------------------------------------------------------

def generate_gradcam_base64(
    image_bytes: bytes,
    model: tf.keras.Model,
    target_size: tuple[int, int] = (128, 128),
) -> dict:
    """
    Complete Grad-CAM pipeline: raw bytes → base64-encoded overlay image.

    This is the main function called by the /explain endpoint. It:
      1. Decodes the uploaded image bytes into a PIL Image.
      2. Resizes to the model's expected input size.
      3. Preprocesses using MobileNetV2's preprocess_input.
      4. Runs inference to determine the predicted class.
      5. Generates the Grad-CAM heatmap for that class.
      6. Overlays the heatmap on the original (resized) image.
      7. Encodes the result as a base64 PNG string for easy transport
         to the React frontend (can be used directly in an <img> src).

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of the uploaded image file.
    model : tf.keras.Model
        The loaded traffic sign classification model.
    target_size : tuple[int, int]
        Spatial dimensions the model expects, default (128, 128).

    Returns
    -------
    dict
        {
            "class_name": str,          # predicted sign name
            "confidence": float,        # prediction confidence (0–1)
            "gradcam_image": str,       # base64-encoded PNG of the overlay
            "pred_index": int,          # numeric class index
        }
    """
    # Import CLASS_NAMES here to avoid circular imports at module level.
    # main.py defines CLASS_NAMES; we import lazily.
    from backend.main import CLASS_NAMES

    # ------------------------------------------------------------------
    # A. Decode and resize the image.
    # ------------------------------------------------------------------
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pil_image = pil_image.resize(target_size, Image.LANCZOS)

    # Keep a copy for overlay (uint8, 0–255, RGB).
    original_for_overlay = np.array(pil_image)

    # ------------------------------------------------------------------
    # B. Preprocess for MobileNetV2 inference.
    # ------------------------------------------------------------------
    # MobileNetV2's preprocess_input expects float32 and scales pixels
    # from [0, 255] to [-1, 1].
    img_array = np.array(pil_image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension → (1, H, W, 3)
    img_array = preprocess_input(img_array)

    # ------------------------------------------------------------------
    # C. Run inference.
    # ------------------------------------------------------------------
    predictions = model.predict(img_array, verbose=0)
    pred_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][pred_index])
    class_name = CLASS_NAMES.get(pred_index, f"Unknown ({pred_index})")

    # ------------------------------------------------------------------
    # D. Generate Grad-CAM heatmap.
    # ------------------------------------------------------------------
    heatmap = make_gradcam_heatmap(
        img_array,
        model,
        last_conv_layer_name="Conv_1",
        pred_index=pred_index,
    )

    # ------------------------------------------------------------------
    # E. Overlay heatmap on the original image.
    # ------------------------------------------------------------------
    superimposed = overlay_gradcam(original_for_overlay, heatmap, alpha=0.4)

    # ------------------------------------------------------------------
    # F. Encode the overlay as a base64 PNG string.
    # ------------------------------------------------------------------
    # We use PNG (lossless) to preserve heatmap colour fidelity.
    overlay_pil = Image.fromarray(superimposed)
    buffer = io.BytesIO()
    overlay_pil.save(buffer, format="PNG")
    buffer.seek(0)
    gradcam_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return {
        "class_name": class_name,
        "confidence": confidence,
        "gradcam_image": gradcam_base64,
        "pred_index": pred_index,
    }
