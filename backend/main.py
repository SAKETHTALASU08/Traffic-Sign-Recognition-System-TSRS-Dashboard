"""
main.py — FastAPI Backend for Traffic Sign Recognition System
==============================================================

This is the main entry point for the backend API. It serves a fine-tuned
MobileNetV2 model trained on the German Traffic Sign Recognition Benchmark
(GTSRB) dataset (43 classes).

Architecture Decisions:
  - FastAPI with the modern "lifespan" pattern (replaces deprecated
    @app.on_event startup/shutdown hooks).
  - Synchronous (def) endpoints for TensorFlow inference. TensorFlow's
    eager execution is NOT async-safe — calling model.predict() inside
    an async def would block the event loop. FastAPI automatically runs
    sync endpoints in a thread pool, which is exactly what we want.
  - SQLite for prediction history — lightweight, zero-config, and perfect
    for a single-server demo/production deployment.
  - CORS configured for the React/Vite frontend dev server.

Run with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

import base64
import io
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from backend import database
from backend import gradcam

# ---------------------------------------------------------------------------
# GTSRB Class Labels (43 classes)
# ---------------------------------------------------------------------------
# These map the model's output index (0–42) to human-readable sign names.
# The names match the official GTSRB dataset labels.
CLASS_NAMES: dict[int, str] = {
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

# ---------------------------------------------------------------------------
# Model Storage
# ---------------------------------------------------------------------------
# Global dictionary to hold the loaded ML model(s). Using a dict makes it
# easy to extend to multiple models in the future (e.g., an ensemble).
ml_models: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Application Lifespan (Startup / Shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's startup and shutdown lifecycle.

    Startup:
      1. Initialize the SQLite database (create tables if needed).
      2. Load the TensorFlow/Keras model into memory.
         - The model file is expected at 'model/gtsrb_mobilenet.h5'
           relative to the working directory.
         - We load eagerly and store in the global ml_models dict so
           that all request-handler threads can share the same model
           instance (Keras models are thread-safe for predict()).

    Shutdown:
      1. Clear the model from memory.
      2. (Optional) Clear TF session to free GPU memory.
    """
    # --- STARTUP ---
    print("[lifespan] Starting up...")

    # 1. Initialize database schema.
    database.init_db()

    # 2. Load the traffic sign classification model.
    model_path = "model/gtsrb_mobilenet.h5"
    try:
        # compile=False because we only need inference, not training.
        # This avoids warnings about missing optimizer state.
        ml_models["traffic_sign"] = tf.keras.models.load_model(
            model_path, compile=False
        )
        print(f"[lifespan] Model loaded successfully from '{model_path}'")
        # Print model summary to logs for debugging shape mismatches.
        ml_models["traffic_sign"].summary(print_fn=lambda x: print(f"  {x}"))
    except Exception as exc:
        # Don't crash the server if the model file is missing — the /health
        # endpoint will report model_loaded=false, and /predict will return
        # a clear error. This lets the frontend still function for history/analytics.
        print(f"[lifespan] WARNING: Failed to load model from '{model_path}': {exc}")
        ml_models["traffic_sign"] = None

    # Yield control to the application (it runs until shutdown is requested).
    yield

    # --- SHUTDOWN ---
    print("[lifespan] Shutting down...")
    ml_models.clear()
    # Clear TensorFlow's backend session to release any GPU memory.
    tf.keras.backend.clear_session()
    print("[lifespan] Cleanup complete.")


# ---------------------------------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Traffic Sign Recognition API",
    description=(
        "A production-ready API for classifying traffic signs using a "
        "MobileNetV2 model trained on the GTSRB dataset. Supports inference, "
        "Grad-CAM visual explanations, prediction history, and analytics."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS Middleware
# ---------------------------------------------------------------------------
# Allow the React frontend dev servers (Vite on 5173, CRA on 3000) to
# communicate with this API. In production, replace these with your actual
# domain(s) and remove the wildcard headers/methods.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server (default port)
        "http://localhost:3000",  # Create React App / alternative dev server
    ],
    allow_credentials=True,  # Allow cookies if needed for auth in the future
    allow_methods=["*"],      # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],      # Allow all headers (Content-Type, Authorization, etc.)
)


# ---------------------------------------------------------------------------
# Helper: Get the loaded model or raise HTTP 503
# ---------------------------------------------------------------------------

def _get_model() -> tf.keras.Model:
    """
    Retrieve the loaded TensorFlow model, or raise an HTTP 503 error
    if the model failed to load at startup.

    This centralises the "is model available?" check so that each endpoint
    doesn't have to repeat it.
    """
    model = ml_models.get("traffic_sign")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. Ensure 'model/gtsrb_mobilenet.h5' exists "
                "and the server was started correctly."
            ),
        )
    return model


# ---------------------------------------------------------------------------
# Helper: Preprocess an uploaded image for MobileNetV2 inference
# ---------------------------------------------------------------------------

def _preprocess_image(
    image_bytes: bytes,
    target_size: tuple[int, int] = (128, 128),
) -> tuple[np.ndarray, Image.Image]:
    """
    Decode, resize, and preprocess an image for MobileNetV2 inference.

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes from the uploaded file.
    target_size : tuple[int, int]
        Spatial dimensions the model expects.

    Returns
    -------
    tuple[np.ndarray, Image.Image]
        - Preprocessed numpy array of shape (1, H, W, 3) ready for model.predict().
        - The resized PIL Image (useful for creating thumbnails).
    """
    try:
        # Open and convert to RGB (handles RGBA PNGs, greyscale, etc.).
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file. Please upload a valid JPEG or PNG image.",
        )

    # Resize to the model's expected input dimensions using LANCZOS
    # (high-quality downsampling filter).
    pil_image = pil_image.resize(target_size, Image.LANCZOS)

    # Convert to float32 numpy array and add batch dimension.
    img_array = np.array(pil_image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # (1, 128, 128, 3)

    # Apply MobileNetV2-specific preprocessing:
    # Scales pixel values from [0, 255] to [-1, 1].
    img_array = preprocess_input(img_array)

    return img_array, pil_image


# ---------------------------------------------------------------------------
# Helper: Create a base64 thumbnail
# ---------------------------------------------------------------------------

def _create_thumbnail_base64(
    pil_image: Image.Image,
    thumbnail_size: tuple[int, int] = (64, 64),
) -> str:
    """
    Create a small JPEG thumbnail and return it as a base64 string.

    We use JPEG (not PNG) for thumbnails because:
      - Much smaller file sizes for photographic content.
      - These are only used in the history UI, so lossy compression is fine.
      - quality=85 gives a good balance of size vs visual quality.

    Parameters
    ----------
    pil_image : Image.Image
        The resized input image (already 128×128).
    thumbnail_size : tuple[int, int]
        Target thumbnail dimensions.

    Returns
    -------
    str
        Base64-encoded JPEG string (no data URI prefix — the frontend adds it).
    """
    thumbnail = pil_image.copy()
    thumbnail.thumbnail(thumbnail_size, Image.LANCZOS)

    buffer = io.BytesIO()
    thumbnail.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")


# ===========================================================================
# API Endpoints
# ===========================================================================


# ---------------------------------------------------------------------------
# POST /predict — Main inference endpoint
# ---------------------------------------------------------------------------

def predict(file: UploadFile = File(...)):
    """
    Classify an uploaded traffic sign image.

    This endpoint is intentionally synchronous (def, NOT async def) because
    TensorFlow's model.predict() is a blocking CPU/GPU operation. FastAPI
    will automatically run this function in a thread pool, preventing it
    from blocking the async event loop.

    Flow:
      1. Read and validate the uploaded image.
      2. Preprocess to (1, 128, 128, 3) float32 tensor.
      3. Run model.predict() and measure wall-clock latency.
      4. Extract predicted class, confidence, and all probabilities.
      5. Create a 64×64 thumbnail for history storage.
      6. Log the prediction to SQLite.
      7. Return comprehensive results as JSON.

    Request:
      - Content-Type: multipart/form-data
      - Body: file (image/jpeg or image/png)

    Response:
      {
          "class_id": 14,
          "class_name": "Stop",
          "confidence": 0.9876,
          "latency_ms": 45.3,
          "all_probabilities": { "Speed limit (20km/h)": 0.001, ... }
      }
    """
    # 1. Validate model availability.
    model = _get_model()

    # 2. Read the uploaded file bytes.
    image_bytes = file.file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    # 3. Preprocess the image.
    img_array, pil_image = _preprocess_image(image_bytes)

    # 4. Run inference with latency measurement.
    # time.perf_counter() provides the highest-resolution timer available,
    # which is important for measuring sub-millisecond operations.
    start_time = time.perf_counter()
    predictions = model.predict(img_array, verbose=0)  # verbose=0 suppresses progress bar
    end_time = time.perf_counter()

    # Calculate latency in milliseconds.
    latency_ms = round((end_time - start_time) * 1000, 2)

    # 5. Extract the prediction results.
    probabilities = predictions[0]  # Shape: (43,) — one probability per class
    class_id = int(np.argmax(probabilities))
    confidence = float(probabilities[class_id])
    class_name = CLASS_NAMES.get(class_id, f"Unknown ({class_id})")

    # 6. Build the all_probabilities dict (useful for bar charts in the frontend).
    # Round to 6 decimal places to keep the response payload reasonable.
    all_probabilities = {
        CLASS_NAMES.get(i, f"Class {i}"): round(float(probabilities[i]), 6)
        for i in range(len(probabilities))
    }

    # 7. Create a thumbnail for history storage.
    thumbnail_b64 = _create_thumbnail_base64(pil_image)

    # 8. Log to database (fire-and-forget — we don't need the row ID here).
    try:
        database.log_prediction(
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            latency_ms=latency_ms,
            thumbnail_base64=thumbnail_b64,
        )
    except Exception as db_err:
        # Database errors shouldn't fail the prediction response.
        # Log the error but still return the prediction to the user.
        print(f"[predict] WARNING: Failed to log prediction to DB: {db_err}")

    # 9. Return the response.
    return {
        "class_id": class_id,
        "class_name": class_name,
        "confidence": round(confidence, 4),
        "latency_ms": latency_ms,
        "all_probabilities": all_probabilities,
    }


# Register the predict function as a POST endpoint.
# We define the function first and register it separately for readability,
# but you can also use @app.post("/predict") as a decorator directly.
app.post(
    "/predict",
    summary="Classify a traffic sign image",
    response_description="Prediction results including class, confidence, and latency",
)(predict)


# ---------------------------------------------------------------------------
# POST /explain — Grad-CAM visual explanation
# ---------------------------------------------------------------------------

def explain(file: UploadFile = File(...)):
    """
    Generate a Grad-CAM explanation for a traffic sign classification.

    Grad-CAM (Gradient-weighted Class Activation Mapping) highlights which
    regions of the input image most influenced the model's prediction. This
    is crucial for building trust in the model — users can verify that the
    model is "looking at" the right part of the image (the sign itself, not
    background noise).

    This endpoint is synchronous for the same reason as /predict — TensorFlow
    operations block the thread.

    Request:
      - Content-Type: multipart/form-data
      - Body: file (image/jpeg or image/png)

    Response:
      {
          "class_name": "Stop",
          "confidence": 0.9876,
          "gradcam_image": "<base64-encoded PNG>",
          "pred_index": 14
      }
    """
    # Validate model availability.
    model = _get_model()

    # Read the uploaded image bytes.
    image_bytes = file.file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    # Delegate to the gradcam module's end-to-end pipeline.
    try:
        result = gradcam.generate_gradcam_base64(image_bytes, model)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Grad-CAM generation failed: {str(exc)}",
        )

    return {
        "class_name": result["class_name"],
        "confidence": round(result["confidence"], 4),
        "gradcam_image": result["gradcam_image"],
    }


app.post(
    "/explain",
    summary="Generate Grad-CAM explanation for a traffic sign image",
    response_description="Grad-CAM overlay image with prediction details",
)(explain)


# ---------------------------------------------------------------------------
# GET /history — Prediction history with pagination
# ---------------------------------------------------------------------------

@app.get(
    "/history",
    summary="Get prediction history",
    response_description="Paginated list of past predictions",
)
async def history(
    limit: int = Query(default=50, ge=1, le=500, description="Number of records per page"),
    offset: int = Query(default=0, ge=0, description="Number of records to skip"),
    sort_by: str = Query(default="timestamp", description="Column to sort by"),
    order: str = Query(default="desc", description="Sort order: 'asc' or 'desc'"),
):
    """
    Retrieve paginated prediction history from the database.

    This endpoint is async because it's lightweight — the database query
    is fast and doesn't involve TensorFlow. FastAPI can handle it directly
    on the event loop. (For a production system with heavy DB load, you'd
    want to use an async DB driver like aiosqlite.)

    Query Parameters:
      - limit:   Page size (1–500, default 50).
      - offset:  Skip N records (for pagination, default 0).
      - sort_by: Column to sort by (default 'timestamp').
      - order:   Sort direction — 'asc' or 'desc' (default 'desc').

    Response:
      {
          "predictions": [ { id, timestamp, class_name, confidence, ... }, ... ],
          "total_count": 142
      }
    """
    try:
        result = database.get_history(
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            order=order,
        )
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve history: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# GET /analytics — Aggregated prediction statistics
# ---------------------------------------------------------------------------

@app.get(
    "/analytics",
    summary="Get prediction analytics",
    response_description="Aggregated statistics about past predictions",
)
async def analytics():
    """
    Return aggregated analytics across all logged predictions.

    Response:
      {
          "total_predictions": 142,
          "avg_confidence": 0.8734,
          "class_distribution": { "Stop": 23, "Yield": 17, ... },
          "recent_trend": [ { ... }, ... ]  // last 10 predictions
      }
    """
    try:
        return database.get_analytics()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve analytics: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# GET /health — Health check
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    summary="Health check",
    response_description="Server and model health status",
)
async def health():
    """
    Simple health check endpoint for monitoring and load balancers.

    Returns the server status and whether the ML model is loaded and
    ready for inference. This is useful for:
      - Kubernetes liveness/readiness probes.
      - Frontend connection status indicators.
      - Monitoring dashboards.

    Response:
      {
          "status": "healthy",
          "model_loaded": true
      }
    """
    model_loaded = ml_models.get("traffic_sign") is not None
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
    }
