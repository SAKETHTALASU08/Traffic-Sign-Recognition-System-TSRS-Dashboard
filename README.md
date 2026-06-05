# Traffic Sign Recognition System (TSRS) Dashboard

An Advanced Driver-Assistance Systems (ADAS) inspired web application that provides real-time traffic sign classification using a fine-tuned MobileNetV2 deep learning model. The system features a modern, responsive Dark Mode UI, live webcam inference, and model explainability via Grad-CAM heatmaps.

## 🚀 Key Features

* **Real-time Inference:** Classifies 43 different German traffic signs with over 91% real-world test accuracy.
* **Multi-Modal Input:** Supports standard image upload (Drag & Drop) and Live Webcam feeds for real-time testing.
* **Explainable AI (Grad-CAM):** Generates Gradient-weighted Class Activation Mapping heatmaps to visually explain *why* the model made a specific prediction (highlighting the pixels the model focused on).
* **Analytics & History:** Logs all predictions to a SQLite database, providing paginated history and aggregated metrics (average confidence, class distribution).

## 🛠️ Tech Stack

### Frontend (User Interface)
* **Framework:** React.js powered by Vite for lightning-fast HMR and building.
* **Styling:** Vanilla CSS with custom design tokens, modern glassmorphism effects, and CSS variables for a premium automotive-inspired Dark Mode aesthetic.
* **Icons:** Lucide React.

### Backend (API & Inference)
* **Framework:** FastAPI (Python) with Uvicorn for high-performance, asynchronous routing.
* **Database:** SQLite for lightweight, zero-config prediction logging.
* **Image Processing:** Pillow (PIL) and OpenCV-Headless.

### Machine Learning
* **Framework:** TensorFlow / Keras.
* **Architecture:** MobileNetV2 (Transfer Learning). The base model acts as a feature extractor, followed by a custom dense classification head.
* **Dataset:** German Traffic Sign Recognition Benchmark (GTSRB).

## 🧠 Architecture Flow

1. **Client Request:** The user captures an image via Webcam or Drag & Drop. The React frontend sends a `multipart/form-data` POST request to the FastAPI backend.
2. **Preprocessing:** The backend decodes the image, resizes it to 128x128, and applies MobileNetV2-specific scaling (`[-1, 1]`).
3. **Inference:** The tensor is passed through the `.h5` model to generate class probabilities.
4. **Grad-CAM Generation:** If the user requests an explanation, the backend computes the gradient of the predicted class with respect to the last convolutional layer (`Conv_1`), applies a ReLU activation, scales it to a heatmap, and overlays it on the original image using OpenCV.
5. **Storage & Response:** The prediction details and a base64 thumbnail are logged to SQLite. The API returns the predicted class, confidence score, and latency to the frontend.

## 💻 Local Setup

### 1. Backend Setup
```bash
# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn backend.main:app --reload --port 8000
```

### 2. Frontend Setup
```bash
cd frontend

# Install Node modules
npm install

# Start the Vite dev server
npm run dev
```
Navigate to `http://localhost:5173` in your browser.

## ☁️ Deployment

This project is optimized for deployment on [Render](https://render.com/).
1. **Backend:** Deploy as a **Web Service** using the Python environment. Set the Start Command to `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`.
2. **Frontend:** Deploy as a **Static Site**. Update `API_BASE` in `frontend/src/api/client.js` to point to your live backend URL, then set the build command to `npm install && npm run build` and publish the `dist` folder.
