/* ============================================================
   API Client — Centralized fetch wrapper for FastAPI backend
   ============================================================ */

// Base URL for the FastAPI backend (change in production)
const API_BASE = 'http://localhost:8000';

/**
 * Generic fetch wrapper with error handling
 * @param {string} endpoint - API endpoint path (e.g., '/predict')
 * @param {object} options - Fetch options (method, body, headers, etc.)
 * @returns {Promise<object>} Parsed JSON response
 */
async function apiFetch(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;

  try {
    const response = await fetch(url, {
      ...options,
      // Don't set Content-Type for FormData — browser does it automatically
      headers: options.body instanceof FormData
        ? options.headers || {}
        : { 'Content-Type': 'application/json', ...options.headers },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `API error: ${response.status} ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    // Re-throw with more context if it's a network error
    if (error.name === 'TypeError' && error.message === 'Failed to fetch') {
      throw new Error('Cannot connect to the backend server. Make sure it is running on http://localhost:8000');
    }
    throw error;
  }
}

/**
 * POST /predict — Send an image for classification
 * @param {File} imageFile - The image file to classify
 * @returns {Promise<{class_id, class_name, confidence, latency_ms, all_probabilities}>}
 */
export async function predictImage(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);

  return apiFetch('/predict', {
    method: 'POST',
    body: formData,
  });
}

// Helper to convert base64 data URI to Blob reliably across all browsers
function dataURItoBlob(dataURI) {
  const parts = dataURI.split(',');
  const byteString = atob(parts[1]);
  const mimeString = parts[0].split(':')[1].split(';')[0];
  
  const ab = new ArrayBuffer(byteString.length);
  const ia = new Uint8Array(ab);
  for (let i = 0; i < byteString.length; i++) {
    ia[i] = byteString.charCodeAt(i);
  }
  return new Blob([ab], { type: mimeString });
}

/**
 * POST /predict from a base64 data URL (for webcam captures)
 * Converts the data URL to a Blob, then sends as FormData
 * @param {string} dataUrl - Base64 data URL from canvas.toDataURL()
 * @returns {Promise<{class_id, class_name, confidence, latency_ms, all_probabilities}>}
 */
export async function predictFromDataUrl(dataUrl) {
  const blob = dataURItoBlob(dataUrl);
  if (blob.size === 0) throw new Error("Empty image captured from webcam.");
  const file = new File([blob], 'webcam-capture.jpg', { type: 'image/jpeg' });

  return predictImage(file);
}

/**
 * POST /explain — Get Grad-CAM explanation for an image
 * @param {File} imageFile - The image file to explain
 * @returns {Promise<{class_name, confidence, gradcam_image}>}
 */
export async function explainImage(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);

  return apiFetch('/explain', {
    method: 'POST',
    body: formData,
  });
}

/**
 * POST /explain from a base64 data URL (for webcam captures)
 * @param {string} dataUrl - Base64 data URL
 * @returns {Promise<{class_name, confidence, gradcam_image}>}
 */
export async function explainFromDataUrl(dataUrl) {
  const blob = dataURItoBlob(dataUrl);
  if (blob.size === 0) throw new Error("Empty image captured from webcam.");
  const file = new File([blob], 'webcam-capture.jpg', { type: 'image/jpeg' });

  return explainImage(file);
}

/**
 * GET /history — Retrieve prediction history
 * @param {object} params - Query parameters
 * @param {number} params.limit - Number of records per page (default 50)
 * @param {number} params.offset - Offset for pagination (default 0)
 * @param {string} params.sort_by - Column to sort by (default 'timestamp')
 * @param {string} params.order - Sort order 'asc' or 'desc' (default 'desc')
 * @returns {Promise<{predictions: Array, total_count: number}>}
 */
export async function getHistory({ limit = 50, offset = 0, sort_by = 'timestamp', order = 'desc' } = {}) {
  const params = new URLSearchParams({ limit, offset, sort_by, order });
  return apiFetch(`/history?${params}`);
}

/**
 * GET /analytics — Retrieve aggregated analytics
 * @returns {Promise<{total_predictions, avg_confidence, class_distribution}>}
 */
export async function getAnalytics() {
  return apiFetch('/analytics');
}

/**
 * GET /health — Check backend health
 * @returns {Promise<{status, model_loaded}>}
 */
export async function checkHealth() {
  return apiFetch('/health');
}
