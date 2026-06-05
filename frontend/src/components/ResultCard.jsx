/* ============================================================
   ResultCard — Prediction result display with glass-morphism
   Shows: predicted class, confidence bar, latency, Grad-CAM
   ============================================================ */

import { useEffect, useState } from 'react';
import { Tag, Clock, Zap, Eye } from 'lucide-react';
import ConfidenceBar from './ConfidenceBar';
import GradCAMOverlay from './GradCAMOverlay';
import { explainImage, explainFromDataUrl } from '../api/client';

/**
 * @param {object} props
 * @param {object} props.result - Prediction result from API
 * @param {string} props.imageSource - Original image URL or data URL
 * @param {File|null} props.imageFile - Original file (for Grad-CAM request)
 * @param {boolean} props.isDataUrl - Whether imageSource is a data URL (webcam)
 */
export default function ResultCard({ result, imageSource, imageFile, isDataUrl = false }) {
  const [gradcamImage, setGradcamImage] = useState(null);
  const [gradcamLoading, setGradcamLoading] = useState(false);
  const [gradcamError, setGradcamError] = useState(null);

  if (!result) return null;

  const { class_name, confidence, latency_ms } = result;
  const confidencePercent = Math.round(confidence * 100);

  /**
   * Fetch Grad-CAM explanation from backend
   */
  const fetchGradCAM = async () => {
    setGradcamLoading(true);
    setGradcamError(null);
    try {
      let data;
      if (isDataUrl && imageSource) {
        data = await explainFromDataUrl(imageSource);
      } else if (imageFile) {
        data = await explainImage(imageFile);
      } else {
        throw new Error('No image source available for Grad-CAM');
      }
      setGradcamImage(data.gradcam_image);
    } catch (err) {
      console.error('Grad-CAM error:', err);
      setGradcamError(err.message);
    } finally {
      setGradcamLoading(false);
    }
  };

  // Get confidence badge style
  const getConfidenceBadge = () => {
    if (confidencePercent >= 80) return { className: 'badge-green', label: 'High Confidence' };
    if (confidencePercent >= 50) return { className: 'badge-yellow', label: 'Medium Confidence' };
    return { className: 'badge-red', label: 'Low Confidence' };
  };

  const badge = getConfidenceBadge();

  return (
    <div className="result-card animate-fade-in-up">
      {/* Header */}
      <div className="result-header">
        <div className="result-status">
          <Zap size={16} style={{ color: 'var(--accent-green)' }} />
          <span className="text-xs text-success" style={{ fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Detection Complete
          </span>
        </div>
        <span className={`badge ${badge.className}`}>{badge.label}</span>
      </div>

      {/* Main content */}
      <div className="result-body">
        {/* Image preview */}
        <div className="result-image-section">
          <img
            src={imageSource}
            alt={`Detected: ${class_name}`}
            className="result-image"
          />
        </div>

        {/* Prediction details */}
        <div className="result-details">
          {/* Predicted class */}
          <div className="result-class">
            <Tag size={18} style={{ color: 'var(--accent-blue)', flexShrink: 0 }} />
            <div>
              <span className="text-xs text-muted" style={{ display: 'block', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600, marginBottom: 2 }}>
                Predicted Class
              </span>
              <span className="heading-lg gradient-text">{class_name}</span>
            </div>
          </div>

          {/* Confidence bar */}
          <div className="mt-lg">
            <ConfidenceBar confidence={confidence} size="lg" />
          </div>

          {/* Metadata row */}
          <div className="result-meta">
            <div className="result-meta-item">
              <Clock size={14} />
              <span className="text-mono text-sm">{latency_ms?.toFixed(1)}ms</span>
              <span className="text-xs text-muted">Latency</span>
            </div>
            <div className="result-meta-divider" />
            <div className="result-meta-item">
              <Zap size={14} />
              <span className="text-mono text-sm">{confidencePercent}%</span>
              <span className="text-xs text-muted">Score</span>
            </div>
          </div>

          {/* Grad-CAM button */}
          <button
            className="btn btn-secondary w-full mt-md"
            onClick={fetchGradCAM}
            disabled={gradcamLoading}
            style={{ justifyContent: 'center' }}
          >
            {gradcamLoading ? (
              <>
                <div className="spinner spinner-sm" />
                Generating Heatmap...
              </>
            ) : (
              <>
                <Eye size={16} />
                Explain Prediction (Grad-CAM)
              </>
            )}
          </button>

          {gradcamError && (
            <p className="text-xs text-danger mt-sm">{gradcamError}</p>
          )}
        </div>
      </div>

      {/* Grad-CAM section */}
      {gradcamImage && (
        <div className="mt-lg">
          <GradCAMOverlay
            originalImage={imageSource}
            gradcamImage={gradcamImage}
          />
        </div>
      )}

      <style>{`
        .result-card {
          background: rgba(17, 24, 39, 0.7);
          backdrop-filter: blur(20px);
          -webkit-backdrop-filter: blur(20px);
          border: 1px solid rgba(0, 212, 255, 0.15);
          border-radius: var(--radius-lg);
          padding: var(--space-lg);
          box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3), 0 0 30px rgba(0, 212, 255, 0.05);
        }

        .result-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: var(--space-lg);
          padding-bottom: var(--space-md);
          border-bottom: 1px solid var(--border-subtle);
        }

        .result-status {
          display: flex;
          align-items: center;
          gap: 6px;
        }

        .result-body {
          display: grid;
          grid-template-columns: auto 1fr;
          gap: var(--space-xl);
          align-items: start;
        }

        .result-image-section {
          display: flex;
          flex-direction: column;
          align-items: center;
        }

        .result-image {
          width: 180px;
          height: 180px;
          object-fit: contain;
          border-radius: var(--radius-md);
          border: 2px solid var(--border-subtle);
          background: #000;
          box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        }

        .result-details {
          min-width: 0;
        }

        .result-class {
          display: flex;
          gap: 12px;
          align-items: flex-start;
        }

        .result-meta {
          display: flex;
          align-items: center;
          gap: var(--space-lg);
          margin-top: var(--space-lg);
          padding: var(--space-md);
          background: var(--bg-secondary);
          border-radius: var(--radius-md);
        }

        .result-meta-item {
          display: flex;
          align-items: center;
          gap: 8px;
          color: var(--text-secondary);
        }

        .result-meta-item .text-xs {
          margin-left: 4px;
        }

        .result-meta-divider {
          width: 1px;
          height: 24px;
          background: var(--border-subtle);
        }

        @media (max-width: 600px) {
          .result-body {
            grid-template-columns: 1fr;
          }

          .result-image-section {
            order: -1;
          }

          .result-image {
            width: 140px;
            height: 140px;
          }

          .result-meta {
            flex-direction: column;
            gap: var(--space-sm);
          }

          .result-meta-divider {
            width: 100%;
            height: 1px;
          }
        }
      `}</style>
    </div>
  );
}
