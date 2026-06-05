/* ============================================================
   GradCAMOverlay — Displays original image alongside its 
   Grad-CAM heatmap explanation side-by-side
   ============================================================ */

/**
 * @param {object} props
 * @param {string} props.originalImage - Original image URL/data URL
 * @param {string} props.gradcamImage - Base64 Grad-CAM overlay image from API
 * @param {string} props.className - Additional class name
 */
export default function GradCAMOverlay({ originalImage, gradcamImage, className = '' }) {
  if (!gradcamImage) return null;

  return (
    <div className={`gradcam-container ${className}`}>
      <div className="gradcam-header">
        <div className="gradcam-badge">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <path d="M12 16v-4M12 8h.01" />
          </svg>
          Grad-CAM Explainability
        </div>
      </div>

      <div className="gradcam-images">
        {/* Original Image */}
        <div className="gradcam-panel">
          <img
            src={originalImage}
            alt="Original traffic sign"
            className="gradcam-img"
          />
          <span className="gradcam-label">Original</span>
        </div>

        {/* Arrow indicator */}
        <div className="gradcam-arrow">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="5" y1="12" x2="19" y2="12" />
            <polyline points="12 5 19 12 12 19" />
          </svg>
        </div>

        {/* Grad-CAM Heatmap */}
        <div className="gradcam-panel">
          <img
            src={`data:image/png;base64,${gradcamImage}`}
            alt="Grad-CAM heatmap showing model attention"
            className="gradcam-img gradcam-heatmap"
          />
          <span className="gradcam-label">Attention Map</span>
        </div>
      </div>

      <p className="gradcam-description">
        Warm regions (red/yellow) indicate areas the CNN focused on for its prediction.
      </p>

      <style>{`
        .gradcam-container {
          background: var(--bg-card);
          border: 1px solid var(--border-subtle);
          border-radius: var(--radius-lg);
          padding: var(--space-lg);
          animation: fadeInUp 0.5s ease-out forwards;
        }

        .gradcam-header {
          margin-bottom: var(--space-md);
        }

        .gradcam-badge {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 4px 12px;
          background: rgba(168, 85, 247, 0.15);
          color: #a855f7;
          border-radius: var(--radius-full);
          font-size: 0.75rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.04em;
        }

        .gradcam-images {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: var(--space-md);
          flex-wrap: wrap;
        }

        .gradcam-panel {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 8px;
        }

        .gradcam-img {
          width: 160px;
          height: 160px;
          object-fit: contain;
          border-radius: var(--radius-md);
          border: 2px solid var(--border-subtle);
          background: #000;
        }

        .gradcam-heatmap {
          border-color: rgba(168, 85, 247, 0.3);
          box-shadow: 0 0 16px rgba(168, 85, 247, 0.2);
        }

        .gradcam-label {
          font-size: 0.7rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          color: var(--text-muted);
        }

        .gradcam-arrow {
          color: var(--text-muted);
          opacity: 0.5;
        }

        .gradcam-description {
          margin-top: var(--space-md);
          font-size: 0.75rem;
          color: var(--text-muted);
          text-align: center;
          font-style: italic;
        }

        @media (max-width: 480px) {
          .gradcam-img {
            width: 120px;
            height: 120px;
          }

          .gradcam-arrow {
            transform: rotate(90deg);
          }
        }
      `}</style>
    </div>
  );
}
