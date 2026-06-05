/* ============================================================
   DropZone — Drag-and-drop image upload with visual feedback
   Features: drag state, file validation, image preview, 
   neon-blue glow on hover/drag
   ============================================================ */

import { useState, useRef, useCallback } from 'react';
import { Upload, ImagePlus, X, FileImage } from 'lucide-react';

/**
 * @param {object} props
 * @param {function} props.onFileSelect - Callback when file(s) are selected. Receives File object.
 * @param {boolean} props.disabled - Disable interactions during processing
 */
export default function DropZone({ onFileSelect, disabled = false }) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [preview, setPreview] = useState(null);
  const [fileName, setFileName] = useState('');
  const fileInputRef = useRef(null);

  // Accepted image MIME types
  const ACCEPTED_TYPES = ['image/png', 'image/jpeg', 'image/webp', 'image/bmp'];

  /**
   * Validates file type and creates a preview
   */
  const processFile = useCallback((file) => {
    if (!ACCEPTED_TYPES.includes(file.type)) {
      alert('Please upload a valid image file (PNG, JPG, WEBP, or BMP).');
      return;
    }

    // Create preview URL
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target.result);
      setFileName(file.name);
    };
    reader.readAsDataURL(file);

    // Notify parent
    onFileSelect(file);
  }, [onFileSelect]);

  /* ---- Drag event handlers ---- */
  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) setIsDragOver(true);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);

    if (disabled) return;

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      processFile(files[0]);
    }
  };

  /* ---- Click handler ---- */
  const handleClick = () => {
    if (!disabled && fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleInputChange = (e) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      processFile(files[0]);
    }
    // Reset input so same file can be re-selected
    e.target.value = '';
  };

  /* ---- Clear preview ---- */
  const clearPreview = (e) => {
    e.stopPropagation();
    setPreview(null);
    setFileName('');
  };

  return (
    <div
      id="dropzone-upload"
      className={`dropzone ${isDragOver ? 'drag-over' : ''} ${disabled ? 'dropzone-disabled' : ''}`}
      onClick={handleClick}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      role="button"
      tabIndex={0}
      aria-label="Upload traffic sign image"
    >
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".png,.jpg,.jpeg,.webp,.bmp"
        onChange={handleInputChange}
        style={{ display: 'none' }}
        aria-hidden="true"
      />

      {preview ? (
        /* ---- Image Preview State ---- */
        <div className="dropzone-preview" style={{ position: 'relative', zIndex: 1 }}>
          <div className="dropzone-preview-image-wrapper">
            <img
              src={preview}
              alt="Upload preview"
              className="dropzone-preview-image"
            />
            {!disabled && (
              <button
                className="dropzone-clear-btn"
                onClick={clearPreview}
                aria-label="Remove image"
              >
                <X size={14} />
              </button>
            )}
          </div>
          <div className="dropzone-file-info">
            <FileImage size={14} />
            <span className="text-sm truncate">{fileName}</span>
          </div>
          <p className="text-xs text-muted mt-sm">Click or drag to replace</p>
        </div>
      ) : (
        /* ---- Empty / Upload State ---- */
        <div className="dropzone-empty" style={{ position: 'relative', zIndex: 1 }}>
          <div className="dropzone-icon">
            {isDragOver ? (
              <ImagePlus size={48} />
            ) : (
              <Upload size={48} />
            )}
          </div>
          <p className="heading-md" style={{ marginBottom: '8px' }}>
            {isDragOver ? 'Drop your image here' : 'Upload Traffic Sign'}
          </p>
          <p className="text-sm text-muted">
            Drag and drop an image, or click to browse
          </p>
          <div className="dropzone-formats">
            <span className="badge badge-blue">PNG</span>
            <span className="badge badge-blue">JPG</span>
            <span className="badge badge-blue">WEBP</span>
            <span className="badge badge-blue">BMP</span>
          </div>
        </div>
      )}

      {/* Disabled overlay */}
      {disabled && (
        <div className="dropzone-disabled-overlay">
          <div className="spinner spinner-sm" />
          <span className="text-xs text-muted">Processing...</span>
        </div>
      )}

      <style>{`
        .dropzone-disabled {
          opacity: 0.6;
          cursor: not-allowed;
          pointer-events: none;
        }

        .dropzone-empty {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 4px;
        }

        .dropzone-formats {
          display: flex;
          gap: 6px;
          margin-top: 16px;
        }

        .dropzone-preview {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 12px;
        }

        .dropzone-preview-image-wrapper {
          position: relative;
          display: inline-block;
        }

        .dropzone-preview-image {
          max-width: 220px;
          max-height: 220px;
          border-radius: var(--radius-md);
          border: 2px solid var(--border-subtle);
          object-fit: contain;
          background: #000;
        }

        .dropzone-clear-btn {
          position: absolute;
          top: -8px;
          right: -8px;
          width: 24px;
          height: 24px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: var(--accent-red);
          color: white;
          border: 2px solid var(--bg-card);
          border-radius: 50%;
          cursor: pointer;
          transition: all var(--transition-fast);
        }

        .dropzone-clear-btn:hover {
          transform: scale(1.15);
        }

        .dropzone-file-info {
          display: flex;
          align-items: center;
          gap: 6px;
          color: var(--text-secondary);
          max-width: 220px;
        }

        .dropzone-disabled-overlay {
          position: absolute;
          inset: 0;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 8px;
          background: rgba(6, 10, 19, 0.7);
          backdrop-filter: blur(4px);
          border-radius: var(--radius-xl);
          z-index: 2;
        }
      `}</style>
    </div>
  );
}
