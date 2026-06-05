/* ============================================================
   InferencePage — Module A: The Core Detection System
   Multi-modal input (upload or webcam) + Results display
   ============================================================ */

import { useState, useCallback } from 'react';
import { Upload, Camera, Cpu, AlertTriangle } from 'lucide-react';
import DropZone from '../components/DropZone';
import WebcamFeed from '../components/WebcamFeed';
import ResultCard from '../components/ResultCard';
import { predictImage, predictFromDataUrl } from '../api/client';

export default function InferencePage() {
  // Input mode: 'upload' or 'webcam'
  const [mode, setMode] = useState('upload');

  // Processing state
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState(null);

  // Result state
  const [result, setResult] = useState(null);
  const [imageSource, setImageSource] = useState(null); // Preview URL
  const [imageFile, setImageFile] = useState(null);      // File for Grad-CAM
  const [isDataUrl, setIsDataUrl] = useState(false);     // Was it from webcam?

  /**
   * Handle file upload from DropZone
   */
  const handleFileSelect = useCallback(async (file) => {
    setError(null);
    setResult(null);
    setProcessing(true);
    setImageFile(file);
    setIsDataUrl(false);

    // Create preview
    const previewUrl = URL.createObjectURL(file);
    setImageSource(previewUrl);

    try {
      const data = await predictImage(file);
      setResult(data);
    } catch (err) {
      console.error('Prediction failed:', err);
      setError(err.message);
    } finally {
      setProcessing(false);
    }
  }, []);

  /**
   * Handle webcam capture
   */
  const handleWebcamCapture = useCallback(async (dataUrl) => {
    setError(null);
    setResult(null);
    setProcessing(true);
    setImageSource(dataUrl);
    setImageFile(null);
    setIsDataUrl(true);

    try {
      const data = await predictFromDataUrl(dataUrl);
      setResult(data);
    } catch (err) {
      console.error('Prediction failed:', err);
      setError(err.message);
    } finally {
      setProcessing(false);
    }
  }, []);

  return (
    <div className="inference-page">
      {/* Page Header */}
      <div className="page-header animate-fade-in-up">
        <div>
          <h1 className="heading-xl">
            <span className="gradient-text">Traffic Sign Detection</span>
          </h1>
          <p className="text-secondary mt-xs">
            Upload an image or use your camera to identify traffic signs in real-time
          </p>
        </div>
        <div className="page-header-badge">
          <Cpu size={14} />
          <span className="text-xs text-mono">MobileNetV2 • 43 Classes</span>
        </div>
      </div>

      {/* Input Mode Selector */}
      <div className="mode-selector animate-fade-in-up" style={{ animationDelay: '0.1s' }}>
        <button
          className={`mode-tab ${mode === 'upload' ? 'mode-tab-active' : ''}`}
          onClick={() => setMode('upload')}
          id="mode-upload"
        >
          <Upload size={18} />
          Image Upload
        </button>
        <button
          className={`mode-tab ${mode === 'webcam' ? 'mode-tab-active' : ''}`}
          onClick={() => setMode('webcam')}
          id="mode-webcam"
        >
          <Camera size={18} />
          Live Camera
        </button>
      </div>

      {/* Input Area */}
      <div className="inference-content">
        <div className="inference-input-area animate-fade-in-up" style={{ animationDelay: '0.15s' }}>
          {mode === 'upload' ? (
            <DropZone onFileSelect={handleFileSelect} disabled={processing} />
          ) : (
            <WebcamFeed onCapture={handleWebcamCapture} disabled={processing} />
          )}
        </div>

        {/* Processing Indicator */}
        {processing && (
          <div className="processing-indicator animate-fade-in">
            <div className="processing-card card-glass">
              <div className="spinner" />
              <div>
                <p className="heading-md">Analyzing Sign...</p>
                <p className="text-xs text-muted mt-xs">Running inference through MobileNetV2</p>
              </div>
            </div>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="error-card animate-fade-in-up">
            <div className="flex items-center gap-sm" style={{ color: 'var(--accent-red)' }}>
              <AlertTriangle size={18} />
              <span className="heading-md">Detection Failed</span>
            </div>
            <p className="text-sm text-secondary mt-sm">{error}</p>
            <button
              className="btn btn-secondary mt-md"
              onClick={() => setError(null)}
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Result Card */}
        {result && !processing && (
          <div className="inference-result" style={{ animationDelay: '0.1s' }}>
            <ResultCard
              result={result}
              imageSource={imageSource}
              imageFile={imageFile}
              isDataUrl={isDataUrl}
            />
          </div>
        )}
      </div>

      <style>{`
        .inference-page {
          max-width: 900px;
          margin: 0 auto;
        }

        .page-header {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          margin-bottom: var(--space-xl);
          gap: var(--space-md);
          flex-wrap: wrap;
        }

        .page-header-badge {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 6px 14px;
          background: var(--bg-card);
          border: 1px solid var(--border-subtle);
          border-radius: var(--radius-full);
          color: var(--text-muted);
          white-space: nowrap;
        }

        /* Mode selector tabs */
        .mode-selector {
          display: flex;
          background: var(--bg-card);
          border: 1px solid var(--border-subtle);
          border-radius: var(--radius-lg);
          padding: 4px;
          margin-bottom: var(--space-xl);
          width: fit-content;
        }

        .mode-tab {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 10px 24px;
          background: transparent;
          border: none;
          border-radius: var(--radius-md);
          color: var(--text-muted);
          font-family: var(--font-sans);
          font-size: 0.875rem;
          font-weight: 600;
          cursor: pointer;
          transition: all var(--transition-fast);
        }

        .mode-tab:hover {
          color: var(--text-primary);
          background: var(--bg-elevated);
        }

        .mode-tab-active {
          background: var(--accent-blue-dim);
          color: var(--accent-blue);
          box-shadow: 0 0 12px var(--accent-blue-dim);
        }

        /* Content layout */
        .inference-content {
          display: flex;
          flex-direction: column;
          gap: var(--space-xl);
        }

        .inference-input-area {
          width: 100%;
        }

        /* Processing indicator */
        .processing-indicator {
          display: flex;
          justify-content: center;
        }

        .processing-card {
          display: flex;
          align-items: center;
          gap: var(--space-md);
          padding: var(--space-lg) var(--space-xl);
        }

        /* Error card */
        .error-card {
          background: rgba(239, 68, 68, 0.08);
          border: 1px solid rgba(239, 68, 68, 0.2);
          border-radius: var(--radius-lg);
          padding: var(--space-lg);
        }

        /* Result section */
        .inference-result {
          animation: fadeInUp 0.5s ease-out forwards;
        }

        @media (max-width: 768px) {
          .mode-selector {
            width: 100%;
          }

          .mode-tab {
            flex: 1;
            justify-content: center;
            padding: 10px 16px;
          }
        }
      `}</style>
    </div>
  );
}
