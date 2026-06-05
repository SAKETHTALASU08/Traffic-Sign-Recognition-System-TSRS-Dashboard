/* ============================================================
   WebcamFeed — Live webcam capture for real-time ADAS simulation
   Uses navigator.mediaDevices.getUserMedia API
   Features: scanning overlay, corner brackets, capture button
   ============================================================ */

import { useState, useRef, useCallback, useEffect } from 'react';
import { Camera, CameraOff, RefreshCw, Aperture } from 'lucide-react';

/**
 * @param {object} props
 * @param {function} props.onCapture - Called with captured data URL when user clicks capture
 * @param {boolean} props.disabled - Disable capture during processing
 */
export default function WebcamFeed({ onCapture, disabled = false }) {
  const [isActive, setIsActive] = useState(false);
  const [error, setError] = useState(null);
  const [facingMode, setFacingMode] = useState('environment'); // 'user' for front cam
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const canvasRef = useRef(null);

  /**
   * Start the webcam stream
   */
  const startCamera = useCallback(async () => {
    try {
      setError(null);
      const constraints = {
        video: {
          facingMode,
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setIsActive(true);
    } catch (err) {
      console.error('Webcam error:', err);
      if (err.name === 'NotAllowedError') {
        setError('Camera access denied. Please allow camera permissions in your browser.');
      } else if (err.name === 'NotFoundError') {
        setError('No camera found on this device.');
      } else {
        setError('Failed to access camera. Please try again.');
      }
    }
  }, [facingMode]);

  /**
   * Stop the webcam stream
   */
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsActive(false);
  }, []);

  /**
   * Capture a frame from the video stream
   */
  const captureFrame = useCallback(() => {
    if (!videoRef.current || !isActive || disabled) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!canvas) return;

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    // Convert to data URL (JPEG for smaller size)
    const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
    onCapture(dataUrl);
  }, [isActive, disabled, onCapture]);

  /**
   * Toggle between front and rear cameras
   */
  const toggleCamera = useCallback(async () => {
    stopCamera();
    setFacingMode(prev => prev === 'user' ? 'environment' : 'user');
  }, [stopCamera]);

  // Restart camera when facingMode changes
  useEffect(() => {
    if (isActive || facingMode !== 'environment') {
      // Small delay to allow previous stream to fully stop
      const timer = setTimeout(() => {
        if (facingMode !== 'environment' || isActive) {
          startCamera();
        }
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [facingMode]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <div className="webcam-wrapper">
      {/* Hidden canvas for frame capture */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {isActive ? (
        /* ---- Active Webcam View ---- */
        <div className="webcam-container">
          <video
            ref={videoRef}
            playsInline
            muted
            style={{ width: '100%', display: 'block' }}
          />

          {/* Scanning overlay effects */}
          <div className="webcam-overlay">
            <div className="webcam-scan-line" />
            <div className="webcam-corners" />
            <div className="webcam-corners-bottom" />
          </div>

          {/* Status badge */}
          <div className="webcam-status-badge">
            <span className="webcam-status-dot" />
            LIVE
          </div>

          {/* Controls bar */}
          <div className="webcam-controls">
            <button
              className="btn btn-secondary btn-icon"
              onClick={toggleCamera}
              aria-label="Switch camera"
              title="Switch camera"
            >
              <RefreshCw size={18} />
            </button>

            <button
              className="webcam-capture-btn"
              onClick={captureFrame}
              disabled={disabled}
              aria-label="Capture frame"
              title="Capture"
            >
              <Aperture size={28} />
            </button>

            <button
              className="btn btn-danger btn-icon"
              onClick={stopCamera}
              aria-label="Stop camera"
              title="Stop camera"
            >
              <CameraOff size={18} />
            </button>
          </div>
        </div>
      ) : (
        /* ---- Inactive / Start View ---- */
        <div className="webcam-inactive" onClick={startCamera}>
          {error ? (
            <div className="webcam-error">
              <CameraOff size={40} />
              <p className="text-sm" style={{ color: 'var(--accent-red)', textAlign: 'center', maxWidth: 280 }}>
                {error}
              </p>
              <button className="btn btn-secondary" onClick={(e) => { e.stopPropagation(); startCamera(); }}>
                Retry
              </button>
            </div>
          ) : (
            <div className="webcam-start">
              <div className="webcam-start-icon">
                <Camera size={36} />
              </div>
              <p className="heading-md">Live Camera Feed</p>
              <p className="text-sm text-muted">
                Activate webcam for real-time sign detection
              </p>
              <button className="btn btn-primary mt-md" onClick={(e) => { e.stopPropagation(); startCamera(); }}>
                <Camera size={16} />
                Start Camera
              </button>
            </div>
          )}
        </div>
      )}

      <style>{`
        .webcam-wrapper {
          width: 100%;
        }

        .webcam-inactive {
          border: 2px dashed var(--border-medium);
          border-radius: var(--radius-xl);
          padding: 48px 24px;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          transition: all var(--transition-base);
          background: var(--bg-card);
          min-height: 280px;
        }

        .webcam-inactive:hover {
          border-color: var(--accent-blue);
          box-shadow: var(--accent-blue-glow);
        }

        .webcam-start, .webcam-error {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 8px;
        }

        .webcam-start-icon {
          width: 72px;
          height: 72px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: var(--accent-blue-dim);
          border-radius: 50%;
          color: var(--accent-blue);
          margin-bottom: 8px;
        }

        /* Status badge */
        .webcam-status-badge {
          position: absolute;
          top: 12px;
          left: 12px;
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 4px 12px;
          background: rgba(239, 68, 68, 0.85);
          backdrop-filter: blur(8px);
          border-radius: var(--radius-full);
          color: white;
          font-size: 0.7rem;
          font-weight: 700;
          letter-spacing: 0.08em;
          z-index: 5;
        }

        .webcam-status-dot {
          width: 6px;
          height: 6px;
          background: white;
          border-radius: 50%;
          animation: pulse-ring 1.5s infinite;
        }

        /* Controls bar */
        .webcam-controls {
          position: absolute;
          bottom: 16px;
          left: 50%;
          transform: translateX(-50%);
          display: flex;
          align-items: center;
          gap: 16px;
          padding: 8px 16px;
          background: rgba(6, 10, 19, 0.8);
          backdrop-filter: blur(12px);
          border-radius: var(--radius-full);
          border: 1px solid var(--border-subtle);
          z-index: 5;
        }

        .webcam-capture-btn {
          width: 56px;
          height: 56px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: linear-gradient(135deg, #00b4d8, #00d4ff);
          color: var(--text-inverse);
          border: 3px solid white;
          border-radius: 50%;
          cursor: pointer;
          transition: all var(--transition-fast);
          box-shadow: 0 2px 16px var(--accent-blue-dim);
        }

        .webcam-capture-btn:hover:not(:disabled) {
          transform: scale(1.1);
          box-shadow: var(--accent-blue-glow);
        }

        .webcam-capture-btn:active:not(:disabled) {
          transform: scale(0.95);
        }

        .webcam-capture-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  );
}
