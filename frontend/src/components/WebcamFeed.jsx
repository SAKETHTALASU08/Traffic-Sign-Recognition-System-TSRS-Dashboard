/* ============================================================
   WebcamFeed — Live webcam capture for real-time ADAS simulation
   Fixed: video element always in DOM so ref is never null.
   Fixed: mobile Safari autoplay, playsInline, muted attributes.
   Fixed: proper facingMode fallback for iOS.
   ============================================================ */

import { useState, useRef, useCallback, useEffect } from 'react';
import { Camera, CameraOff, RefreshCw, Aperture } from 'lucide-react';

export default function WebcamFeed({ onCapture, disabled = false }) {
  const [isActive, setIsActive] = useState(false);
  const [error, setError] = useState(null);
  const [facingMode, setFacingMode] = useState('environment');
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const canvasRef = useRef(null);

  // Attach stream to video element whenever isActive or stream changes
  useEffect(() => {
    if (isActive && streamRef.current && videoRef.current) {
      videoRef.current.srcObject = streamRef.current;
      videoRef.current.play().catch((e) => {
        console.error('Video play error:', e);
      });
    }
  }, [isActive]);

  const startCamera = useCallback(async () => {
    try {
      setError(null);

      // Stop any existing stream first
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
        streamRef.current = null;
      }

      let stream = null;

      // Try with preferred facingMode first, fall back to any camera
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode, width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false,
        });
      } catch {
        // Fallback: try without facingMode constraint (works on most desktops & older mobiles)
        stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });
      }

      streamRef.current = stream;

      // Set isActive TRUE first so the video element renders in the DOM
      setIsActive(true);

      // The useEffect above will attach the stream once the element is mounted
    } catch (err) {
      console.error('Webcam error:', err);
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setError('Camera access denied. Please allow camera permissions and try again.');
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        setError('No camera found on this device.');
      } else if (err.name === 'NotReadableError') {
        setError('Camera is already in use by another app. Please close it and retry.');
      } else {
        setError(`Failed to access camera: ${err.message}`);
      }
    }
  }, [facingMode]);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsActive(false);
  }, []);

  const captureFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !isActive || disabled) return;

    // If video hasn't loaded metadata yet, retry in 500ms
    if (video.videoWidth === 0 || video.videoHeight === 0 || video.readyState < 2) {
      console.warn('Video not ready, retrying in 500ms...');
      setTimeout(captureFrame, 500);
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
    onCapture(dataUrl);
  }, [isActive, disabled, onCapture]);

  const toggleCamera = useCallback(async () => {
    const newMode = facingMode === 'user' ? 'environment' : 'user';
    setFacingMode(newMode);
    // Stop current stream, restart with new facing mode
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: newMode },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play().catch(console.error);
      }
    } catch (e) {
      console.error('Toggle camera error:', e);
    }
  }, [facingMode]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
      }
    };
  }, []);

  return (
    <div className="webcam-wrapper">
      {/* Hidden canvas for frame capture */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {/*
        CRITICAL FIX: The video element is ALWAYS in the DOM.
        Previously it was inside the {isActive ? ...} block, making
        videoRef.current null when startCamera ran. Now we just
        toggle visibility with CSS.
      */}
      <video
        ref={videoRef}
        playsInline
        muted
        autoPlay
        style={{ display: isActive ? 'block' : 'none', width: '100%' }}
      />

      {isActive ? (
        /* ---- Active Webcam Overlay (controls on top of video) ---- */
        <div className="webcam-active-ui">
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
        <div className="webcam-inactive">
          {error ? (
            <div className="webcam-error">
              <CameraOff size={40} />
              <p className="text-sm" style={{ color: 'var(--accent-red)', textAlign: 'center', maxWidth: 280 }}>
                {error}
              </p>
              <button className="btn btn-secondary" onClick={startCamera}>
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
              <button className="btn btn-primary mt-md" onClick={startCamera}>
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
          position: relative;
        }

        /* Active UI sits on top of the always-rendered video element */
        .webcam-active-ui {
          position: absolute;
          inset: 0;
          pointer-events: none;
        }

        .webcam-active-ui .webcam-controls {
          pointer-events: all;
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

        /* Video border when active */
        video {
          border-radius: var(--radius-xl);
          border: 1px solid var(--border-subtle);
          background: #000;
          min-height: 280px;
          object-fit: cover;
        }
      `}</style>
    </div>
  );
}
