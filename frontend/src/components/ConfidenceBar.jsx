/* ============================================================
   ConfidenceBar — Animated progress bar for prediction confidence
   Color-coded: green (>80%), yellow (50-80%), red (<50%)
   ============================================================ */

import { useEffect, useState } from 'react';

/**
 * @param {object} props
 * @param {number} props.confidence - Confidence value between 0 and 1
 * @param {boolean} props.showLabel - Show percentage text (default true)
 * @param {string} props.size - 'sm' | 'md' | 'lg' (default 'md')
 */
export default function ConfidenceBar({ confidence = 0, showLabel = true, size = 'md' }) {
  const [width, setWidth] = useState(0);
  const percentage = Math.round(confidence * 100);

  // Animate the bar width on mount/update
  useEffect(() => {
    const timer = setTimeout(() => {
      setWidth(percentage);
    }, 100);
    return () => clearTimeout(timer);
  }, [percentage]);

  // Determine color tier
  const getColorClass = () => {
    if (percentage >= 80) return 'confidence-high';
    if (percentage >= 50) return 'confidence-medium';
    return 'confidence-low';
  };

  const getTextColor = () => {
    if (percentage >= 80) return 'var(--accent-green)';
    if (percentage >= 50) return 'var(--accent-yellow)';
    return 'var(--accent-red)';
  };

  const heights = { sm: '6px', md: '12px', lg: '16px' };

  return (
    <div className="confidence-wrapper">
      {showLabel && (
        <div className="flex justify-between items-center mb-sm">
          <span className="text-xs text-muted" style={{ textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
            Confidence
          </span>
          <span
            className="text-mono"
            style={{
              fontSize: size === 'lg' ? '1.25rem' : '0.9rem',
              fontWeight: 700,
              color: getTextColor(),
            }}
          >
            {percentage}%
          </span>
        </div>
      )}
      <div
        className="confidence-bar-container"
        style={{ height: heights[size] }}
        role="progressbar"
        aria-valuenow={percentage}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`Confidence: ${percentage}%`}
      >
        <div
          className={`confidence-bar-fill ${getColorClass()}`}
          style={{ width: `${width}%` }}
        />
      </div>
    </div>
  );
}
