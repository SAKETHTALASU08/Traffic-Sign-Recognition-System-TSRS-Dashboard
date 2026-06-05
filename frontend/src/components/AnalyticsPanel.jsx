/* ============================================================
   AnalyticsPanel — Dashboard metrics with stat cards and pie chart
   Shows: total processed, avg confidence, class distribution
   ============================================================ */

import { useState, useEffect, useRef } from 'react';
import { BarChart3, Target, TrendingUp, RefreshCw } from 'lucide-react';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { Pie } from 'react-chartjs-2';
import { getAnalytics } from '../api/client';

// Register Chart.js components
ChartJS.register(ArcElement, Tooltip, Legend);

/**
 * Animated counter component for stat cards
 */
function AnimatedCounter({ value, duration = 1000, suffix = '' }) {
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    if (value === 0) { setDisplay(0); return; }

    let startTime = null;
    const startValue = 0;

    const animate = (timestamp) => {
      if (!startTime) startTime = timestamp;
      const progress = Math.min((timestamp - startTime) / duration, 1);
      // Ease-out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(Math.round(startValue + (value - startValue) * eased));

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    requestAnimationFrame(animate);
  }, [value, duration]);

  return <>{display}{suffix}</>;
}

export default function AnalyticsPanel() {
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchAnalytics = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getAnalytics();
      setAnalytics(data);
    } catch (err) {
      console.error('Failed to load analytics:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAnalytics();
  }, []);

  // Prepare pie chart data
  const getPieData = () => {
    if (!analytics?.class_distribution) return null;

    const entries = Object.entries(analytics.class_distribution)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10); // Top 10 classes

    // Curated color palette
    const colors = [
      '#00d4ff', '#10b981', '#fbbf24', '#ef4444', '#a855f7',
      '#f97316', '#06b6d4', '#ec4899', '#84cc16', '#6366f1',
    ];

    return {
      labels: entries.map(([name]) => name),
      datasets: [{
        data: entries.map(([, count]) => count),
        backgroundColor: colors.slice(0, entries.length).map(c => c + '33'),
        borderColor: colors.slice(0, entries.length),
        borderWidth: 2,
        hoverBorderWidth: 3,
        hoverOffset: 8,
      }],
    };
  };

  const pieOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right',
        labels: {
          color: '#94a3b8',
          font: { family: 'Inter', size: 11 },
          padding: 12,
          usePointStyle: true,
          pointStyleWidth: 8,
        },
      },
      tooltip: {
        backgroundColor: '#1a2236',
        titleColor: '#f1f5f9',
        bodyColor: '#94a3b8',
        borderColor: '#334155',
        borderWidth: 1,
        cornerRadius: 8,
        padding: 12,
        titleFont: { family: 'Inter', weight: '600' },
        bodyFont: { family: 'Inter' },
      },
    },
  };

  const pieData = getPieData();

  if (loading) {
    return (
      <div className="analytics-loading">
        <div className="spinner" />
        <span className="text-sm text-muted">Loading analytics...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="analytics-error card">
        <p className="text-sm text-danger">{error}</p>
        <button className="btn btn-secondary mt-sm" onClick={fetchAnalytics}>Retry</button>
      </div>
    );
  }

  if (!analytics || analytics.total_predictions === 0) {
    return (
      <div className="analytics-empty card">
        <BarChart3 size={40} style={{ color: 'var(--text-muted)', opacity: 0.3 }} />
        <p className="text-sm text-muted mt-md">No analytics data yet</p>
        <p className="text-xs text-muted">Run some predictions to see your dashboard</p>
      </div>
    );
  }

  return (
    <div className="analytics-panel stagger-children">
      {/* Stat Cards Row */}
      <div className="analytics-stats">
        {/* Total Processed */}
        <div className="stat-card stat-card-blue">
          <div className="flex items-center gap-sm">
            <div className="stat-icon" style={{ background: 'var(--accent-blue-dim)', color: 'var(--accent-blue)' }}>
              <BarChart3 size={18} />
            </div>
            <span className="stat-label">Total Processed</span>
          </div>
          <div className="stat-value" style={{ color: 'var(--accent-blue)' }}>
            <AnimatedCounter value={analytics.total_predictions} />
          </div>
          <span className="text-xs text-muted">signs analyzed</span>
        </div>

        {/* Average Confidence */}
        <div className="stat-card stat-card-green">
          <div className="flex items-center gap-sm">
            <div className="stat-icon" style={{ background: 'var(--accent-green-dim)', color: 'var(--accent-green)' }}>
              <Target size={18} />
            </div>
            <span className="stat-label">Avg Confidence</span>
          </div>
          <div className="stat-value" style={{ color: 'var(--accent-green)' }}>
            <AnimatedCounter value={Math.round((analytics.avg_confidence || 0) * 100)} suffix="%" />
          </div>
          <span className="text-xs text-muted">mean prediction score</span>
        </div>

        {/* Top Class */}
        <div className="stat-card stat-card-yellow">
          <div className="flex items-center gap-sm">
            <div className="stat-icon" style={{ background: 'var(--accent-yellow-dim)', color: 'var(--accent-yellow)' }}>
              <TrendingUp size={18} />
            </div>
            <span className="stat-label">Most Detected</span>
          </div>
          <div className="stat-value-text">
            {analytics.class_distribution
              ? Object.entries(analytics.class_distribution).sort((a, b) => b[1] - a[1])[0]?.[0] || '—'
              : '—'}
          </div>
          <span className="text-xs text-muted">most frequent class</span>
        </div>
      </div>

      {/* Pie Chart Section */}
      {pieData && (
        <div className="analytics-chart card">
          <div className="flex items-center justify-between mb-lg">
            <div>
              <h3 className="heading-md">Class Distribution</h3>
              <p className="text-xs text-muted mt-xs">Top 10 most frequently detected signs</p>
            </div>
            <button className="btn btn-ghost" onClick={fetchAnalytics}>
              <RefreshCw size={14} />
            </button>
          </div>
          <div className="pie-chart-container">
            <Pie data={pieData} options={pieOptions} />
          </div>
        </div>
      )}

      <style>{`
        .analytics-panel {
          display: flex;
          flex-direction: column;
          gap: var(--space-lg);
        }

        .analytics-stats {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: var(--space-md);
        }

        .analytics-loading,
        .analytics-empty {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          padding: 48px;
          text-align: center;
        }

        .stat-icon {
          width: 36px;
          height: 36px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: var(--radius-md);
          flex-shrink: 0;
        }

        .stat-value-text {
          font-size: 1rem;
          font-weight: 700;
          color: var(--accent-yellow);
          margin: var(--space-sm) 0;
          line-height: 1.3;
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }

        .pie-chart-container {
          height: 300px;
          position: relative;
        }

        .analytics-chart {
          min-height: 0;
        }

        @media (max-width: 768px) {
          .analytics-stats {
            grid-template-columns: 1fr;
          }

          .pie-chart-container {
            height: 250px;
          }
        }

        @media (min-width: 769px) and (max-width: 1024px) {
          .analytics-stats {
            grid-template-columns: repeat(2, 1fr);
          }
        }
      `}</style>
    </div>
  );
}
