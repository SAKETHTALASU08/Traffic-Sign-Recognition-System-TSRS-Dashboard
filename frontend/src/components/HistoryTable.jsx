/* ============================================================
   HistoryTable — Sortable, filterable prediction history table
   Features: column sorting, pagination, thumbnail previews
   ============================================================ */

import { useState, useEffect, useCallback } from 'react';
import {
  ArrowUpDown, ArrowUp, ArrowDown,
  ChevronLeft, ChevronRight,
  RefreshCw, Clock, Filter
} from 'lucide-react';
import { getHistory } from '../api/client';
import ConfidenceBar from './ConfidenceBar';

const PAGE_SIZE = 15;

export default function HistoryTable() {
  const [predictions, setPredictions] = useState([]);
  const [totalCount, setTotalCount] = useState(0);
  const [page, setPage] = useState(0);
  const [sortBy, setSortBy] = useState('timestamp');
  const [order, setOrder] = useState('desc');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const totalPages = Math.ceil(totalCount / PAGE_SIZE);

  /**
   * Fetch history from backend
   */
  const fetchHistory = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getHistory({
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
        sort_by: sortBy,
        order,
      });
      setPredictions(data.predictions || []);
      setTotalCount(data.total_count || 0);
    } catch (err) {
      console.error('Failed to load history:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [page, sortBy, order]);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  /**
   * Handle column header click for sorting
   */
  const handleSort = (column) => {
    if (sortBy === column) {
      setOrder(prev => prev === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(column);
      setOrder('desc');
    }
    setPage(0);
  };

  /**
   * Render sort indicator icon
   */
  const SortIcon = ({ column }) => {
    if (sortBy !== column) return <ArrowUpDown size={12} style={{ opacity: 0.3 }} />;
    return order === 'asc'
      ? <ArrowUp size={12} style={{ color: 'var(--accent-blue)' }} />
      : <ArrowDown size={12} style={{ color: 'var(--accent-blue)' }} />;
  };

  /**
   * Format timestamp for display
   */
  const formatTime = (ts) => {
    try {
      const d = new Date(ts);
      return d.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
    } catch {
      return ts;
    }
  };

  // Column definitions
  const columns = [
    { key: 'timestamp', label: 'Timestamp', sortable: true },
    { key: 'thumbnail', label: 'Image', sortable: false },
    { key: 'class_name', label: 'Predicted Class', sortable: true },
    { key: 'confidence', label: 'Confidence', sortable: true },
    { key: 'latency_ms', label: 'Latency', sortable: true },
  ];

  return (
    <div className="history-table-wrapper">
      {/* Table header bar */}
      <div className="table-toolbar">
        <div className="flex items-center gap-sm">
          <Clock size={16} className="text-muted" />
          <span className="heading-md">Prediction History</span>
          <span className="badge badge-blue">{totalCount} records</span>
        </div>
        <button
          className="btn btn-ghost"
          onClick={fetchHistory}
          disabled={loading}
          title="Refresh"
        >
          <RefreshCw size={14} className={loading ? 'rotating' : ''} />
          Refresh
        </button>
      </div>

      {/* Error state */}
      {error && (
        <div className="table-error">
          <p className="text-sm text-danger">{error}</p>
          <button className="btn btn-secondary" onClick={fetchHistory}>Retry</button>
        </div>
      )}

      {/* Table */}
      <div className="table-scroll-container">
        <table className="data-table" id="history-table">
          <thead>
            <tr>
              {columns.map(col => (
                <th
                  key={col.key}
                  onClick={col.sortable ? () => handleSort(col.key) : undefined}
                  style={{ cursor: col.sortable ? 'pointer' : 'default' }}
                >
                  <div className="flex items-center gap-xs">
                    {col.label}
                    {col.sortable && <SortIcon column={col.key} />}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading && predictions.length === 0 ? (
              <tr>
                <td colSpan={columns.length} style={{ textAlign: 'center', padding: '48px 16px' }}>
                  <div className="flex flex-col items-center gap-md">
                    <div className="spinner" />
                    <span className="text-sm text-muted">Loading history...</span>
                  </div>
                </td>
              </tr>
            ) : predictions.length === 0 ? (
              <tr>
                <td colSpan={columns.length} style={{ textAlign: 'center', padding: '48px 16px' }}>
                  <div className="flex flex-col items-center gap-sm">
                    <Filter size={32} style={{ color: 'var(--text-muted)', opacity: 0.4 }} />
                    <span className="text-sm text-muted">No predictions yet</span>
                    <span className="text-xs text-muted">Run an inference to see results here</span>
                  </div>
                </td>
              </tr>
            ) : (
              predictions.map((pred, idx) => (
                <tr key={pred.id || idx}>
                  <td>
                    <span className="text-sm text-mono" style={{ color: 'var(--text-secondary)' }}>
                      {formatTime(pred.timestamp)}
                    </span>
                  </td>
                  <td>
                    {pred.thumbnail ? (
                      <img
                        src={`data:image/jpeg;base64,${pred.thumbnail}`}
                        alt={pred.class_name}
                        className="table-thumbnail"
                      />
                    ) : (
                      <div className="table-thumbnail-placeholder" />
                    )}
                  </td>
                  <td>
                    <span className="text-sm" style={{ fontWeight: 600 }}>
                      {pred.class_name}
                    </span>
                  </td>
                  <td style={{ minWidth: 160 }}>
                    <ConfidenceBar confidence={pred.confidence} showLabel={false} size="sm" />
                    <span className="text-xs text-mono" style={{
                      color: pred.confidence >= 0.8 ? 'var(--accent-green)' : pred.confidence >= 0.5 ? 'var(--accent-yellow)' : 'var(--accent-red)',
                      marginTop: 4,
                      display: 'block'
                    }}>
                      {Math.round(pred.confidence * 100)}%
                    </span>
                  </td>
                  <td>
                    <span className="text-mono text-sm text-secondary">
                      {pred.latency_ms?.toFixed(1)}ms
                    </span>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="table-pagination">
          <span className="text-xs text-muted">
            Page {page + 1} of {totalPages}
          </span>
          <div className="flex gap-sm">
            <button
              className="btn btn-ghost btn-icon"
              onClick={() => setPage(p => Math.max(0, p - 1))}
              disabled={page === 0}
            >
              <ChevronLeft size={16} />
            </button>
            <button
              className="btn btn-ghost btn-icon"
              onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1}
            >
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      )}

      <style>{`
        .history-table-wrapper {
          background: var(--bg-card);
          border: 1px solid var(--border-subtle);
          border-radius: var(--radius-lg);
          overflow: hidden;
        }

        .table-toolbar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: var(--space-md) var(--space-lg);
          border-bottom: 1px solid var(--border-subtle);
        }

        .table-error {
          padding: var(--space-lg);
          display: flex;
          align-items: center;
          justify-content: space-between;
          background: var(--accent-red-dim);
        }

        .table-scroll-container {
          overflow-x: auto;
        }

        .table-thumbnail {
          width: 40px;
          height: 40px;
          border-radius: var(--radius-sm);
          object-fit: cover;
          border: 1px solid var(--border-subtle);
          background: #000;
        }

        .table-thumbnail-placeholder {
          width: 40px;
          height: 40px;
          border-radius: var(--radius-sm);
          background: var(--bg-elevated);
          border: 1px solid var(--border-subtle);
        }

        .table-pagination {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: var(--space-md) var(--space-lg);
          border-top: 1px solid var(--border-subtle);
        }

        .rotating {
          animation: rotate 1s linear infinite;
        }
      `}</style>
    </div>
  );
}
