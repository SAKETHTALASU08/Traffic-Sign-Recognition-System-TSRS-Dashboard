/* ============================================================
   HistoryPage — Module B: Prediction History & Analytics
   Combines analytics dashboard with sortable history table
   ============================================================ */

import AnalyticsPanel from '../components/AnalyticsPanel';
import HistoryTable from '../components/HistoryTable';

export default function HistoryPage() {
  return (
    <div className="history-page">
      {/* Page Header */}
      <div className="page-header animate-fade-in-up">
        <div>
          <h1 className="heading-xl">
            <span className="gradient-text">History & Analytics</span>
          </h1>
          <p className="text-secondary mt-xs">
            View past predictions, performance metrics, and class distribution insights
          </p>
        </div>
      </div>

      {/* Analytics Dashboard */}
      <div className="animate-fade-in-up" style={{ animationDelay: '0.1s' }}>
        <AnalyticsPanel />
      </div>

      {/* History Table */}
      <div className="animate-fade-in-up mt-xl" style={{ animationDelay: '0.2s' }}>
        <HistoryTable />
      </div>

      <style>{`
        .history-page {
          max-width: 1100px;
          margin: 0 auto;
        }

        .page-header {
          margin-bottom: var(--space-xl);
        }
      `}</style>
    </div>
  );
}
