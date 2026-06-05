/* ============================================================
   App.jsx — Root application component with React Router
   Sets up client-side routing between Inference and History views
   ============================================================ */

import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import InferencePage from './pages/InferencePage';
import HistoryPage from './pages/HistoryPage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Layout wraps all pages with sidebar + main content area */}
        <Route element={<Layout />}>
          {/* Module A: Inference Interface */}
          <Route path="/" element={<InferencePage />} />

          {/* Module B: History & Analytics */}
          <Route path="/history" element={<HistoryPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
