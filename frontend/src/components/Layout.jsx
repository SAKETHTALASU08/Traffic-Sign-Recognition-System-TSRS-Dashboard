/* ============================================================
   Layout — Dashboard shell wrapping sidebar + main content area
   ============================================================ */

import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';

export default function Layout() {
  return (
    <div className="app-layout">
      {/* Sidebar navigation */}
      <Sidebar />

      {/* Main content area — pages render here via <Outlet /> */}
      <main className="main-content">
        <Outlet />
      </main>
    </div>
  );
}
