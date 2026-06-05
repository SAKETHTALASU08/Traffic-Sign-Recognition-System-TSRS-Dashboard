/* ============================================================
   Sidebar — Navigation sidebar with glow effects and 
   automotive-inspired styling
   ============================================================ */

import { useState, useEffect } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  ScanLine,
  History,
  Activity,
  ChevronLeft,
  ChevronRight,
  Shield,
  Cpu,
  Menu,
  X
} from 'lucide-react';

export default function Sidebar() {
  const location = useLocation();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  // Close mobile sidebar on route change
  useEffect(() => {
    setMobileOpen(false);
  }, [location.pathname]);

  // Navigation items configuration
  const navItems = [
    {
      path: '/',
      icon: ScanLine,
      label: 'Inference',
      description: 'Detect traffic signs',
    },
    {
      path: '/history',
      icon: History,
      label: 'History',
      description: 'Prediction logs',
    },
  ];

  return (
    <>
      {/* Mobile hamburger button */}
      <button
        className="mobile-menu-btn"
        onClick={() => setMobileOpen(!mobileOpen)}
        aria-label="Toggle menu"
      >
        {mobileOpen ? <X size={20} /> : <Menu size={20} />}
      </button>

      {/* Mobile overlay */}
      {mobileOpen && (
        <div
          className="sidebar-overlay"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`sidebar ${collapsed ? 'sidebar-collapsed' : ''} ${mobileOpen ? 'sidebar-mobile-open' : ''}`}
      >
        {/* Logo / Brand */}
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <div className="sidebar-logo-icon">
              <Shield size={collapsed ? 20 : 24} />
            </div>
            {!collapsed && (
              <div className="sidebar-brand">
                <span className="sidebar-brand-name">TSRS</span>
                <span className="sidebar-brand-subtitle">Dashboard</span>
              </div>
            )}
          </div>
        </div>

        {/* System Status Indicator */}
        <div className="sidebar-status">
          <div className="status-dot" />
          {!collapsed && <span className="text-xs text-muted">System Online</span>}
        </div>

        {/* Navigation Links */}
        <nav className="sidebar-nav">
          <div className="sidebar-section-label">
            {!collapsed && 'MODULES'}
          </div>
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.path === '/'}
              className={({ isActive }) =>
                `sidebar-link ${isActive ? 'sidebar-link-active' : ''}`
              }
              data-tooltip={collapsed ? item.label : undefined}
            >
              <item.icon size={20} className="sidebar-link-icon" />
              {!collapsed && (
                <div className="sidebar-link-text">
                  <span className="sidebar-link-label">{item.label}</span>
                  <span className="sidebar-link-desc">{item.description}</span>
                </div>
              )}
              {/* Active indicator bar */}
              <span className="sidebar-link-indicator" />
            </NavLink>
          ))}
        </nav>

        {/* Footer with system info */}
        <div className="sidebar-footer">
          <div className="sidebar-system-info">
            <Cpu size={14} />
            {!collapsed && <span className="text-xs">MobileNetV2</span>}
          </div>
          <div className="sidebar-system-info">
            <Activity size={14} />
            {!collapsed && <span className="text-xs">43 Classes</span>}
          </div>

          {/* Collapse toggle (desktop only) */}
          <button
            className="sidebar-collapse-btn"
            onClick={() => setCollapsed(!collapsed)}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
          </button>
        </div>

        <style>{`
          /* ---- Mobile Menu Button ---- */
          .mobile-menu-btn {
            display: none;
            position: fixed;
            top: 16px;
            left: 16px;
            z-index: 1001;
            padding: 10px;
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            color: var(--text-primary);
            cursor: pointer;
            transition: all var(--transition-fast);
          }

          .mobile-menu-btn:hover {
            background: var(--bg-elevated);
            border-color: var(--accent-blue);
          }

          .sidebar-overlay {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(4px);
            z-index: 999;
          }

          /* ---- Sidebar Container ---- */
          .sidebar {
            position: fixed;
            top: 0;
            left: 0;
            bottom: 0;
            width: var(--sidebar-width);
            background: var(--bg-secondary);
            border-right: 1px solid var(--border-subtle);
            display: flex;
            flex-direction: column;
            z-index: 1000;
            transition: width var(--transition-base);
            overflow-x: hidden;
          }

          .sidebar-collapsed {
            width: var(--sidebar-collapsed);
          }

          /* ---- Header / Logo ---- */
          .sidebar-header {
            padding: 20px 16px 12px;
            border-bottom: 1px solid var(--border-subtle);
          }

          .sidebar-logo {
            display: flex;
            align-items: center;
            gap: 12px;
          }

          .sidebar-logo-icon {
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #00b4d8, #00d4ff);
            border-radius: var(--radius-md);
            color: var(--text-inverse);
            flex-shrink: 0;
            box-shadow: 0 2px 12px var(--accent-blue-dim);
          }

          .sidebar-brand {
            display: flex;
            flex-direction: column;
          }

          .sidebar-brand-name {
            font-size: 1.1rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            color: var(--text-primary);
          }

          .sidebar-brand-subtitle {
            font-size: 0.65rem;
            font-weight: 500;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.1em;
          }

          /* ---- Status ---- */
          .sidebar-status {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 12px 20px;
          }

          .status-dot {
            width: 8px;
            height: 8px;
            background: var(--accent-green);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--accent-green);
            animation: pulse-ring 2s infinite;
            flex-shrink: 0;
          }

          /* ---- Navigation ---- */
          .sidebar-nav {
            flex: 1;
            padding: 8px 12px;
            overflow-y: auto;
          }

          .sidebar-section-label {
            font-size: 0.65rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            padding: 8px 8px 4px;
            min-height: 24px;
          }

          .sidebar-link {
            position: relative;
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px 12px;
            margin: 2px 0;
            border-radius: var(--radius-md);
            color: var(--text-secondary);
            text-decoration: none;
            transition: all var(--transition-fast);
            overflow: hidden;
          }

          .sidebar-link:hover {
            background: var(--bg-elevated);
            color: var(--text-primary);
          }

          .sidebar-link-active {
            background: rgba(0, 212, 255, 0.08);
            color: var(--accent-blue);
          }

          .sidebar-link-active .sidebar-link-icon {
            color: var(--accent-blue);
          }

          .sidebar-link-indicator {
            position: absolute;
            left: 0;
            top: 50%;
            transform: translateY(-50%);
            width: 3px;
            height: 0;
            background: var(--accent-blue);
            border-radius: 0 2px 2px 0;
            transition: height var(--transition-fast);
            box-shadow: 0 0 8px var(--accent-blue);
          }

          .sidebar-link-active .sidebar-link-indicator {
            height: 60%;
          }

          .sidebar-link-icon {
            flex-shrink: 0;
            transition: color var(--transition-fast);
          }

          .sidebar-link-text {
            display: flex;
            flex-direction: column;
            min-width: 0;
          }

          .sidebar-link-label {
            font-size: 0.875rem;
            font-weight: 600;
          }

          .sidebar-link-desc {
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-top: 1px;
          }

          /* ---- Footer ---- */
          .sidebar-footer {
            padding: 12px 16px;
            border-top: 1px solid var(--border-subtle);
            display: flex;
            flex-direction: column;
            gap: 6px;
          }

          .sidebar-system-info {
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--text-muted);
            padding: 4px 4px;
          }

          .sidebar-collapse-btn {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 8px;
            margin-top: 4px;
            background: transparent;
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            color: var(--text-muted);
            cursor: pointer;
            transition: all var(--transition-fast);
          }

          .sidebar-collapse-btn:hover {
            background: var(--bg-elevated);
            color: var(--text-primary);
            border-color: var(--border-medium);
          }

          /* ---- Responsive ---- */
          @media (max-width: 1024px) {
            .mobile-menu-btn {
              display: flex;
            }

            .sidebar-overlay {
              display: block;
            }

            .sidebar {
              transform: translateX(-100%);
              width: var(--sidebar-width) !important;
            }

            .sidebar-mobile-open {
              transform: translateX(0);
            }

            .sidebar-collapse-btn {
              display: none;
            }
          }
        `}</style>
      </aside>
    </>
  );
}
