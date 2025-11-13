import React, { useState } from "react";
import '../styles/AdminDashboard.css';

export default function AdminDashboard({ onLogout, onTakeAttendance, starting, onNavigate }) {
  const [isOpen, setIsOpen] = useState(false);

  const handleTakeAttendance = async () => {
    if (typeof onTakeAttendance === "function") {
      await onTakeAttendance();
    }
  };

  return (
    <div className="admin-dashboard-wrapper">

      {/* Mobile hamburger button */}
      {!isOpen && (
        <button 
          className="mobile-menu-btn"
          onClick={() => setIsOpen(true)}
          aria-label="Open menu"
        >
          ☰
        </button>
      )}

      {/* Sidebar */}
      <aside className={`admin-sidebar ${isOpen ? 'open' : 'closed'}`}>
        
        {/* Sidebar Header */}
        <div className="sidebar-header">
          <div className="logo-section">
            <div className="logo-icon">A</div>
            {isOpen && <h2 className="logo-text">Admin Panel</h2>}
          </div>

          {/* Desktop open/close toggle */}
          <button
            className="desktop-toggle-btn"
            onClick={() => setIsOpen(!isOpen)}
            aria-label={isOpen ? "Close sidebar" : "Open sidebar"}
          >
            {isOpen ? "←" : "→"}
          </button>

          {/* Mobile close X */}
          {isOpen && (
            <button 
              className="mobile-close-btn inline-close"
              onClick={() => setIsOpen(false)}
              aria-label="Close menu"
            >
              ✕
            </button>
          )}
        </div>

        {/* Navigation */}
        <nav className="sidebar-nav">
          <div className="nav-section">
            <button
              className="nav-item"
              onClick={handleTakeAttendance}
              disabled={typeof onTakeAttendance !== 'function' || starting}
            >
              <span className="nav-icon">📸</span>
              {isOpen && <span className="nav-text">{starting ? "Starting…" : "Take Attendance"}</span>}
            </button>

            <button className="nav-item" onClick={() => onNavigate?.("studentRegis")}>
              <span className="nav-icon">👤</span>
              {isOpen && <span className="nav-text">Register Student</span>}
            </button>

            <button className="nav-item" onClick={() => onNavigate?.("teachregis")}>
              <span className="nav-icon">👨‍🏫</span>
              {isOpen && <span className="nav-text">Register Teacher</span>}
            </button>
          </div>

          <div className="nav-section">
            <button className="nav-item"><span className="nav-icon">📋</span>{isOpen && <span className="nav-text">Student List</span>}</button>
            <button className="nav-item"><span className="nav-icon">🏫</span>{isOpen && <span className="nav-text">Manage Classes</span>}</button>
            <button className="nav-item"><span className="nav-icon">📊</span>{isOpen && <span className="nav-text">Reports</span>}</button>
            <button className="nav-item"><span className="nav-icon">📈</span>{isOpen && <span className="nav-text">Analytics</span>}</button>
            <button className="nav-item"><span className="nav-icon">📝</span>{isOpen && <span className="nav-text">Logs</span>}</button>
          </div>
        </nav>

        <button className="sidebar-logout" onClick={onLogout}>
          <span className="nav-icon">🚪</span>
          {isOpen && <span className="nav-text">Logout</span>}
        </button>
      </aside>

      {/* Content */}
      <main className="admin-main-content">
        <div className="content-header">
          <h1>Admin Dashboard</h1>
          <p>Welcome, Admin. You can manage users and system settings here.</p>
        </div>
      </main>
    </div>
  );
}
