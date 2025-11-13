import React, { useState } from "react";
import '../styles/TeacherDashboard.css';

export default function TeacherDashboard({ onLogout, onTakeAttendance, starting }) {
  const [isOpen, setIsOpen] = useState(false);

  const handleTakeAttendance = async () => {
    if (typeof onTakeAttendance === "function") {
      await onTakeAttendance();
    }
  };

  return (
    <div className="teacher-dashboard-wrapper">

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
      <aside className={`teacher-sidebar ${isOpen ? 'open' : 'closed'}`}>
        
        {/* Sidebar Header */}
        <div className="sidebar-header">
          <div className="logo-section">
            <div className="logo-icon">T</div>
            {isOpen && <h2 className="logo-text">Teacher Panel</h2>}
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

            <button className="nav-item">
              <span className="nav-icon">👥</span>
              {isOpen && <span className="nav-text">Class Roster</span>}
            </button>

            <button className="nav-item">
              <span className="nav-icon">📊</span>
              {isOpen && <span className="nav-text">Attendance</span>}
            </button>

            <button className="nav-item">
              <span className="nav-icon">📝</span>
              {isOpen && <span className="nav-text">Assignments</span>}
            </button>

            <button className="nav-item">
              <span className="nav-icon">💬</span>
              {isOpen && <span className="nav-text">Messages</span>}
            </button>
          </div>
        </nav>

        <button className="sidebar-logout" onClick={onLogout}>
          <span className="nav-icon">🚪</span>
          {isOpen && <span className="nav-text">Logout</span>}
        </button>
      </aside>

      {/* Content */}
      <main className="teacher-main-content">
        <div className="content-header">
          <h1>Teacher Dashboard</h1>
          <p>Welcome, Teacher. Take attendance, review students, and manage classes.</p>
        </div>
      </main>
    </div>
  );
}