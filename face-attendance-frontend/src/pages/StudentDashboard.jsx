import React, { useState } from "react";
import '../styles/StudentDashboard.css';

export default function StudentDashboard({ onLogout, onFaceRecognition }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="student-dashboard-wrapper">

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
      <aside className={`student-sidebar ${isOpen ? 'open' : 'closed'}`}>
        
        {/* Sidebar Header */}
        <div className="sidebar-header">
          <div className="logo-section">
            <div className="logo-icon">S</div>
            {isOpen && <h2 className="logo-text">Student Panel</h2>}
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
        <nav className="sidebar-nav" aria-label="student actions">
          <div className="nav-section">
            <button className="nav-item">
              <span className="nav-icon">📊</span>
              {isOpen && <span className="nav-text">My Attendance</span>}
            </button>

            <button className="nav-item">
              <span className="nav-icon">📅</span>
              {isOpen && <span className="nav-text">Schedule</span>}
            </button>

            <button className="nav-item">
              <span className="nav-icon">👤</span>
              {isOpen && <span className="nav-text">Profile</span>}
            </button>

            <button className="nav-item">
              <span className="nav-icon">❓</span>
              {isOpen && <span className="nav-text">Help</span>}
            </button>
          </div>
        </nav>

        <button className="sidebar-logout" onClick={onLogout}>
          <span className="nav-icon">🚪</span>
          {isOpen && <span className="nav-text">Logout</span>}
        </button>
      </aside>

      {/* Content */}
      <main className="student-main-content">
        <div className="content-header">
          <h1>Student Dashboard</h1>
          <p>Welcome, Student. View your attendance and schedule here.</p>
        </div>
      </main>
    </div>
  );
}