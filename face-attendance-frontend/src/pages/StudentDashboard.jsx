import React, { useState } from "react";
import '../styles/StudentDashboard.css';

export default function StudentDashboard({ onLogout, onFaceRecognition, onNavigate, userInfo }) {
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
              <span className="nav-icon">📚</span>
              {isOpen && <span className="nav-text">Assignments</span>}
            </button>

            <button className="nav-item">
              <span className="nav-icon">📝</span>
              {isOpen && <span className="nav-text">Grades</span>}
            </button>

            <button className="nav-item">
              <span className="nav-icon">👤</span>
              {isOpen && <span className="nav-text">Profile</span>}
            </button>

            <button className="nav-item" onClick={() => onNavigate && onNavigate("settings")}>
              <span className="nav-icon">⚙️</span>
              {isOpen && <span className="nav-text">Settings</span>}
            </button>

            <button className="nav-item">
              <span className="nav-icon">💬</span>
              {isOpen && <span className="nav-text">Messages</span>}
            </button>

            <button className="nav-item" onClick={() => onNavigate && onNavigate("receiptSubmission")}>
              <span className="nav-icon">🧾</span>
              {isOpen && <span className="nav-text">Submit Receipt</span>}
            </button>

            <button className="nav-item">
              <span className="nav-icon">❓</span>
              {isOpen && <span className="nav-text">Help & Support</span>}
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
          <h1>Welcome, Student{userInfo?.full_name ? ` ${userInfo.full_name}` : ''}</h1>
          <p>View your attendance and schedule here.</p>
        </div>

        {/* Student-specific content */}
        <div className="student-stats">
          <div className="stat-card">
            <h3>Attendance Rate</h3>
            <div className="stat-value">95%</div>
            <div className="stat-description">This semester</div>
          </div>
          <div className="stat-card">
            <h3>Classes Today</h3>
            <div className="stat-value">4</div>
            <div className="stat-description">Scheduled classes</div>
          </div>
          <div className="stat-card">
            <h3>Next Class</h3>
            <div className="stat-value">Math</div>
            <div className="stat-description">In 30 minutes</div>
          </div>
          <div className="stat-card">
            <h3>Pending Assignments</h3>
            <div className="stat-value">3</div>
            <div className="stat-description">Due this week</div>
          </div>
        </div>

        {/* Today's Schedule */}
        <div className="student-schedule">
          <h2>Today's Schedule</h2>
          <div className="schedule-list">
            <div className="schedule-item">
              <div className="schedule-time">9:00 AM</div>
              <div className="schedule-details">
                <div className="schedule-subject">Mathematics</div>
                <div className="schedule-room">Room 201</div>
                <div className="schedule-status attended">✓ Attended</div>
              </div>
            </div>
            <div className="schedule-item">
              <div className="schedule-time">10:30 AM</div>
              <div className="schedule-details">
                <div className="schedule-subject">Physics</div>
                <div className="schedule-room">Lab 3</div>
                <div className="schedule-status upcoming">Next</div>
              </div>
            </div>
            <div className="schedule-item">
              <div className="schedule-time">2:00 PM</div>
              <div className="schedule-details">
                <div className="schedule-subject">English Literature</div>
                <div className="schedule-room">Room 105</div>
                <div className="schedule-status pending">Pending</div>
              </div>
            </div>
          </div>
        </div>

        <div className="recent-activity">
          <h2>Recent Activity</h2>
          <div className="activity-list">
            <div className="activity-item">
              <span className="activity-icon">✅</span>
              <div className="activity-details">
                <div className="activity-title">Attendance Marked</div>
                <div className="activity-time">Math Class - 2 hours ago</div>
              </div>
            </div>
            <div className="activity-item">
              <span className="activity-icon">📚</span>
              <div className="activity-details">
                <div className="activity-title">Assignment Submitted</div>
                <div className="activity-time">Physics Lab Report - 1 day ago</div>
              </div>
            </div>
            <div className="activity-item">
              <span className="activity-icon">📝</span>
              <div className="activity-details">
                <div className="activity-title">Grade Posted</div>
                <div className="activity-time">Chemistry Quiz - A- (92%)</div>
              </div>
            </div>
            <div className="activity-item">
              <span className="activity-icon">💬</span>
              <div className="activity-details">
                <div className="activity-title">New Message</div>
                <div className="activity-time">From Teacher Smith - 3 hours ago</div>
              </div>
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="student-actions">
          <h2>Quick Actions</h2>
          <div className="action-buttons">
            <button className="action-btn primary">View Full Schedule</button>
            <button className="action-btn secondary">Check Grades</button>
            <button className="action-btn secondary">Contact Teacher</button>
          </div>
        </div>
      </main>
    </div>
  );
}