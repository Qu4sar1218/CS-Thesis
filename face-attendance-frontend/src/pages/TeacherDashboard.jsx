import React, { useState } from "react";
import '../styles/TeacherDashboard.css';

export default function TeacherDashboard({ onLogout, onTakeAttendance, starting, userInfo }) {
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
              {isOpen && <span className="nav-text">Attendance Reports</span>}
            </button>

            <button className="nav-item">
              <span className="nav-icon">📝</span>
              {isOpen && <span className="nav-text">Assignments</span>}
            </button>

            <button className="nav-item">
              <span className="nav-icon">📚</span>
              {isOpen && <span className="nav-text">Grade Book</span>}
            </button>

            <button className="nav-item">
              <span className="nav-icon">💬</span>
              {isOpen && <span className="nav-text">Messages</span>}
            </button>

            <button className="nav-item">
              <span className="nav-icon">📅</span>
              {isOpen && <span className="nav-text">Schedule</span>}
            </button>

            <button className="nav-item">
              <span className="nav-icon">📋</span>
              {isOpen && <span className="nav-text">Lesson Plans</span>}
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
          <h1>Welcome, Teacher{userInfo?.full_name && userInfo.full_name !== 'Teacher' ? ` ${userInfo.full_name}` : ''}</h1>
          <p>Take attendance, review students, and manage classes.</p>
        </div>

        {/* Teacher-specific content */}
        <div className="teacher-overview">
          <div className="overview-card">
            <h3>Today's Classes</h3>
            <div className="class-list">
              <div className="class-item">
                <span className="class-time">9:00 AM</span>
                <span className="class-name">Mathematics - Grade 10A</span>
                <span className="class-status">Attendance: 28/30</span>
              </div>
              <div className="class-item">
                <span className="class-time">11:00 AM</span>
                <span className="class-name">Physics - Grade 11B</span>
                <span className="class-status">Attendance: Pending</span>
              </div>
              <div className="class-item">
                <span className="class-time">2:00 PM</span>
                <span className="class-name">Chemistry - Grade 12A</span>
                <span className="class-status">Attendance: 25/25</span>
              </div>
            </div>
          </div>

          <div className="overview-card">
            <h3>Quick Actions</h3>
            <div className="action-buttons">
              <button className="action-btn primary">Take Attendance</button>
              <button className="action-btn secondary">View Reports</button>
              <button className="action-btn secondary">Manage Students</button>
            </div>
          </div>
        </div>

        <div className="teacher-stats">
          <div className="stat-card">
            <h3>Average Attendance</h3>
            <div className="stat-value">92%</div>
            <div className="stat-description">This month</div>
          </div>
          <div className="stat-card">
            <h3>Classes Today</h3>
            <div className="stat-value">3</div>
            <div className="stat-description">Scheduled</div>
          </div>
          <div className="stat-card">
            <h3>Students</h3>
            <div className="stat-value">85</div>
            <div className="stat-description">Total enrolled</div>
          </div>
          <div className="stat-card">
            <h3>Assignments Due</h3>
            <div className="stat-value">12</div>
            <div className="stat-description">This week</div>
          </div>
        </div>

        {/* Recent Messages */}
        <div className="teacher-messages">
          <h2>Recent Messages</h2>
          <div className="message-list">
            <div className="message-item">
              <div className="message-sender">Parent of John Doe</div>
              <div className="message-preview">Question about tomorrow's assignment...</div>
              <div className="message-time">5 min ago</div>
            </div>
            <div className="message-item">
              <div className="message-sender">Admin Office</div>
              <div className="message-preview">Schedule change for next week</div>
              <div className="message-time">2 hours ago</div>
            </div>
            <div className="message-item">
              <div className="message-sender">Student Sarah Smith</div>
              <div className="message-preview">Request for extension on lab report</div>
              <div className="message-time">1 day ago</div>
            </div>
          </div>
        </div>

        {/* Upcoming Deadlines */}
        <div className="teacher-deadlines">
          <h2>Upcoming Deadlines</h2>
          <div className="deadline-list">
            <div className="deadline-item">
              <div className="deadline-icon">📝</div>
              <div className="deadline-details">
                <div className="deadline-title">Physics Lab Report</div>
                <div className="deadline-date">Due: Tomorrow, 11:59 PM</div>
                <div className="deadline-class">Grade 11B Physics</div>
              </div>
            </div>
            <div className="deadline-item">
              <div className="deadline-icon">📊</div>
              <div className="deadline-details">
                <div className="deadline-title">Mid-term Exam</div>
                <div className="deadline-date">Due: Friday, 3:00 PM</div>
                <div className="deadline-class">Grade 10A Mathematics</div>
              </div>
            </div>
            <div className="deadline-item">
              <div className="deadline-icon">📚</div>
              <div className="deadline-details">
                <div className="deadline-title">Grade Submissions</div>
                <div className="deadline-date">Due: Next Monday</div>
                <div className="deadline-class">All Classes</div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}