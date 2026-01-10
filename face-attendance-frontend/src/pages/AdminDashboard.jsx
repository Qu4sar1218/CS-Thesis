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
            <button className="nav-item" onClick={() => onNavigate?.("studentList")}><span className="nav-icon">📋</span>{isOpen && <span className="nav-text">Student List</span>}</button>
            <button className="nav-item" onClick={() => onNavigate?.("teacherList")}><span className="nav-icon">👨‍🏫</span>{isOpen && <span className="nav-text">Teacher List</span>}</button>
            <button className="nav-item" onClick={() => onNavigate?.("manageClasses")}><span className="nav-icon">🏫</span>{isOpen && <span className="nav-text">Manage Classes</span>}</button>
            <button className="nav-item" onClick={() => onNavigate?.("reports")}><span className="nav-icon">📊</span>{isOpen && <span className="nav-text">Reports</span>}</button>
            <button className="nav-item" onClick={() => onNavigate?.("analytics")}><span className="nav-icon">📈</span>{isOpen && <span className="nav-text">Analytics</span>}</button>
            <button className="nav-item" onClick={() => onNavigate?.("systemLogs")}><span className="nav-icon">📝</span>{isOpen && <span className="nav-text">System Logs</span>}</button>
            <button className="nav-item" onClick={() => onNavigate?.("settings")}><span className="nav-icon">⚙️</span>{isOpen && <span className="nav-text">Settings</span>}</button>
            <button className="nav-item" onClick={() => onNavigate?.("notifications")}><span className="nav-icon">🔔</span>{isOpen && <span className="nav-text">Notifications</span>}</button>
            <button className="nav-item" onClick={() => onNavigate?.("receiptVerification")}><span className="nav-icon">🧾</span>{isOpen && <span className="nav-text">Receipt Verification</span>}</button>
            <button className="nav-item" onClick={() => onNavigate?.("eventManagement")}><span className="nav-icon">🎉</span>{isOpen && <span className="nav-text">Manage Events</span>}</button>
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

        {/* Quick Stats */}
        <div className="admin-quick-stats">
          <div className="stat-card">
            <h3>Total Students</h3>
            <div className="stat-value">2</div>
            <div className="stat-description">Active enrollments</div>
          </div>
          <div className="stat-card">
            <h3>Total Teachers</h3>
            <div className="stat-value">2</div>
            <div className="stat-description">Faculty members</div>
          </div>
          <div className="stat-card">
            <h3>Today's Attendance</h3>
            <div className="stat-value">100%</div>
            <div className="stat-description">Average rate</div>
          </div>
          <div className="stat-card">
            <h3>Active Classes</h3>
            <div className="stat-value">2</div>
            <div className="stat-description">Currently running</div>
          </div>
        </div>

        {/* Main Dashboard Grid */}
        <div className="dashboard-grid">
          <div className="dashboard-card">
            <h3>Attendance Overview</h3>
            <div className="graph-placeholder">📊 Attendance trends chart</div>
            <div className="card-actions">
              <button className="action-btn">View Details</button>
            </div>
          </div>
          <div className="dashboard-card">
            <h3>Student Analytics</h3>
            <div className="graph-placeholder">📈 Student performance metrics</div>
            <div className="card-actions">
              <button className="action-btn">Generate Report</button>
            </div>
          </div>
          <div className="dashboard-card system-health-card">
            <h3>System Health</h3>
            <div className="system-status">
              <div className="status-subcard">
                <div className="status-icon">🤖</div>
                <div className="status-content">
                  <div className="status-name">Face Recognition</div>
                  <div className="status-detail">AI Model Service</div>
                  <div className="status-usage"></div>
                </div>
                <div className="status-indicator">
                  <span className="status-label online">Online</span>
                </div>
              </div>
              <div className="status-subcard">
                <div className="status-icon">🗄️</div>
                <div className="status-content">
                  <div className="status-name">Database</div>
                  <div className="status-detail">PostgreSQL Server</div>
                  <div className="status-usage"></div>
                </div>
                <div className="status-indicator">
                  <span className="status-label online">Online</span>
                </div>
              </div>
              <div className="status-subcard">
                <div className="status-icon">💾</div>
                <div className="status-content">
                  <div className="status-name">Storage</div>
                  <div className="status-detail">File System</div>
                  <div className="status-usage">78% used</div>
                </div>
                <div className="status-indicator">
                  <span className="status-label warning">Warning</span>
                </div>
              </div>
            </div>
          </div>
          <div className="dashboard-card">
            <h3>Recent Activity</h3>
            <div className="activity-feed">
              <div className="activity-item">
                <span className="activity-time">Just now</span>
                <span className="activity-text">Sample data loaded to MongoDB</span>
              </div>
              <div className="activity-item">
                <span className="activity-time">5 min ago</span>
                <span className="activity-text">Ana Rodriguez marked present</span>
              </div>
              <div className="activity-item">
                <span className="activity-time">10 min ago</span>
                <span className="activity-text">Miguel Lopez marked present</span>
              </div>
            </div>
          </div>
        </div>

        {/* Additional Features */}
        <div className="admin-features">
          <div className="feature-card">
            <h3>🚀 Quick Actions</h3>
            <div className="feature-buttons">
              <button className="feature-btn primary">Bulk Import Students</button>
              <button className="feature-btn secondary" onClick={() => onNavigate?.("receiptVerification")}>Receipt Verification</button>
              <button className="feature-btn secondary">Export Reports</button>
              <button className="feature-btn secondary">System Maintenance</button>
            </div>
          </div>
          <div className="feature-card">
            <h3>📋 Pending Tasks</h3>
            <div className="task-list">
              <div className="task-item">
                <input type="checkbox" id="task1" />
                <label htmlFor="task1">Test MongoDB registration endpoints</label>
              </div>
              <div className="task-item">
                <input type="checkbox" id="task2" />
                <label htmlFor="task2">Configure online deployment</label>
              </div>
              <div className="task-item">
                <input type="checkbox" id="task3" />
                <label htmlFor="task3">Add authentication system</label>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
