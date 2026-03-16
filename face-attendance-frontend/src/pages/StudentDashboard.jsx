import React, { useState, useEffect } from "react";
import '../styles/StudentDashboard.css';

export default function StudentDashboard({ onLogout, onFaceRecognition, onNavigate, userInfo }) {
  const [isOpen, setIsOpen] = useState(false);
  const [studentName, setStudentName] = useState('');
  const [attendanceRecords, setAttendanceRecords] = useState([]);
  const [loadingAttendance, setLoadingAttendance] = useState(true);

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:8000";

  const formatName = (first, middle, last) => {
    if (!first || !last) return `${first || ''} ${last || ''}`.trim();

    const capitalize = (s) => s.split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()).join(' ');

    const capitalizedFirst = capitalize(first);
    const capitalizedLast = capitalize(last);

    const lastParts = capitalizedLast.split(' ');

    if (lastParts.length > 1) {
      return `${capitalizedFirst} ${capitalizedLast}`.trim();
    } else {
      const middleInitial = middle ? ` ${middle.charAt(0).toUpperCase()}.` : '';
      return `${capitalizedFirst}${middleInitial} ${capitalizedLast}`.trim();
    }
  };

  useEffect(() => {
    const fetchStudentName = async () => {
      if (userInfo?.user_id) {
        try {
          const response = await fetch(`${BACKEND_URL}/students/${userInfo.user_id}`);
          if (response.ok) {
            const data = await response.json();
            const fullName = formatName(data.first_name, data.middle_name, data.last_name);
            setStudentName(fullName);
          }
        } catch (error) {
          console.error('Error fetching student name:', error);
        }
      }
    };
    fetchStudentName();
  }, [userInfo?.user_id, BACKEND_URL]);

  useEffect(() => {
    let cancelled = false;
    let poller = null;

    const fetchAttendance = async () => {
      if (!userInfo?.user_id) return;
      try {
        const response = await fetch(`${BACKEND_URL}/analytics/student/${userInfo.user_id}`);
        if (response.ok) {
          const data = await response.json();
          if (!cancelled) {
            setAttendanceRecords(data.attendance || []);
          }
        }
      } catch (error) {
        console.error('Error fetching student attendance:', error);
      } finally {
        if (!cancelled) {
          setLoadingAttendance(false);
        }
      }
    };

    fetchAttendance();
    poller = setInterval(fetchAttendance, 3000);

    return () => {
      cancelled = true;
      if (poller) clearInterval(poller);
    };
  }, [userInfo?.user_id, BACKEND_URL]);

  const today = new Date().toISOString().slice(0, 10);
  const todaysAttendance = attendanceRecords.filter((item) => item.date === today);
  const presentCount = attendanceRecords.filter((item) => ["present", "late", "PRESENT", "LATE"].includes(item.status)).length;
  const pendingCount = attendanceRecords.filter((item) => item.status === "PENDING_REVALIDATION").length;
  const attendanceRate = attendanceRecords.length > 0 ? Math.round((presentCount / attendanceRecords.length) * 100) : 0;
  const latestAttendance = attendanceRecords.length > 0 ? attendanceRecords[0] : null;

  const getStatusClass = (statusValue) => {
    const value = String(statusValue || "").toLowerCase();
    if (value.includes("pending")) return "pending";
    if (value === "present" || value === "late") return "attended";
    return "upcoming";
  };

  return (
    <div className="student-dashboard-wrapper">
      {!isOpen && (
        <button
          className="mobile-menu-btn"
          onClick={() => setIsOpen(true)}
          aria-label="Open menu"
        >
          Menu
        </button>
      )}

      <aside className={`student-sidebar ${isOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-header">
          <div className="logo-section">
            <div className="logo-icon">S</div>
            {isOpen && <h2 className="logo-text">Student Panel</h2>}
          </div>

          <button
            className="desktop-toggle-btn"
            onClick={() => setIsOpen(!isOpen)}
            aria-label={isOpen ? "Close sidebar" : "Open sidebar"}
          >
            {isOpen ? "<" : ">"}
          </button>

          {isOpen && (
            <button
              className="mobile-close-btn inline-close"
              onClick={() => setIsOpen(false)}
              aria-label="Close menu"
            >
              X
            </button>
          )}
        </div>

        <nav className="sidebar-nav" aria-label="student actions">
          <div className="nav-section">
            <button className="nav-item" onClick={() => onNavigate && onNavigate("schedule")}>
              <span className="nav-icon">S</span>
              {isOpen && <span className="nav-text">Schedule</span>}
            </button>

            <button className="nav-item">
              <span className="nav-icon">A</span>
              {isOpen && <span className="nav-text">Assignments</span>}
            </button>

            <button className="nav-item">
              <span className="nav-icon">G</span>
              {isOpen && <span className="nav-text">Grades</span>}
            </button>

            <button className="nav-item" onClick={() => onNavigate && onNavigate("profile")}>
              <span className="nav-icon">P</span>
              {isOpen && <span className="nav-text">Profile</span>}
            </button>

            <button className="nav-item" onClick={() => onNavigate && onNavigate("settings")}>
              <span className="nav-icon">T</span>
              {isOpen && <span className="nav-text">Settings</span>}
            </button>

            <button className="nav-item">
              <span className="nav-icon">M</span>
              {isOpen && <span className="nav-text">Messages</span>}
            </button>

            <button className="nav-item" onClick={() => onNavigate && onNavigate("receiptSubmission")}>
              <span className="nav-icon">R</span>
              {isOpen && <span className="nav-text">Submit Receipt</span>}
            </button>

            <button className="nav-item">
              <span className="nav-icon">?</span>
              {isOpen && <span className="nav-text">Help & Support</span>}
            </button>
          </div>
        </nav>

        <button className="sidebar-logout" onClick={onLogout}>
          <span className="nav-icon">L</span>
          {isOpen && <span className="nav-text">Logout</span>}
        </button>
      </aside>

      <main className="student-main-content">
        <div className="content-header">
          <h1>Welcome, {studentName ? ` ${studentName}` : ' Student'}</h1>
          <p>View your attendance and schedule here.</p>
        </div>

        <div className="student-stats">
          <div className="stat-card">
            <h3>Attendance Rate</h3>
            <div className="stat-value">{attendanceRate}%</div>
            <div className="stat-description">Live from database</div>
          </div>
          <div className="stat-card">
            <h3>Today&apos;s Records</h3>
            <div className="stat-value">{todaysAttendance.length}</div>
            <div className="stat-description">Updated every 3s</div>
          </div>
          <div className="stat-card">
            <h3>Pending Revalidation</h3>
            <div className="stat-value">{pendingCount}</div>
            <div className="stat-description">Monitoring mode</div>
          </div>
          <div className="stat-card">
            <h3>Latest Status</h3>
            <div className="stat-value">{latestAttendance?.status || "N/A"}</div>
            <div className="stat-description">{latestAttendance?.subject || latestAttendance?.class_id || "No records yet"}</div>
          </div>
        </div>

        <div className="student-schedule">
          <h2>Attendance Status (Live)</h2>
          <div className="schedule-list">
            {loadingAttendance ? (
              <div className="schedule-item">
                <div className="schedule-details">
                  <div className="schedule-subject">Loading attendance...</div>
                </div>
              </div>
            ) : attendanceRecords.length === 0 ? (
              <div className="schedule-item">
                <div className="schedule-details">
                  <div className="schedule-subject">No attendance records found.</div>
                </div>
              </div>
            ) : (
              attendanceRecords.slice(0, 6).map((record, idx) => (
                <div className="schedule-item" key={`${record._id || record.class_id}-${idx}`}>
                  <div className="schedule-time">{record.check_in_time || "--:--:--"}</div>
                  <div className="schedule-details">
                    <div className="schedule-subject">{record.subject || record.class_id || "Subject"}</div>
                    <div className="schedule-room">{record.date}</div>
                    <div className={`schedule-status ${getStatusClass(record.status)}`}>
                      {record.status}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
