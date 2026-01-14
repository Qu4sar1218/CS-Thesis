import { useState, useEffect, useCallback } from "react";
import '../styles/TeacherDashboard.css';

export default function TeacherDashboard({ onLogout, onTakeAttendance, starting, userInfo, onNavigate }) {
  const [isOpen, setIsOpen] = useState(false);
  const [teacherClasses, setTeacherClasses] = useState([]);
  const [loadingClasses, setLoadingClasses] = useState(true);
  const [showClassRoster, setShowClassRoster] = useState(false);
  const [teacherName, setTeacherName] = useState('');


  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:8000";

  const handleTakeAttendance = async () => {
    if (typeof onTakeAttendance === "function") {
      await onTakeAttendance();
    }
  };

  const fetchClasses = useCallback(async () => {
    console.log("🔍 fetchClasses called with userInfo:", userInfo);
    if (!userInfo?.user_id) {
      console.error("❌ No teacher ID available in userInfo");
      console.log("userInfo object:", userInfo);
      return;
    }

    console.log(`📡 Fetching classes for teacher ID: ${userInfo.user_id}`);
    setLoadingClasses(true);
    try {
      const url = `${BACKEND_URL}/classes/teacher/${userInfo.user_id}`;
      console.log(`🌐 Making request to: ${url}`);
      const response = await fetch(url);
      console.log(`📥 Response status: ${response.status}`);

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`❌ Failed to fetch classes: ${response.status} - ${errorText}`);
        throw new Error(`Failed to fetch classes: ${response.status}`);
      }

      const data = await response.json();
      console.log("📦 Received data:", data);
      console.log(`📚 Found ${data.classes?.length || 0} classes`);
      setTeacherClasses(data.classes || []);
    } catch (error) {
      console.error("💥 Error fetching classes:", error);
      setTeacherClasses([]);
    } finally {
      setLoadingClasses(false);
    }
  }, [userInfo, BACKEND_URL]);

  useEffect(() => {
    fetchClasses();
  }, [fetchClasses]);

  useEffect(() => {
    const fetchTeacherName = async () => {
      console.log('🔍 fetchTeacherName called with userInfo:', userInfo);
      if (!userInfo?.user_id) {
        console.log('❌ No user_id in userInfo');
        return;
      }

      console.log(`📡 Fetching teacher name for ID: ${userInfo.user_id}`);
      try {
        let url = `${BACKEND_URL}/teachers/${userInfo.user_id}`;
        console.log(`🌐 Making request to: ${url}`);
        let response = await fetch(url);
        console.log(`📥 Response status: ${response.status}`);

        if (!response.ok && userInfo.email) {
          console.log(`🔄 Retrying with email: ${userInfo.email}`);
          url = `${BACKEND_URL}/teachers/${userInfo.email}`;
          console.log(`🌐 Making request to: ${url}`);
          response = await fetch(url);
          console.log(`📥 Response status: ${response.status}`);
        }

        if (response.ok) {
          const teacherData = await response.json();
          console.log('📦 Teacher data received:', teacherData);
          const fullName = `${teacherData.first_name || teacherData.firstName || ''} ${teacherData.last_name || teacherData.lastName || ''}`.trim();
          if (fullName) {
            console.log(`✅ Setting teacher name to: "${fullName}"`);
            setTeacherName(fullName);
          } else {
            console.log('⚠️ No full name available in teacher data');
          }
        } else {
          const errorText = await response.text();
          console.error(`❌ Failed to fetch teacher name: ${response.status} - ${errorText}`);
        }
      } catch (error) {
        console.error('💥 Error fetching teacher name:', error);
      }
    };

    fetchTeacherName();
  }, [userInfo, BACKEND_URL]);

  // eslint-disable-next-line no-unused-vars
  const handleClassRoster = () => {
    if (!showClassRoster) {
      fetchClasses();
    }
    setShowClassRoster(!showClassRoster);
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

            <button className="nav-item" onClick={() => onNavigate("teacherClassRoster")}>
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
          <h1>Welcome Teacher {teacherName ? teacherName : 'Teacher'}</h1>
          <p>Take attendance, review students, and manage classes.</p>
        </div>

        {/* Teacher-specific content */}
        <div className="teacher-overview">
          <div className="overview-card">
            <h3>My Subjects</h3>
            <div className="class-list">
              {loadingClasses ? (
                <div className="loading-classes">Loading subjects...</div>
              ) : teacherClasses.length === 0 ? (
                <div className="no-classes">No subjects assigned yet.</div>
              ) : (
                teacherClasses.map((cls) => {
                  const parts = cls.schedule.split(' ');
                  const days = parts[0];
                  const timeRange = parts.slice(1).join(' ');
                  const [startTime, endTime] = timeRange.split('-');
                  return (
                    <div key={cls._id} className="class-item">
                      <span className="class-time">{startTime} - {endTime}</span>
                      <span className="class-name">{cls.class_name} ({cls.class_code})</span>
                      <span className="class-status">{days} - {cls.room}</span>
                    </div>
                  );
                })
              )}
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
