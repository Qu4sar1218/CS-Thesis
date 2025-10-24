import React, { useState, useEffect } from "react";
import StatusPanel from "./pages/StatusPanel.js";
import Login from "./pages/Login.jsx";
import AdminDashboard from "./pages/AdminDashboard.jsx";
import TeacherDashboard from "./pages/TeacherDashboard.jsx";
import StudentDashboard from "./pages/StudentDashboard.jsx";
import "./App.css";

function App() {
  const [role, setRole] = useState(null);
  const [showStatusPanel, setShowStatusPanel] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [showAttendanceMode, setShowAttendanceMode] = useState(false);
  const [panelMode, setPanelMode] = useState(null);
  const [selectedSubject, setSelectedSubject] = useState(null);
  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://127.0.0.1:8000';

  const handleLogin = (userRole) => {
    setRole(userRole || 'admin');
  };

  const handleLogout = () => {
    setRole(null);
    setShowStatusPanel(false);
    setShowAttendanceMode(false);
    setPanelMode(null);
    setSelectedSubject(null);
  };

  // Function to activate face recognition and open camera
  const activateFaceRecognition = async ({ mode, subject } = {}) => {
    setIsStarting(true);
    // Proactively start backend to improve readiness before opening panel
    const tryPost = async (path) => {
      try {
        const res = await fetch(`${BACKEND_URL}${path}`, { method: 'POST' });
        if (!res.ok) throw new Error(`${path} returned ${res.status}`);
        return true;
      } catch (e) {
        return false;
      }
    };

    let started = await tryPost('/start_recognition');
    if (!started) started = await tryPost('/start');
    if (!started) started = await tryPost('/start_stream_only');

    // Poll camera_status briefly to see if a frame is ready
    const start = Date.now();
    while (Date.now() - start < 6000) {
      try {
        const res = await fetch(`${BACKEND_URL}/camera_status`);
        if (res.ok) {
          const json = await res.json();
          if (json && (json.has_frame || json.camera_active)) break;
        }
      } catch (e) {
        // ignore
      }
      await new Promise(r => setTimeout(r, 250));
    }

    // Decide where to go next based on selected mode
    if (mode) {
      setPanelMode(mode);
      setSelectedSubject(subject || null);
      setShowStatusPanel(true);
    } else {
      setShowAttendanceMode(true);
    }
    setIsStarting(false);
  };

  // Function to close face recognition and stop camera
  const closeFaceRecognition = async () => {
    setShowStatusPanel(false);
    setShowAttendanceMode(false);
    setPanelMode(null);
    setSelectedSubject(null);
    try {
      await fetch(`${BACKEND_URL}/stop`, { method: 'POST' });
    } catch (e) {
      console.error('Failed to stop recognition:', e);
    }
  };

  // No global events needed; dashboards call activateFaceRecognition directly
  useEffect(() => {}, []);

  if (!role) return <Login onLogin={handleLogin} />;

  const AttendanceMode = () => {
    const subjects = [
      { id: 'math101', name: 'Mathematics 101', weekday: 1 },
      { id: 'eng201', name: 'English 201', weekday: 2 },
      { id: 'cs301', name: 'Computer Science 301', weekday: 3 },
      { id: 'phy110', name: 'Physics 110', weekday: 4 },
      { id: 'hist210', name: 'History 210', weekday: 5 },
    ];
    const today = new Date().getDay();
    const isToday = (s) => s.weekday === today;

    if (panelMode === 'class') {
      return (
        <div className="attendance-container">
          <div className="attendance-card">
            <div className="attendance-header">
              <div className="attendance-title">Select Subject</div>
            </div>
            <div className="attendance-body">
              <p className="section-subtle">Choose a subject to start taking attendance.</p>
              <div className="subjects-grid mt-16">
                {subjects.map(s => (
                  <button
                    key={s.id}
                    onClick={() => { setSelectedSubject(s); setShowAttendanceMode(false); setShowStatusPanel(true); }}
                    className={`btn subject-card ${isToday(s) ? 'subject-today' : ''}`}
                  >
                    <div className="subject-name">{s.name} {isToday(s) ? '• Today' : ''}</div>
                    <div className="subject-meta">Weekday: {s.weekday}</div>
                  </button>
                ))}
              </div>
              <div className="toolbar">
                <button className="btn btn-secondary" onClick={() => setPanelMode(null)}>Back</button>
              </div>
            </div>
          </div>
        </div>
      );
    }

    return (
      <div className="attendance-container">
        <div className="attendance-card">
          <div className="attendance-header">
            <div className="attendance-title">Take Attendance</div>
          </div>
          <div className="attendance-body">
            <h3 className="section-heading">Choose Mode</h3>
            <p className="section-subtle">Select how you want to take attendance.</p>
            <div className="mode-actions mt-16">
              <button className="btn btn-primary" onClick={() => setPanelMode('class')}>Class Mode</button>
              <button className="btn btn-secondary" onClick={() => { setPanelMode('events'); setShowAttendanceMode(false); setShowStatusPanel(true); }}>Events Mode</button>
            </div>
            <div className="toolbar">
              <button className="btn btn-secondary" onClick={() => { setShowAttendanceMode(false); setPanelMode(null); }}>Cancel</button>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // Admin gets StatusPanel when active, otherwise shows dashboard
  if (role === 'admin') {
    if (showAttendanceMode && !showStatusPanel) {
      return <AttendanceMode />;
    }
    return showStatusPanel ? (
      <div style={{ position: 'relative', height: '100vh' }}>
        <button 
          onClick={closeFaceRecognition} 
          style={{ 
            position: "absolute", 
            zIndex: 1000, 
            right: 20, 
            top: 20, 
            padding: "10px 20px",
            backgroundColor: "#ef4444",
            color: "white",
            border: "none",
            borderRadius: "5px",
            cursor: "pointer",
            fontSize: "14px"
          }}
        >
          Close Panel
        </button>
        <StatusPanel mode={panelMode} subject={selectedSubject} />
      </div>
    ) : (
      <AdminDashboard onLogout={handleLogout} onTakeAttendance={() => activateFaceRecognition()} starting={isStarting} />
    );
  }

  if (role === 'teacher') {
    if (showAttendanceMode && !showStatusPanel) {
      return <AttendanceMode />;
    }
    return showStatusPanel ? (
      <div style={{ position: 'relative', height: '100vh' }}>
        <button 
          onClick={closeFaceRecognition} 
          style={{ 
            position: "absolute", 
            zIndex: 1000, 
            right: 20, 
            top: 20, 
            padding: "10px 20px",
            backgroundColor: "#ef4444",
            color: "white",
            border: "none",
            borderRadius: "5px",
            cursor: "pointer",
            fontSize: "14px"
          }}
        >
          Close Panel
        </button>
        <StatusPanel mode={panelMode} subject={selectedSubject} />
      </div>
    ) : (
      <TeacherDashboard onLogout={handleLogout} onTakeAttendance={() => activateFaceRecognition()} starting={isStarting} />
    );
  }

  if (showAttendanceMode && !showStatusPanel) {
    return <AttendanceMode />;
  }

  return showStatusPanel ? (
    <div style={{ position: 'relative', height: '100vh' }}>
      <button 
        onClick={closeFaceRecognition} 
        style={{ 
          position: "absolute", 
          zIndex: 1000, 
          right: 20, 
          top: 20, 
          padding: "10px 20px",
          backgroundColor: "#ef4444",
          color: "white",
          border: "none",
          borderRadius: "5px",
          cursor: "pointer",
          fontSize: "14px"
        }}
      >
        Close Panel
      </button>
      <StatusPanel mode={panelMode} subject={selectedSubject} />
    </div>
  ) : (
    <StudentDashboard onLogout={handleLogout} onFaceRecognition={() => activateFaceRecognition()} starting={isStarting} />
  );
}

export default App;