import React, { useState, useEffect } from "react";
import StatusPanel from "./StatusPanel.js";
import Login from "./Login.jsx";
import AdminDashboard from "./AdminDashboard.jsx";
import TeacherDashboard from "./TeacherDashboard.jsx";
import StudentDashboard from "./StudentDashboard.jsx";
import "../styles/App.css";
import StudentRegis from "./studentregis.jsx";
import RegisterTeacher from "./teachregis.jsx";


function App() {
  const [role, setRole] = useState(null);
  const [showStatusPanel, setShowStatusPanel] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [showAttendanceMode, setShowAttendanceMode] = useState(false);
  const [panelMode, setPanelMode] = useState(null);
  const [selectedSubject, setSelectedSubject] = useState(null);
  const [currentPage, setCurrentPage] = useState("dashboard");


  const BACKEND_URL =
    process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:8000";

  // Handle login and logout
  const handleLogin = (userRole) => setRole(userRole || "admin");

  const handleLogout = () => {
    setRole(null);
    setShowStatusPanel(false);
    setShowAttendanceMode(false);
    setPanelMode(null);
    setSelectedSubject(null);
  };

  // Trigger face recognition
  const activateFaceRecognition = async ({ mode, subject } = {}) => {
    setIsStarting(true);

    const tryPost = async (path) => {
      try {
        const res = await fetch(`${BACKEND_URL}${path}`, { method: "POST" });
        if (!res.ok) throw new Error(`${path} returned ${res.status}`);
        return true;
      } catch {
        return false;
      }
    };

    // Try all backend start endpoints (no need to store 'started')
    await tryPost("/start_recognition") ||
      await tryPost("/start") ||
      await tryPost("/start_stream_only");

    // Wait for camera confirmation
    const startTime = Date.now();
    while (Date.now() - startTime < 6000) {
      try {
        const res = await fetch(`${BACKEND_URL}/camera_status`);
        if (res.ok) {
          const data = await res.json();
          if (data && (data.has_frame || data.camera_active)) break;
        }
      } catch {}
      await new Promise((r) => setTimeout(r, 250));
    }

    if (mode) {
      setPanelMode(mode);
      setSelectedSubject(subject || null);
      setShowStatusPanel(true);
    } else {
      setShowAttendanceMode(true);
    }

    setIsStarting(false);
  };

  // Close recognition session
  const closeFaceRecognition = async () => {
    setShowStatusPanel(false);
    setShowAttendanceMode(false);
    setPanelMode(null);
    setSelectedSubject(null);
    try {
      await fetch(`${BACKEND_URL}/stop`, { method: "POST" });
    } catch (err) {
      console.error("Failed to stop recognition:", err);
    }
  };

  useEffect(() => {
    // Optional initialization logic
  }, []);

  // If not logged in, show login page
  if (!role) return <Login onLogin={handleLogin} />;

  // Attendance mode screen
  const AttendanceMode = () => {
    const subjects = [
      { id: "math101", name: "Mathematics 101", weekday: 1 },
      { id: "eng201", name: "English 201", weekday: 2 },
      { id: "cs301", name: "Computer Science 301", weekday: 3 },
      { id: "phy110", name: "Physics 110", weekday: 4 },
      { id: "hist210", name: "History 210", weekday: 5 },
    ];

    const today = new Date().getDay();
    const isToday = (s) => s.weekday === today;

    if (panelMode === "class") {
      return (
        <div className="attendance-container">
          <div className="attendance-card">
            <div className="attendance-header">
              <h2 className="attendance-title">Select Subject</h2>
            </div>
            <div className="attendance-body">
              <p className="section-subtle">
                Choose a subject to start taking attendance.
              </p>
              <div className="subjects-grid mt-16">
                {subjects.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => {
                      setSelectedSubject(s);
                      setShowAttendanceMode(false);
                      setShowStatusPanel(true);
                    }}
                    className={`btn subject-card ${
                      isToday(s) ? "subject-today" : ""
                    }`}
                  >
                    <div className="subject-name">
                      {s.name} {isToday(s) ? "• Today" : ""}
                    </div>
                    <div className="subject-meta">
                      Weekday: {s.weekday}
                    </div>
                  </button>
                ))}
              </div>
              <div className="toolbar">
                <button
                  className="btn btn-secondary"
                  onClick={() => setPanelMode(null)}
                >
                  Back
                </button>
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
            <h2 className="attendance-title">Take Attendance</h2>
          </div>
          <div className="attendance-body">
            <h3 className="section-heading">Choose Mode</h3>
            <p className="section-subtle">
              Select how you want to take attendance.
            </p>
            <div className="mode-actions mt-16">
              <button
                className="btn btn-primary"
                onClick={() => setPanelMode("class")}
              >
                Class Mode
              </button>
              <button
                className="btn btn-secondary"
                onClick={() => {
                  setPanelMode("events");
                  setShowAttendanceMode(false);
                  setShowStatusPanel(true);
                }}
              >
                Events Mode
              </button>
            </div>
            <div className="toolbar">
              <button
                className="btn btn-secondary"
                onClick={() => {
                  setShowAttendanceMode(false);
                  setPanelMode(null);
                }}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // Status panel renderer
  const renderStatusPanel = () => (
    <div className="app-fullscreen">
      <button className="app-close-btn" onClick={closeFaceRecognition}>
        Close Panel
      </button>
      <StatusPanel mode={panelMode} subject={selectedSubject} />
    </div>
  );

  // Dashboard handling based on role
  if (role === "admin") {
  if (currentPage === "studentRegis") {
    return <StudentRegis onBack={() => setCurrentPage("dashboard")} />;
  }
  if (currentPage === "teachregis") {
    return <RegisterTeacher onBack={() => setCurrentPage("dashboard")} />;
  }

  if (showAttendanceMode && !showStatusPanel) return <AttendanceMode />;
  return showStatusPanel ? (
    renderStatusPanel()
  ) : (
    <AdminDashboard
      onLogout={handleLogout}
      onTakeAttendance={() => activateFaceRecognition()}
      starting={isStarting}
      onNavigate={(page) => setCurrentPage(page)}
    />
  );
}


  if (role === "teacher") {
    if (showAttendanceMode && !showStatusPanel) return <AttendanceMode />;
    return showStatusPanel ? (
      renderStatusPanel()
    ) : (
      <TeacherDashboard
        onLogout={handleLogout}
        onTakeAttendance={() => activateFaceRecognition()}
        starting={isStarting}
      />
    );
  }

  if (showAttendanceMode && !showStatusPanel) return <AttendanceMode />;

  return showStatusPanel ? (
    renderStatusPanel()
  ) : (
    <StudentDashboard
      onLogout={handleLogout}
      onFaceRecognition={() => activateFaceRecognition()}
      starting={isStarting}
    />
  );
}

export default App;
