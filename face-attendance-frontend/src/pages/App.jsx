import React, { useState, useEffect } from 'react';

import Login from "./Login.jsx";
import AdminDashboard from "./AdminDashboard.jsx";
import AdminReceiptVerification from "./AdminReceiptVerification.jsx";
import StudentReceiptSubmission from "./StudentReceiptSubmission.jsx";
import TeacherDashboard from "./TeacherDashboard.jsx";
import TeacherClassRoster from "./TeacherClassRoster.jsx";
import StudentDashboard from "./StudentDashboard.jsx";
import StatusPanel from "./StatusPanel.js";
import EventManagement from "./EventManagement.jsx";
import "../styles/App.css";
import StudentRegis from "./studentregis.jsx";
import RegisterTeacher from "./teachregis.jsx";
import StudentList from "./StudentList.jsx";
import TeacherList from "./TeacherList.jsx";
import ManageClasses from "./ManageClasses.jsx";
import Reports from "./Reports.jsx";
import Analytics from "./Analytics.jsx";
import Settings from "./Settings.jsx";
import Notifications from "./Notifications.jsx";

function App() {
  const [role, setRole] = useState(null);
  const [userInfo, setUserInfo] = useState(null);
  const [showStatusPanel, setShowStatusPanel] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [showAttendanceMode, setShowAttendanceMode] = useState(false);
  const [panelMode, setPanelMode] = useState(null);
  const [selectedSubject, setSelectedSubject] = useState(null);
  const [currentPage, setCurrentPage] = useState("dashboard");

  const BACKEND_URL =
    process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:8000";

  // Handle login and logout
  const handleLogin = (userRole, userData) => {
    setRole(userRole || "admin");
    setUserInfo(userData || null);
  };

  const handleLogout = () => {
    setRole(null);
    setUserInfo(null);
    setShowStatusPanel(false);
    setShowAttendanceMode(false);
    setPanelMode(null);
    setSelectedSubject(null);
  };

  // Trigger face recognition - FIXED VERSION
  const activateFaceRecognition = async ({ mode, subject } = {}) => {
    console.log("🚀 Starting face recognition...");
    setIsStarting(true);

    try {
      // ✅ FIXED: Only call the correct /start endpoint
      console.log(`Calling ${BACKEND_URL}/start`);
      const res = await fetch(`${BACKEND_URL}/start`, { method: "POST" });
      
      if (!res.ok) {
        console.error(`Start failed with status ${res.status}`);
        alert(`Failed to start recognition. Status: ${res.status}`);
        setIsStarting(false);
        return;
      }

      const data = await res.json();
      console.log("Start response:", data);

      // Wait for camera to be ready
      console.log("⏳ Waiting for camera to initialize...");
      const startTime = Date.now();
      let cameraReady = false;

      while (Date.now() - startTime < 8000) {
        try {
          const statusRes = await fetch(`${BACKEND_URL}/camera_status`);
          if (statusRes.ok) {
            const statusData = await statusRes.json();
            console.log("Camera status:", statusData);
            
            if (statusData && (statusData.has_frame || statusData.camera_active)) {
              console.log("✅ Camera is ready!");
              cameraReady = true;
              break;
            }
          }
        } catch (err) {
          console.warn("Camera status check failed:", err);
        }
        await new Promise((r) => setTimeout(r, 300));
      }

      if (!cameraReady) {
        console.warn("⚠️ Camera took longer than expected to initialize");
      }

      // Show the status panel
      if (mode) {
        setPanelMode(mode);
        setSelectedSubject(subject || null);
        setShowStatusPanel(true);
      } else {
        setShowAttendanceMode(true);
      }

    } catch (error) {
      console.error("❌ Failed to activate face recognition:", error);
      alert(
        `Failed to connect to backend at ${BACKEND_URL}\n\n` +
        `Error: ${error.message}\n\n` +
        `Make sure the backend server is running:\n` +
        `python main.py`
      );
    } finally {
      setIsStarting(false);
    }
  };

  // Close recognition session
  const closeFaceRecognition = async () => {
    console.log("🛑 Stopping face recognition...");
    setShowStatusPanel(false);
    setShowAttendanceMode(false);
    setPanelMode(null);
    setSelectedSubject(null);
    
    try {
      await fetch(`${BACKEND_URL}/stop`, { method: "POST" });
      console.log("✅ Recognition stopped");
    } catch (err) {
      console.error("Failed to stop recognition:", err);
    }
  };

  useEffect(() => {
    // Test backend connectivity on mount
    const testBackend = async () => {
      try {
        console.log(`Testing backend at ${BACKEND_URL}`);
        const res = await fetch(`${BACKEND_URL}/health`);
        if (res.ok) {
          const data = await res.json();
          console.log("✅ Backend connected:", data);
        } else {
          console.warn("⚠️ Backend responded with status:", res.status);
        }
      } catch (err) {
        console.error("❌ Backend not reachable:", err.message);
        console.error("Make sure to run: python main.py");
      }
    };
    testBackend();
  }, [BACKEND_URL]);

  // If not logged in, show login page
  if (!role) return <Login onLogin={handleLogin} />;

  // Attendance mode screen
  const AttendanceMode = () => {
    const [teacherSubjects, setTeacherSubjects] = useState([]);
    const [loadingSubjects, setLoadingSubjects] = useState(false);

    const subjects = role === "teacher" ? teacherSubjects : [
      { id: "math101", name: "Mathematics 101", accessible: true },
      { id: "eng201", name: "English 201", accessible: true },
      { id: "cs301", name: "Computer Science 301", accessible: true },
      { id: "phy110", name: "Physics 110", accessible: true },
      { id: "hist210", name: "History 210", accessible: true },
    ];

    useEffect(() => {
      if (role === "teacher" && panelMode === "class" && userInfo?.user_id) {
        const fetchTeacherSubjects = async () => {
          setLoadingSubjects(true);
          try {
            const response = await fetch(`${BACKEND_URL}/classes/teacher/${userInfo.user_id}`);
            if (response.ok) {
              const data = await response.json();
              const subjectsFromClasses = data.classes.map(cls => ({
                id: cls.class_code,
                name: cls.class_name,
                accessible: cls.accessible
              }));
              setTeacherSubjects(subjectsFromClasses);
            }
          } catch (error) {
            console.error("Error fetching teacher subjects:", error);
          } finally {
            setLoadingSubjects(false);
          }
        };
        fetchTeacherSubjects();
      }
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

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
              {role === "teacher" && loadingSubjects ? (
                <div className="loading-subjects">Loading your subjects...</div>
              ) : (
                <div className="subjects-grid mt-16">
                  {subjects.map((s) => (
                    <button
                      key={s.id}
                      onClick={() => {
                        if (s.accessible) {
                          setSelectedSubject(s);
                          setShowAttendanceMode(false);
                          setShowStatusPanel(true);
                        }
                      }}
                      disabled={!s.accessible}
                      className={`btn subject-card ${
                        s.accessible ? "subject-today" : "subject-disabled"
                      }`}
                    >
                      <div className="subject-name">
                        {s.name} {s.accessible ? "• Scheduled today" : " - Not scheduled for today"}
                      </div>
                      <div className="subject-meta">{s.accessible ? "Scheduled today" : "Not scheduled for today"}</div>
                    </button>
                  ))}
                </div>
              )}
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
    if (currentPage === "studentList") {
      return <StudentList onBack={() => setCurrentPage("dashboard")} />;
    }
    if (currentPage === "teacherList") {
      return <TeacherList onBack={() => setCurrentPage("dashboard")} />;
    }
    if (currentPage === "manageClasses") {
      return <ManageClasses onBack={() => setCurrentPage("dashboard")} />;
    }
    if (currentPage === "reports") {
      return <Reports onBack={() => setCurrentPage("dashboard")} />;
    }
    if (currentPage === "analytics") {
      return <Analytics onBack={() => setCurrentPage("dashboard")} />;
    }
    if (currentPage === "settings") {
      return <Settings onBack={() => setCurrentPage("dashboard")} />;
    }
    if (currentPage === "notifications") {
      return <Notifications onBack={() => setCurrentPage("dashboard")} />;
    }
    if (currentPage === "receiptVerification") {
      return <AdminReceiptVerification onBack={() => setCurrentPage("dashboard")} />;
    }
    if (currentPage === "eventManagement") {
      return <EventManagement onBack={() => setCurrentPage("dashboard")} />;
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
    if (currentPage === "teacherClassRoster") {
      return <TeacherClassRoster onBack={() => setCurrentPage("dashboard")} userInfo={userInfo} />;
    }

    if (showAttendanceMode && !showStatusPanel) return <AttendanceMode />;
    return showStatusPanel ? (
      renderStatusPanel()
    ) : (
      <TeacherDashboard
        onLogout={handleLogout}
        onTakeAttendance={() => activateFaceRecognition()}
        starting={isStarting}
        userInfo={userInfo}
        onNavigate={(page) => setCurrentPage(page)}
      />
    );
  }

  if (currentPage === "receiptSubmission") {
    return <StudentReceiptSubmission studentId={userInfo?.user_id || "114001"} onBack={() => setCurrentPage("dashboard")} />;
  }

  if (showAttendanceMode && !showStatusPanel) return <AttendanceMode />;

  return showStatusPanel ? (
    renderStatusPanel()
  ) : (
    <StudentDashboard
      onLogout={handleLogout}
      onFaceRecognition={() => activateFaceRecognition()}
      onNavigate={(page) => setCurrentPage(page)}
      starting={isStarting}
      userInfo={userInfo}
    />
  );
}

export default App;