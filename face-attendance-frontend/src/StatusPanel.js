// Import React hooks and axios for HTTP requests
import React, { useEffect, useState } from "react";
import axios from "axios";

// StatusPanel component displays live video, detected student info, and attendance
function StatusPanel() {
  // State to hold status info from backend
  const [status, setStatus] = useState({});
  // State to hold attendance info from backend
  const [attendance, setAttendance] = useState({});
  // Add connection state
  const [isConnected, setIsConnected] = useState(false);

  // Backend base URL
  const BACKEND_URL = "http://127.0.0.1:8000";

  // useEffect sets up polling to fetch status and attendance every second
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        // Fetch current status (detected student, liveness, etc.)
        const statusRes = await axios.get(`${BACKEND_URL}/status`);
        setStatus(statusRes.data);

        // Fetch attendance list
        const attendanceRes = await axios.get(`${BACKEND_URL}/attendance`);
        setAttendance(attendanceRes.data);

        // Mark as connected
        setIsConnected(true);
      } catch (err) {
        // Mark as disconnected on error
        setIsConnected(false);
      }
    }, 1000); // Poll every 1 second

    // Cleanup interval on component unmount
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ display: "flex", height: "100vh", fontFamily: "Inter, sans-serif" }}>

      {/* Left Panel: Live Video Feed from backend camera */}
      <div style={{
        flex: 1,
        backgroundColor: "#1f2937",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "20px",
        position: "relative"
      }}>
        {/* Show connection status */}
        <div style={{
          position: "absolute",
          top: "10px",
          left: "10px",
          padding: "5px 10px",
          backgroundColor: isConnected ? "#10b981" : "#ef4444",
          color: "white",
          borderRadius: "5px",
          fontSize: "12px",
          zIndex: 1000
        }}>
          {isConnected ? "Connected" : "Disconnected"}
        </div>

        {/* Video stream from backend */}
        <img
          src={`${BACKEND_URL}/video`}
          alt="Live Camera"
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
            borderRadius: "20px",
            boxShadow: "0 8px 24px rgba(0,0,0,0.3)",
            border: "3px solid #4f46e5"
          }}
        />

        {/* Error overlay for disconnected state */}
        {!isConnected && (
          <div style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            backgroundColor: "rgba(239, 68, 68, 0.9)",
            color: "white",
            padding: "20px",
            borderRadius: "10px",
            textAlign: "center",
            maxWidth: "80%"
          }}>
            <h3>Connection Lost</h3>
            <p>Attempting to reconnect...</p>
            <p style={{ fontSize: "12px", marginTop: "10px" }}>
              Make sure the backend server is running on port 8000
            </p>
          </div>
        )}
      </div>

      {/* Right Panel: Info and Attendance */}
      <div style={{
        flex: 1,
        padding: "40px",
        backgroundColor: "#f3f4f6",
        overflowY: "auto"
      }}>
        {/* Title of the system */}
        <h1 style={{ color: "#111827", marginBottom: "25px" }}>Face Attendance System</h1>

        {/* Current Student Info Panel */}
        <div style={{
          marginBottom: "30px",
          padding: "25px",
          borderRadius: "20px",
          background: "linear-gradient(135deg, #e0f2fe, #bae6fd)",
          boxShadow: "0 6px 20px rgba(0,0,0,0.1)"
        }}>
          <h2 style={{ marginBottom: "15px", color: "#1e3a8a" }}>Current Detected Student</h2>
          <p><strong>Name:</strong> {status.detected_name || "Unknown"}</p>
          {status.student_details && Object.keys(status.student_details).length > 0 && (
            <>
              <p><strong>Course:</strong> {status.student_details.course}</p>
              <p><strong>Year:</strong> {status.student_details.year}</p>
              <p><strong>Section:</strong> {status.student_details.section}</p>
              <p><strong>Schedule:</strong></p>
              <ul style={{ maxHeight: "100px", overflowY: "auto", paddingLeft: "20px" }}>
                {status.student_details.schedule?.map((item, index) => (
                  <li key={index}>{item}</li>
                ))}
              </ul>
            </>
          )}
          <p>
            <strong>Liveness:</strong>
            <span style={{ color: status.liveness ? "green" : "red", marginLeft: "5px" }}>
              {status.liveness ? "Confirmed ✅" : "Not Confirmed ❌"}
            </span>
          </p>
          <p><strong>Last Scanned:</strong> {status.last_scanned || "None"}</p>
          
          {/* Brightness info */}
          {status.brightness_level !== undefined && (
            <p><strong>Brightness Adjustment:</strong> {status.brightness_level}</p>
          )}
          {status.auto_brightness_active !== undefined && (
            <p><strong>Auto Brightness:</strong> {status.auto_brightness_active ? "Active" : "Inactive"}</p>
          )}
        </div>

        {/* Attendance Panel */}
        <div style={{
          padding: "25px",
          borderRadius: "20px",
          background: "linear-gradient(135deg, #fef3c7, #fde68a)",
          boxShadow: "0 6px 20px rgba(0,0,0,0.1)"
        }}>
          <h2 style={{ marginBottom: "15px", color: "#78350f" }}>Attendance</h2>
          <p><strong>Total Present:</strong> {attendance.total_present || 0}</p>
          <ul style={{ maxHeight: "200px", overflowY: "auto", paddingLeft: "20px" }}>
            {attendance.attendance &&
              Object.entries(attendance.attendance).map(([name, time]) => (
                <li key={name}>{name} - {time}</li>
              ))
            }
          </ul>
        </div>
      </div>
    </div>
  );
}

// Export the StatusPanel component as default
export default StatusPanel;