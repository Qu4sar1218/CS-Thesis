import React from "react";
import "./StudentDashboard.css";

export default function StudentDashboard({ onLogout, onFaceRecognition }) {
  return (
    <div className="student-dashboard">
      <h1>Student Dashboard</h1>
      <p>Welcome, Student. View your attendance and schedule here.</p>
      <div className="student-btns">
        <button className="student-btn secondary">My Attendance</button>
        <button className="student-btn secondary">Schedule</button>
        <button className="student-btn secondary">Profile</button>
        <button className="student-btn secondary">Help</button>
      </div>
      <button className="student-logout" onClick={onLogout} style={{ marginTop: 12 }}>Logout</button>
    </div>
  );
}
