import React from "react";
import '../styles/StudentDashboard.css';


export default function StudentDashboard({ onLogout, onFaceRecognition }) {
  return (
    <main className="student-dashboard">
      <h1>Student Dashboard</h1>
      <p>Welcome, Student. View your attendance and schedule here.</p>
      <nav className="student-btns" aria-label="student actions">
        <button className="student-btn secondary">My Attendance</button>
        <button className="student-btn secondary">Schedule</button>
        <button className="student-btn secondary">Profile</button>
        <button className="student-btn secondary">Help</button>
      </nav>
      <button className="student-logout" onClick={onLogout} style={{ marginTop: 12 }}>Logout</button>
    </main>
  );
}
