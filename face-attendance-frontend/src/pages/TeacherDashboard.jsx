import React from "react";
import '../styles/TeacherDashboard.css';


export default function TeacherDashboard({ onLogout, onTakeAttendance, starting }) {
  const handleTakeAttendance = async () => {
    if (typeof onTakeAttendance === "function") {
      await onTakeAttendance();
    } else {
      console.warn('onTakeAttendance callback not provided');
    }
  };

  return (
       <main className="teacher-dashboard">
      <h1>Teacher Dashboard</h1>
      <p>Welcome, Teacher. Take attendance, review students, and manage classes.</p>
         <nav className="teacher-btns" aria-label="teacher actions">
        <button className="teacher-btn" onClick={handleTakeAttendance} disabled={typeof onTakeAttendance !== 'function' || starting}>
          {typeof onTakeAttendance !== 'function' ? 'Take Attendance (unavailable)' : (starting ? 'Starting…' : 'Take Attendance')}
        </button>
        <button className="teacher-btn secondary">Class Roster</button>
        <button className="teacher-btn secondary">Attendance</button>
        <button className="teacher-btn secondary">Assignments</button>
        <button className="teacher-btn secondary">Messages</button>
         </nav>
      <button className="teacher-logout" onClick={onLogout} style={{ marginTop: 12 }}>Logout</button>
       </main>
  );
}
