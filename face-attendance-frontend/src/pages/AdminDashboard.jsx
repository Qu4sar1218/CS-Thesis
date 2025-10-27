import React from "react";
import '../styles/AdminDashboard.css';

export default function AdminDashboard({ onLogout, onTakeAttendance, starting, onNavigate }) {
  // Delegate to parent-provided handler which opens StatusPanel; StatusPanel starts camera itself
  const handleTakeAttendance = async () => {
    if (typeof onTakeAttendance === "function") {
      await onTakeAttendance();
    } else {
      console.warn('onTakeAttendance callback not provided');
    }
  };

  return (
    <main className="admin-dashboard">
      <h1>Admin Dashboard</h1>
      <p>Welcome, Admin. You can manage users and system settings here.</p>

      <nav className="admin-btns" aria-label="admin actions">
        <button
          className="admin-btn primary"
          onClick={handleTakeAttendance}
          disabled={typeof onTakeAttendance !== 'function' || starting}
        >
          {typeof onTakeAttendance !== 'function'
            ? 'Take Attendance (unavailable)'
            : (starting ? 'Starting…' : 'Take Attendance')}
        </button>

        {/* opens registration form with face recognition training */}
        <button
          className="admin-btn primary"
          onClick={() => onNavigate && onNavigate("studentRegis")}
        >
          Register Student
        </button>

        {/* opens registration form */}
        <button 
        className="admin-btn secondary"  
        onClick={() => onNavigate && onNavigate("teachregis")}
        >
          Register Teacher
        </button>

        {/* with edit/delete of students */}
        <button className="admin-btn secondary">Student List</button>

        <button className="admin-btn secondary">Manage Classes</button>
        <button className="admin-btn secondary">Reports</button>
        <button className="admin-btn secondary">Analytics</button>
        <button className="admin-btn secondary">Logs</button>
      </nav>

      <button className="admin-logout" onClick={onLogout}>Logout</button>
    </main>
  );
}
