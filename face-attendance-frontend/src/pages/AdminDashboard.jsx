import React from "react";
import "./AdminDashboard.css";

export default function AdminDashboard({ onLogout, onTakeAttendance, starting }) {
  // Delegate to parent-provided handler which opens StatusPanel; StatusPanel starts camera itself
  const handleTakeAttendance = async () => {
    if (typeof onTakeAttendance === "function") {
      await onTakeAttendance();
    } else {
      console.warn('onTakeAttendance callback not provided');
    }
  };

  return (
    <div className="admin-dashboard">
      <h1>Admin Dashboard</h1>
      <p>Welcome, Admin. You can manage users and system settings here.</p>
      <div className="admin-btns">
        <button className="admin-btn primary" onClick={handleTakeAttendance} disabled={typeof onTakeAttendance !== 'function' || starting}>
          {typeof onTakeAttendance !== 'function' ? 'Take Attendance (unavailable)' : (starting ? 'Starting…' : 'Take Attendance')}
        </button>
        <button className="admin-btn secondary">Users</button>
        <button className="admin-btn secondary">Settings</button>
        <button className="admin-btn secondary">Reports</button>
        <button className="admin-btn secondary">Logs</button>
      </div>
      <button className="admin-logout" onClick={onLogout}>Logout</button>
    </div>
  );
}