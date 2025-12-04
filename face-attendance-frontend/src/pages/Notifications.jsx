import React from "react";

export default function Notifications({ onBack }) {
  return (
    <div className="notifications" style={{ background: 'var(--dashboard-bg)', minHeight: '100vh', textAlign: 'center' }}>
      <h1>Notifications</h1>
      <p>Notifications page content goes here.</p>
      <div className="notifications-form-buttons">
        <button type="button" className="notifications-secondary" onClick={onBack}>
          Back
        </button>
      </div>
    </div>
  );
}
