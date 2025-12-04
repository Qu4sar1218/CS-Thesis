import React from "react";

export default function Settings({ onBack }) {
  return (
    <div className="settings" style={{ background: 'var(--dashboard-bg)', minHeight: '100vh', textAlign: 'center' }}>
      <h1>Settings</h1>
      <p>Settings page content goes here.</p>
      <div className="settings-form-buttons">
        <button type="button" className="settings-secondary" onClick={onBack}>
          Back
        </button>
      </div>
    </div>
  );
}
