import React from "react";

export default function Analytics({ onBack }) {
  return (
    <div className="analytics" style={{ background: 'var(--dashboard-bg)', minHeight: '100vh' }}>
      <h1>Analytics</h1>
      <p>Analytics page content goes here.</p>
      <div className="analytics-form-buttons">
        <button type="button" className="analytics-secondary" onClick={onBack}>
          Back
        </button>
      </div>
    </div>
  );
}
