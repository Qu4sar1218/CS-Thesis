import React from "react";
import "../styles/Reports.css";

export default function Reports({ onBack }) {
  return (
    <div className="reports" style={{ background: 'var(--dashboard-bg)', minHeight: '100vh', textAlign: 'center' }}>
      <h1>Reports</h1>
      <p>Reports page content goes here.</p>
      <div className="reports-form-buttons">
        <button type="button" className="reports-secondary" onClick={onBack} style={{ padding: '4px 8px', fontSize: '12px' }}>
          Back
        </button>
      </div>
    </div>
  );
}
