import React from "react";
import "../styles/TeacherList.css";

export default function TeacherList({ onBack }) {
  return (
    <div className="teacher-list">
      <h1>Teacher List</h1>

      <div className="list-container">
        <table className="teacher-table">
          <thead>
            <tr>
              <th>Teacher ID</th>
              <th>Name</th>
              <th>Subject</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>T001</td>
              <td>Dr. Maria Santos</td>
              <td>Mathematics</td>
              <td>
                <button className="action-btn view">View</button>
                <button className="action-btn edit">Edit</button>
              </td>
            </tr>
            <tr>
              <td>T002</td>
              <td>Prof. Juan Dela Cruz</td>
              <td>Computer Science</td>
              <td>
                <button className="action-btn view">View</button>
                <button className="action-btn edit">Edit</button>
              </td>
            </tr>
            <tr>
              <td>T003</td>
              <td>Ms. Elena Reyes</td>
              <td>English Literature</td>
              <td>
                <button className="action-btn view">View</button>
                <button className="action-btn edit">Edit</button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="teacher-list-form-buttons">
        <button type="button" className="teacher-list-secondary" onClick={onBack} style={{ padding: '4px 8px', fontSize: '12px' }}>
          Back
        </button>
      </div>
    </div>
  );
}
