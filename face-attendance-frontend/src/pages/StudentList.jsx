import React, { useState } from "react";
import "../styles/StudentList.css";

export default function StudentList({ onBack }) {
  const [students, setStudents] = useState([
    { id: "STU001", name: "Ana Rodriguez", course: "BSIT", year: "1st Year", section: "A" },
    { id: "STU002", name: "Miguel Lopez", course: "BSIT", year: "2nd Year", section: "B" },
    { id: "STU003", name: "Sofia Garcia", course: "BSCS", year: "3rd Year", section: "C" },
    { id: "STU004", name: "Carlos Martinez", course: "BSIT", year: "1st Year", section: "A" },
    { id: "STU005", name: "Isabella Fernandez", course: "BSCS", year: "2nd Year", section: "B" },
  ]);

  const [searchTerm, setSearchTerm] = useState("");
  const [selectedStudent, setSelectedStudent] = useState(null);
  const [isViewModalOpen, setIsViewModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [editForm, setEditForm] = useState({});

  const filteredStudents = students.filter(student =>
    student.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    student.course.toLowerCase().includes(searchTerm.toLowerCase()) ||
    student.id.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleView = (student) => {
    setSelectedStudent(student);
    setIsViewModalOpen(true);
  };

  const handleEdit = (student) => {
    setSelectedStudent(student);
    setEditForm({ ...student });
    setIsEditModalOpen(true);
  };

  const handleSaveEdit = () => {
    setStudents(students.map(s => s.id === selectedStudent.id ? editForm : s));
    setIsEditModalOpen(false);
    setSelectedStudent(null);
  };

  const closeModals = () => {
    setIsViewModalOpen(false);
    setIsEditModalOpen(false);
    setSelectedStudent(null);
  };

  return (
    <div className="student-list">
      <h1>Student List</h1>

      <div className="search-bar">
        <input
          type="text"
          placeholder="Search by name, course, or ID..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="search-input"
        />
      </div>

      <div className="list-container">
        <table className="student-table">
          <thead>
            <tr>
              <th>Student ID</th>
              <th>Name</th>
              <th>Course</th>
              <th>Year Level</th>
              <th>Section</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filteredStudents.map((student, index) => (
              <tr key={student.id} className={index % 2 === 0 ? 'even-row' : 'odd-row'}>
                <td>{student.id}</td>
                <td>{student.name}</td>
                <td>{student.course}</td>
                <td>{student.year}</td>
                <td>{student.section}</td>
                <td>
                  <button className="action-btn view" onClick={() => handleView(student)}>
                    👁️ View
                  </button>
                  <button className="action-btn edit" onClick={() => handleEdit(student)}>
                    ✏️ Edit
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="teacher-list-form-buttons">
        <button type="button" className="teacher-list-secondary" onClick={onBack} style={{ padding: '4px 8px', fontSize: '12px' }}>
          Back
        </button>
      </div>

      {/* View Modal */}
      {isViewModalOpen && selectedStudent && (
        <div className="modal-overlay" onClick={closeModals}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h2>Student Details</h2>
            <div className="detail-row">
              <strong>ID:</strong> {selectedStudent.id}
            </div>
            <div className="detail-row">
              <strong>Name:</strong> {selectedStudent.name}
            </div>
            <div className="detail-row">
              <strong>Course:</strong> {selectedStudent.course}
            </div>
            <div className="detail-row">
              <strong>Year:</strong> {selectedStudent.year}
            </div>
            <div className="detail-row">
              <strong>Section:</strong> {selectedStudent.section}
            </div>
            <div className="modal-actions">
              <button className="btn-secondary" onClick={closeModals}>Close</button>
            </div>
          </div>
        </div>
      )}

      {/* Edit Modal */}
      {isEditModalOpen && (
        <div className="modal-overlay" onClick={closeModals}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h2>Edit Student</h2>
            <form className="edit-form">
              <div className="form-group">
                <label>Name:</label>
                <input
                  type="text"
                  value={editForm.name}
                  onChange={(e) => setEditForm({ ...editForm, name: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label>Course:</label>
                <input
                  type="text"
                  value={editForm.course}
                  onChange={(e) => setEditForm({ ...editForm, course: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label>Year:</label>
                <input
                  type="text"
                  value={editForm.year}
                  onChange={(e) => setEditForm({ ...editForm, year: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label>Section:</label>
                <input
                  type="text"
                  value={editForm.section}
                  onChange={(e) => setEditForm({ ...editForm, section: e.target.value })}
                />
              </div>
            </form>
            <div className="modal-actions">
              <button className="btn-primary" onClick={handleSaveEdit}>Save</button>
              <button className="btn-secondary" onClick={closeModals}>Cancel</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
