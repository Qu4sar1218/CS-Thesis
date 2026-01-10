import React, { useState, useEffect } from "react";
import "../styles/StudentList.css";

const API_BASE_URL = "http://localhost:8000";

export default function StudentList({ onBack }) {
  const [students, setStudents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [searchTerm, setSearchTerm] = useState("");
  const [selectedStudent, setSelectedStudent] = useState(null);
  const [isViewModalOpen, setIsViewModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [editForm, setEditForm] = useState({});
  const [selectedCourse, setSelectedCourse] = useState(null);

  // Fetch students from API
  useEffect(() => {
    fetchStudents();
  }, []);

  const fetchStudents = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/students`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      if (!data.students || !Array.isArray(data.students)) {
        throw new Error('Invalid response format: expected {students: [...]}');
      }
      // Transform API data to match component expectations
      const transformedStudents = data.students.map(student => ({
        id: student.student_id,
        name: `${student.first_name} ${student.last_name}`,
        course: student.course,
        year: student.year,
        section: 'A', // Default section since not in API
        email: student.email,
        contact: 'N/A', // Not in API
        _id: student._id
      }));
      setStudents(transformedStudents);
      setError(null);
    } catch (err) {
      if (err.name === 'TypeError' && err.message === 'Failed to fetch') {
        setError('Network error: Unable to connect to the server. Please ensure the backend is running on http://localhost:8000');
      } else {
        setError(`Error fetching students: ${err.message}`);
      }
      console.error('Error fetching students:', err);
    } finally {
      setLoading(false);
    }
  };

  const filteredStudents = students.filter(student => {
    const matchesSearch = student.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      student.course.toLowerCase().includes(searchTerm.toLowerCase()) ||
      student.id.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCourse = selectedCourse === null || student.course === selectedCourse;
    return matchesSearch && matchesCourse;
  });

  const handleView = (student) => {
    setSelectedStudent(student);
    setIsViewModalOpen(true);
  };

  const handleEdit = (student) => {
    setSelectedStudent(student);
    setEditForm({ ...student });
    setIsEditModalOpen(true);
  };

  const handleSaveEdit = async () => {
    try {
      const updateData = {
        student_id: editForm.id,
        first_name: editForm.name.split(' ')[0] || editForm.name,
        last_name: editForm.name.split(' ').slice(1).join(' ') || '',
        course: editForm.course,
        year: editForm.year,
        email: editForm.email,
        // Optional fields
        ...(editForm.contact && { guardian_contact: editForm.contact }),
        ...(editForm.section && { section: editForm.section })
      };

      const response = await fetch(`${API_BASE_URL}/students/${selectedStudent._id}`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(updateData),
      });

      const result = await response.json();

      if (response.ok) {
        setIsEditModalOpen(false);
        setSelectedStudent(null);
        fetchStudents(); // Refresh the list
      } else {
        setError(`Failed to update student: ${result.detail || result.error}`);
      }
    } catch (error) {
      setError(`Error updating student: ${error.message}`);
    }
  };

  const handleDelete = async (student) => {
    const ok = window.confirm(`Delete student ${student.name} (STU ${student.id})?`);
    if (!ok) return;

    try {
      const response = await fetch(`${API_BASE_URL}/students/${student.id}`, {
        method: "DELETE",
      });

      if (response.ok) {
        fetchStudents(); // Refresh the list
      } else {
        const result = await response.json();
        setError(`Failed to delete student: ${result.detail || result.error}`);
      }
    } catch (error) {
      setError(`Error deleting student: ${error.message}`);
    }
  };

  const closeModals = () => {
    setIsViewModalOpen(false);
    setIsEditModalOpen(false);
    setSelectedStudent(null);
  };

  if (loading) {
    return (
      <div className="student-list">
        <h1>Student List</h1>
        <div className="loading">Loading students...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="student-list">
        <h1>Student List</h1>
        <div className="error">Error: {error}</div>
        <button onClick={fetchStudents} className="retry-btn">Retry</button>
      </div>
    );
  }

  return (
    <div className="student-list">
      <h1>Student List</h1>

      <div className="filter-buttons">
        <button
          className={`filter-btn ${selectedCourse === null ? 'active' : ''}`}
          onClick={() => setSelectedCourse(null)}
        >
          All
        </button>
        <button
          className={`filter-btn ${selectedCourse === 'BSIT' ? 'active' : ''}`}
          onClick={() => setSelectedCourse('BSIT')}
        >
          BSIT
        </button>
        <button
          className={`filter-btn ${selectedCourse === 'BSCS' ? 'active' : ''}`}
          onClick={() => setSelectedCourse('BSCS')}
        >
          BSCS
        </button>
        <button
          className={`filter-btn ${selectedCourse === 'BSBA' ? 'active' : ''}`}
          onClick={() => setSelectedCourse('BSBA')}
        >
          BSBA
        </button>
        <button
          className={`filter-btn ${selectedCourse === 'STEM' ? 'active' : ''}`}
          onClick={() => setSelectedCourse('STEM')}
        >
          STEM
        </button>
        <button
          className={`filter-btn ${selectedCourse === 'GAS' ? 'active' : ''}`}
          onClick={() => setSelectedCourse('GAS')}
        >
          GAS
        </button>
        <button
          className={`filter-btn ${selectedCourse === 'ICT' ? 'active' : ''}`}
          onClick={() => setSelectedCourse('ICT')}
        >
          ICT
        </button>
      </div>

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
              <th>Email</th>
              <th>Contact</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filteredStudents.map((student, index) => (
              <tr key={student.id} className={index % 2 === 0 ? 'even-row' : 'odd-row'}>
                <td>STU {student.id}</td>
                <td>{student.name}</td>
                <td>{student.course}</td>
                <td>{student.year}</td>
                <td>{student.section}</td>
                <td>{student.email}</td>
                <td>{student.contact}</td>
                <td>
                  <button className="action-btn view" onClick={() => handleView(student)}>
                    👁️ View
                  </button>
                  <button className="action-btn edit" onClick={() => handleEdit(student)}>
                    ✏️ Edit
                  </button>
                  <button className="action-btn delete" onClick={() => handleDelete(student)}>
                    🗑️ Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="teacher-list-form-buttons">
        <button type="button" className="teacher-list-secondary" onClick={onBack} style={{ padding: '8px 16px', fontSize: '13px', marginTop: '10px' }}>
          Back
        </button>
      </div>

      {/* View Modal */}
      {isViewModalOpen && selectedStudent && (
        <div className="modal-overlay" onClick={closeModals}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h2>Student Details</h2>
            <div className="detail-row">
              <strong>ID:</strong> STU {selectedStudent.id}
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
              <div className="form-group">
              <label>Email:</label>
              <input
              type="email"
              value={editForm.email}
              onChange={(e) => setEditForm({ ...editForm, email: e.target.value })}
              />
              </div>
              <div className="form-group">
              <label>Contact:</label>
              <input
              type="text"
              value={editForm.contact}
              onChange={(e) => setEditForm({ ...editForm, contact: e.target.value })}
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
