import React, { useState, useEffect } from "react";
import "../styles/TeacherList.css";

const API_BASE_URL = "http://localhost:8000";

export default function TeacherList({ onBack }) {
  const [teachers, setTeachers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [openDropdown, setOpenDropdown] = useState(null);

  const [selectedTeacher, setSelectedTeacher] = useState(null);
  const [isViewModalOpen, setIsViewModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false) ;
  const [editForm, setEditForm] = useState({});

  const getDisplayDepartment = (dept) => {
    if (dept === 'Both') return 'College & SHS';
    return dept;
  };

  // Fetch teachers and classes from API
  useEffect(() => {
    fetchTeachersAndClasses();
  }, []);

  const fetchTeachersAndClasses = async () => {
    try {
      setLoading(true);

      // Fetch teachers
      const teachersResponse = await fetch(`${API_BASE_URL}/teachers`);
      if (!teachersResponse.ok) {
        throw new Error('Failed to fetch teachers');
      }
      const teachersData = await teachersResponse.json();

      // Fetch classes
      const classesResponse = await fetch(`${API_BASE_URL}/classes`);
      if (!classesResponse.ok) {
        throw new Error('Failed to fetch classes');
      }
      const classesData = await classesResponse.json();

      // Transform API data to match component expectations
      const transformedTeachers = teachersData.teachers.map(teacher => {
        // Find classes taught by this teacher
        const teacherClasses = classesData.classes.filter(cls => cls.teacher_id === teacher.teacher_id);
        const subjects = teacherClasses.map(cls => cls.class_name);

        return {
          id: teacher.teacher_id,
          name: `${teacher.first_name} ${teacher.middle_name ? teacher.middle_name + ' ' : ''}${teacher.last_name}`,
          subjects: subjects,
          department: teacher.department,
          email: teacher.email,
          _id: teacher._id
        };
      });
      setTeachers(transformedTeachers);
      setError(null);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching teachers and classes:', err);
    } finally {
      setLoading(false);
    }
  };

  const filteredTeachers = teachers.filter(t =>
    t.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    (t.subjects && t.subjects.some(subject => subject.toLowerCase().includes(searchTerm.toLowerCase()))) ||
    t.department.toLowerCase().includes(searchTerm.toLowerCase()) ||
    t.id.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const toggleDropdown = (teacherId) => {
    setOpenDropdown(openDropdown === teacherId ? null : teacherId);
  };

  const handleView = async (teacher) => {
    try {
      const response = await fetch(`${API_BASE_URL}/teachers/${teacher.id}`);
      if (!response.ok) {
        throw new Error('Failed to fetch teacher details');
      }
      const data = await response.json();
      setSelectedTeacher(data);
      setIsViewModalOpen(true);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleEdit = async (teacher) => {
    try {
      const response = await fetch(`${API_BASE_URL}/teachers/${teacher.id}`);
      if (!response.ok) {
        throw new Error('Failed to fetch teacher details');
      }
      const data = await response.json();
      setSelectedTeacher(data);
      setEditForm({
        teacher_id: data.teacher_id,
        first_name: data.first_name,
        middle_name: data.middle_name,
        last_name: data.last_name,
        department: data.department,
        email: data.email
      });
      setIsEditModalOpen(true);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleSaveEdit = async () => {
    try {
      const updateData = {
        first_name: editForm.first_name,
        middle_name: editForm.middle_name || null,
        last_name: editForm.last_name,
        department: editForm.department,
        email: editForm.email
      };

      const response = await fetch(`${API_BASE_URL}/teachers/${selectedTeacher.teacher_id}`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(updateData),
      });

      const result = await response.json();

      if (response.ok) {
        setIsEditModalOpen(false);
        setSelectedTeacher(null);
        fetchTeachersAndClasses(); // Refresh the list
      } else {
        setError(`Failed to update teacher: ${result.detail || result.error}`);
      }
    } catch (error) {
      setError(`Error updating teacher: ${error.message}`);
    }
  };

  const handleDelete = async (teacher) => {
    const ok = window.confirm(`Delete teacher ${teacher.name} (${teacher.id})?`);
    if (!ok) return;

    try {
      const response = await fetch(`${API_BASE_URL}/teachers/${teacher.id}`, {
        method: "DELETE",
      });

      if (response.ok) {
        fetchTeachersAndClasses(); // Refresh the list
      } else {
        const result = await response.json();
        setError(`Failed to delete teacher: ${result.detail || result.error}`);
      }
    } catch (error) {
      setError(`Error deleting teacher: ${error.message}`);
    }
  };

  const closeModals = () => {
    setIsViewModalOpen(false);
    setIsEditModalOpen(false);
    setSelectedTeacher(null);
  };

  return (
    <div className="teacher-list">
      <h1>Teacher List</h1>

      <div className="search-bar">
        <input
          type="text"
          placeholder="Search by name, subject, or ID..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="search-input"
        />
      </div>

      {loading && <p>Loading teachers...</p>}
      {error && <p>Error: {error}</p>}
      {!loading && !error && (
        <div className="list-container">
          <table className="teacher-table">
            <thead>
              <tr>
                <th>Teacher ID</th>
                <th>Name</th>
                <th>Subjects</th>
                <th>Department</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredTeachers.map((teacher) => (
                <tr key={teacher.id}>
                  <td>{teacher.id}</td>
                  <td>{teacher.name}</td>
                  <td>
                    {teacher.subjects && teacher.subjects.length > 0 ? (
                      teacher.subjects.length === 1 ? (
                        teacher.subjects[0]
                      ) : (
                        <div className="subjects-dropdown">
                          <button
                            className="dropdown-btn"
                            onClick={() => toggleDropdown(teacher.id)}
                          >
                            {teacher.subjects.length} subjects ▼
                          </button>
                          {openDropdown === teacher.id && (
                            <div className="dropdown-menu">
                              {teacher.subjects.map((subject, index) => (
                                <div key={index} className="dropdown-item">
                                  {subject}
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )
                    ) : (
                      'None'
                    )}
                  </td>
                  <td>{getDisplayDepartment(teacher.department)}</td>
                  <td>
                    <button className="action-btn view" onClick={() => handleView(teacher)}>👁️ View</button>
                    <button className="action-btn edit" onClick={() => handleEdit(teacher)}>✏️ Edit</button>
                    <button className="action-btn delete" onClick={() => handleDelete(teacher)}>🗑️ Delete</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="teacher-list-form-buttons">
        <button type="button" className="teacher-list-secondary" onClick={onBack} style={{ padding: '8px 16px', fontSize: '13px', marginTop: '10px' }}>
          Back
        </button>
      </div>

      {isViewModalOpen && selectedTeacher && (
        <div className="modal-overlay" onClick={closeModals}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h2>Teacher Details</h2>
            <div className="detail-row">
              <strong>Teacher ID:</strong> {selectedTeacher.teacher_id}
            </div>
            <div className="detail-row">
              <strong>First Name:</strong> {selectedTeacher.first_name}
            </div>
            <div className="detail-row">
              <strong>Middle Name:</strong> {selectedTeacher.middle_name || 'N/A'}
            </div>
            <div className="detail-row">
              <strong>Last Name:</strong> {selectedTeacher.last_name}
            </div>
            <div className="detail-row">
              <strong>Subjects:</strong> {selectedTeacher.subjects && selectedTeacher.subjects.length > 0 ? selectedTeacher.subjects.join(', ') : 'None'}
            </div>
            <div className="detail-row">
              <strong>Department:</strong> {getDisplayDepartment(selectedTeacher.department)}
            </div>
            <div className="detail-row">
              <strong>Email:</strong> {selectedTeacher.email}
            </div>
            <div className="modal-actions">
              <button className="btn-secondary" onClick={closeModals}>Close</button>
            </div>
          </div>
        </div>
      )}

      {isEditModalOpen && (
        <div className="modal-overlay" onClick={closeModals}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h2>Edit Infortmation</h2>
            <form className="edit-form">
              <div className="form-group">
                <label>First Name:</label>
                <input
                  type="text"
                  value={editForm.first_name}
                  onChange={(e) => setEditForm({ ...editForm, first_name: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label>Middle Name:</label>
                <input
                  type="text"
                  value={editForm.middle_name}
                  onChange={(e) => setEditForm({ ...editForm, middle_name: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label>Last Name:</label>
                <input
                  type="text"
                  value={editForm.last_name}
                  onChange={(e) => setEditForm({ ...editForm, last_name: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label>Department:</label>
                <select
                  value={editForm.department}
                  onChange={(e) => setEditForm({ ...editForm, department: e.target.value })}
                  required
                >
                  <option value="">Select Department</option>
                  <option value="College">College</option>
                  <option value="SHS">Senior High School (SHS)</option>
                  <option value="Both">Both</option>
                </select>
              </div>
              <div className="form-group">
                <label>Email:</label>
                <input
                  type="email"
                  value={editForm.email}
                  onChange={(e) => setEditForm({ ...editForm, email: e.target.value })}
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
