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
  const [courses, setCourses] = useState([]);
  const [coursesLoading, setCoursesLoading] = useState(true);
  const [coursesError, setCoursesError] = useState(null);
  const [sortBy, setSortBy] = useState("name");
  const [classes, setClasses] = useState([]);
  const [classesLoading, setClassesLoading] = useState(true);
  const [classesError, setClassesError] = useState(null);
  const [isEnrollModalOpen, setIsEnrollModalOpen] = useState(false);
  const [studentToEnroll, setStudentToEnroll] = useState(null);

  // Fetch students, courses, and classes from API
  useEffect(() => {
    fetchStudents();
    fetchCourses();
    fetchClasses();
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

  const fetchCourses = async () => {
    try {
      setCoursesLoading(true);
      const response = await fetch(`${API_BASE_URL}/courses`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      if (!data.courses || !Array.isArray(data.courses)) {
        throw new Error('Invalid response format: expected {courses: [...]}');
      }
      setCourses(data.courses);
      setCoursesError(null);
    } catch (err) {
      if (err.name === 'TypeError' && err.message === 'Failed to fetch') {
        setCoursesError('Network error: Unable to connect to the server. Please ensure the backend is running on http://localhost:8000');
      } else {
        setCoursesError(`Error fetching courses: ${err.message}`);
      }
      console.error('Error fetching courses:', err);
    } finally {
      setCoursesLoading(false);
    }
  };

  const fetchClasses = async () => {
    try {
      setClassesLoading(true);
      const response = await fetch(`${API_BASE_URL}/classes`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      if (!data.classes || !Array.isArray(data.classes)) {
        throw new Error('Invalid response format: expected {classes: [...]}');
      }
      setClasses(data.classes);
      setClassesError(null);
    } catch (err) {
      if (err.name === 'TypeError' && err.message === 'Failed to fetch') {
        setClassesError('Network error: Unable to connect to the server. Please ensure the backend is running on http://localhost:8000');
      } else {
        setClassesError(`Error fetching classes: ${err.message}`);
      }
      console.error('Error fetching classes:', err);
    } finally {
      setClassesLoading(false);
    }
  };

  const filteredStudents = students
    .filter(student => {
      const matchesSearch = student.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        student.course.toLowerCase().includes(searchTerm.toLowerCase()) ||
        student.id.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesCourse = selectedCourse === null || student.course === selectedCourse;
      return matchesSearch && matchesCourse;
    })
    .sort((a, b) => {
      switch (sortBy) {
        case "name":
          return a.name.localeCompare(b.name);
        case "course":
          return a.course.localeCompare(b.course);
        case "year":
          return a.year.localeCompare(b.year);
        case "id":
          return a.id.localeCompare(b.id);
        default:
          return 0;
      }
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

      const response = await fetch(`${API_BASE_URL}/students/${selectedStudent.id}`, {
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

  const handleEnroll = (student) => {
    setStudentToEnroll(student);
    setIsEnrollModalOpen(true);
  };

  const handleEnrollStudent = async (classId) => {
    if (!studentToEnroll) return;

    try {
      const response = await fetch(`${API_BASE_URL}/classes/${classId}/enroll`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ student_id: studentToEnroll.id }),
      });

      if (response.ok) {
        alert(`Student ${studentToEnroll.name} enrolled successfully!`);
        setIsEnrollModalOpen(false);
        setStudentToEnroll(null);
      } else {
        const result = await response.json();
        const errorMessage = result.detail || result.error || JSON.stringify(result);
        setError(`Failed to enroll student: ${errorMessage}`);
      }
    } catch (error) {
      setError(`Error enrolling student: ${error.message}`);
    }
  };

  const closeModals = () => {
    setIsViewModalOpen(false);
    setIsEditModalOpen(false);
    setIsEnrollModalOpen(false);
    setSelectedStudent(null);
    setStudentToEnroll(null);
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
      {/* Breadcrumbs */}
      <nav className="breadcrumbs">
        <span>Dashboard</span>
        <span className="breadcrumb-separator">/</span>
        <span>Students</span>
        <span className="breadcrumb-separator">/</span>
        <span className="breadcrumb-current">List</span>
      </nav>

      <h1>Student Management</h1>

      {/* Metric Header */}
      <div className="metric-header">
        <div className="metric-card">
          <div className="metric-icon">👥</div>
          <div className="metric-content">
            <div className="metric-value">{filteredStudents.length}</div>
            <div className="metric-label">Total Students</div>
          </div>
        </div>
      </div>

      {/* Controls Bar */}
      <div className="controls-bar">
        <div className="search-section">
          <div className="search-wrapper">
            <span className="search-icon">🔍</span>
            <input
              type="text"
              placeholder="Search by name, course, or ID..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="search-input"
            />
          </div>
        </div>

        <div className="filters-section">
          <div className="filter-group">
            <label htmlFor="course-filter">Course:</label>
            <select
              id="course-filter"
              value={selectedCourse || ""}
              onChange={(e) => setSelectedCourse(e.target.value === "" ? null : e.target.value)}
              className="filter-select"
              disabled={coursesLoading || coursesError}
            >
              <option value="">All Courses</option>
              {courses.map((course) => (
                <option key={course.code} value={course.code}>
                  {course.code} - {course.name}
                </option>
              ))}
            </select>
            {coursesLoading && <span className="loading-text">Loading...</span>}
            {coursesError && <span className="error-text">{coursesError}</span>}
          </div>

          <div className="filter-group">
            <label htmlFor="sort-filter">Sort by:</label>
            <select
              id="sort-filter"
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="filter-select"
            >
              <option value="name">Name</option>
              <option value="course">Course</option>
              <option value="year">Year Level</option>
              <option value="id">Student ID</option>
            </select>
          </div>
        </div>
      </div>

      {/* Student Cards Grid */}
      <div className="students-grid">
        {filteredStudents.map((student) => (
          <div key={student.id} className="student-card">
            <div className="student-header">
              <div className="student-avatar">
                <span className="avatar-icon">👤</span>
              </div>
              <div className="student-basic-info">
                <h3 className="student-name">{student.name}</h3>
                <p className="student-id">STU {student.id}</p>
              </div>
            </div>

            <div className="student-details">
              <div className="detail-row">
                <span className="detail-label">Course:</span>
                <span className="detail-value">{student.course}</span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Year:</span>
                <span className="detail-value">{student.year}</span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Section:</span>
                <span className="detail-value">{student.section || <span className="muted-placeholder">Not assigned</span>}</span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Email:</span>
                <span className="detail-value">{student.email}</span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Contact:</span>
                <span className="detail-value">{student.contact || <span className="muted-placeholder">Not provided</span>}</span>
              </div>
            </div>

            <div className="student-actions">
              <button className="action-btn view" onClick={() => handleView(student)}>
                <span className="btn-icon">👁️</span>
                <span className="btn-text">View</span>
              </button>
              <button className="action-btn enroll" onClick={() => handleEnroll(student)}>
                <span className="btn-icon">📝</span>
                <span className="btn-text">Enroll</span>
              </button>
              <button className="action-btn edit" onClick={() => handleEdit(student)}>
                <span className="btn-icon">✏️</span>
                <span className="btn-text">Edit</span>
              </button>
              <button className="action-btn delete" onClick={() => handleDelete(student)}>
                <span className="btn-icon">🗑️</span>
                <span className="btn-text">Delete</span>
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Back Button */}
      <div className="page-actions">
        <button type="button" className="back-btn" onClick={onBack}>
          <span className="btn-icon">←</span>
          <span className="btn-text">Back to Dashboard</span>
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

      {/* Enroll Modal */}
      {isEnrollModalOpen && studentToEnroll && (
        <div className="modal-overlay" onClick={closeModals}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h2>Enroll Student</h2>
            <p>Enroll {studentToEnroll.name} (ID: {studentToEnroll.id}) in a class:</p>
            <div className="classes-list">
              {classesLoading ? (
                <div>Loading classes...</div>
              ) : classesError ? (
                <div className="error">Error loading classes: {classesError}</div>
              ) : (
                classes.map((cls) => (
                  <div key={cls._id} className="class-item" style={{ marginBottom: '10px', padding: '10px', border: '1px solid #ccc', borderRadius: '5px' }}>
                    <div><strong>{cls.class_name}</strong> ({cls.class_code})</div>
                    <div>Teacher: {cls.teacher_id}</div>
                    <div>Room: {cls.room}</div>
                    <button
                      className="btn-primary"
                      onClick={() => handleEnrollStudent(cls._id)}
                      style={{ marginTop: '5px' }}
                    >
                      Enroll in this class
                    </button>
                  </div>
                ))
              )}
            </div>
            <div className="modal-actions">
              <button className="btn-secondary" onClick={closeModals}>Cancel</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
