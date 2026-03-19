import React, { useState, useEffect } from "react";
import "../styles/StudentList.css";

const API_BASE_URL = "http://localhost:8000";

// Helper function to format student names as: Firstname M. Lastname
const normalizeYear = (year) => {
  if (!year || typeof year !== 'string') return 'Unknown';
  
  let normalized = year.trim().toLowerCase();
  
  // Handle Grade levels
  if (normalized.includes('grade')) {
    normalized = normalized.replace(/grade\s*(\d+)/i, 'Grade $1');
    return normalized.charAt(0).toUpperCase() + normalized.slice(1);
  }
  
  // Strip common suffixes
  normalized = normalized.replace(/\s*(college|year level)/gi, '');
  
  // Standardize ordinal formats
  const ordinalMap = {
    '1st': '1st Year',
    '2nd': '2nd Year', 
    '3rd': '3rd Year',
    '4th': '4th Year',
    '1': '1st Year',
    '2': '2nd Year',
    '3': '3rd Year',
    '4': '4th Year'
  };
  
  for (const [key, value] of Object.entries(ordinalMap)) {
    if (normalized.includes(key)) {
      return value;
    }
  }
  
  // Fallback: title case
  return normalized.split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

const formatStudentName = (firstName, middleName, lastName) => {
  if (!firstName || !lastName) {
    return `${firstName || ''} ${lastName || ''}`.trim();
  }

  // Capitalize each word in firstName and lastName
  const capitalize = (str) => str.split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()).join(' ');

  const capitalizedFirst = capitalize(firstName);
  const capitalizedLast = capitalize(lastName);

  // Get middle initial if middle name exists, capitalized
  const middleInitial = middleName ? ` ${middleName.charAt(0).toUpperCase()}.` : '';

  return `${capitalizedFirst}${middleInitial} ${capitalizedLast}`.trim();
};

export default function StudentList({ onBack }) {
  // Multi-select deletion states
  const [selectedStudents, setSelectedStudents] = useState(new Set());
  const [selectAll, setSelectAll] = useState(false);
  const [bulkDeleting, setBulkDeleting] = useState(false);

  // const [students, setStudents] = useState([]); // Deprecated, use rawStudents
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

// const [searchTerm, setSearchTerm] = useState(""); // Deprecated
// const [selectedCourse, setSelectedCourse] = useState(null); // ESLint unused - deprecated
  const [selectedStudent, setSelectedStudent] = useState(null);
  const [isViewModalOpen, setIsViewModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
const [editForm, setEditForm] = useState({});
  const [courses, setCourses] = useState([]);
  const [coursesLoading, setCoursesLoading] = useState(true);
  const [coursesError, setCoursesError] = useState(null);
  const [sortBy, setSortBy] = useState("name");
  const [classes, setClasses] = useState([]);
  const [classesLoading, setClassesLoading] = useState(true);
  const [classesError, setClassesError] = useState(null);
  const [isEnrollModalOpen, setIsEnrollModalOpen] = useState(false);
  const [studentToEnroll, setStudentToEnroll] = useState(null);
  
  // State for enrollment type selection in modal
  const [enrollType, setEnrollType] = useState(null); // 'single' or 'all'
  
  // State for "Enroll to All Subjects" feature
  const [isEnrollAllModalOpen, setIsEnrollAllModalOpen] = useState(false);
  const [studentToEnrollAll, setStudentToEnrollAll] = useState(null);
  const [enrollAllLoading, setEnrollAllLoading] = useState(false);
const [enrollAllResult, setEnrollAllResult] = useState(null);

  // Advanced filter states (copied from Analytics)
  const [rawStudents, setRawStudents] = useState([]);
  const [filteredStudentsState, setFilteredStudentsState] = useState([]);
  const [filters, setFilters] = useState({
    search: '',
    course: 'All',
    yearLevel: 'All'
  });
  const [yearLevels, setYearLevels] = useState([]);

  // Pagination states
  const [currentPage, setCurrentPage] = useState(1);

  const studentsPerPage = 10;

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
        name: formatStudentName(student.first_name, student.middle_name, student.last_name),
        course: student.course,
        year: student.year || 'Unknown',
        section: 'A', // Default section since not in API
        email: student.email,
        contact: 'N/A', // Not in API
        _id: student._id
      }));
      
      // Set raw data and derive year levels like Analytics (with normalization)
      transformedStudents.forEach(s => {
        s.normalizedYear = normalizeYear(s.year);
      });
      setRawStudents(transformedStudents);
      const yearSet = new Set(transformedStudents.map(s => s.normalizedYear));
      const sortedYears = Array.from(yearSet).sort();
      setYearLevels(['All', ...sortedYears]);
      
      // Initial filter
      setFilteredStudentsState(transformedStudents);
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

  const filteredStudentsStateSorted = [...filteredStudentsState].sort((a, b) => {
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

  // Pagination logic
  const indexOfLastStudent = currentPage * studentsPerPage;
  const indexOfFirstStudent = indexOfLastStudent - studentsPerPage;
  const currentStudents = filteredStudentsStateSorted.slice(indexOfFirstStudent, indexOfLastStudent);
  const totalPages = Math.ceil(filteredStudentsStateSorted.length / studentsPerPage);

  const handleFilterChange = (key, value) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  const applyFilters = () => {
    let data = [...rawStudents];

    // Search filter
    if (filters.search.trim()) {
      const query = filters.search.toLowerCase().trim();
      data = data.filter(student =>
        student.name.toLowerCase().includes(query) ||
        student.course.toLowerCase().includes(query) ||
        student.id.toLowerCase().includes(query)
      );
    }

    // Course filter
    if (filters.course !== 'All') {
      data = data.filter(s => s.course === filters.course);
    }

    // Year filter (normalized)
    if (filters.yearLevel !== 'All') {
      data = data.filter(s => s.normalizedYear === filters.yearLevel);
    }

    setFilteredStudentsState(data);
    setCurrentPage(1); // Reset to first page
  };

  const handleView = (student) => {
    setSelectedStudent(student);
    setIsViewModalOpen(true);
  };

  const handleEdit = (student) => {
    setSelectedStudent(student);
    
    // Parse formatted name back to first/middle/last
    const { firstName, middleName, lastName } = parseFormattedName(student.name);
    
    setEditForm({ 
      ...student, 
      firstName,
      middleName, 
      lastName 
    });
    setIsEditModalOpen(true);
  };

  // Helper to parse formatted name (handles First M. Last, First Last, etc.)
  const parseFormattedName = (formattedName) => {
    if (!formattedName) return { firstName: '', middleName: '', lastName: '' };
    
    const parts = formattedName.trim().split(/\s+/);
    if (parts.length === 0) return { firstName: '', middleName: '', lastName: '' };
    
    let firstName = parts[0];
    let lastName = '';
    let middleName = '';
    
    if (parts.length === 2) {
      // First Last
      lastName = parts[1];
    } else if (parts.length === 3 && parts[1].endsWith('.')) {
      // First M. Last
      firstName = parts[0];
      middleName = parts[1].slice(0, -1); // Remove dot
      lastName = parts[2];
    } else {
      // First Middle Last or more complex - take first as first, last as last, middle as rest
      firstName = parts[0];
      lastName = parts[parts.length - 1];
      middleName = parts.slice(1, -1).join(' ');
    }
    
    return { firstName, middleName, lastName };
  };

  const handleSaveEdit = async () => {
    try {
      const updateData = {
        student_id: editForm.id,
        first_name: editForm.firstName || '',
        middle_name: editForm.middleName || '',
        last_name: editForm.lastName || '',
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

  const toggleSelection = (studentId) => {
    setSelectedStudents(prev => {
      const newSet = new Set(prev);
      if (newSet.has(studentId)) {
        newSet.delete(studentId);
        // If deselecting last item, clear select all
        if (newSet.size === 0) setSelectAll(false);
      } else {
        newSet.add(studentId);
      }
      return newSet;
    });
  };

  const toggleSelectAll = () => {
    setSelectAll(!selectAll);
    if (!selectAll) {
      // Select all visible students (filtered + current page)
      const visibleIds = currentStudents.map(s => s.id);
      setSelectedStudents(new Set(visibleIds));
    } else {
      setSelectedStudents(new Set());
    }
  };

  const handleBulkDelete = async () => {
    if (selectedStudents.size === 0) return;

    // Enhanced confirmation popup
    const studentIds = Array.from(selectedStudents);
    const result = window.confirm(
      `⚠️ PERMANENTLY DELETE ${selectedStudents.size} STUDENT(S)?\n\n` +
      `This will remove:\n` +
      `• Database records\n` +
      `• Face images (StudentFaceData/${studentIds[0] || 'ID'}...)\n` +
      `• Face encodings (.pkl files)\n\n` +
      `Selected: ${studentIds.slice(0, 3).join(', ')}${studentIds.length > 3 ? ` +${studentIds.length-3} more` : ''}\n\n` +
      `This action CANNOT be undone.\n\n` +
      `Continue?`
    );
    
    if (!result) return;

    setBulkDeleting(true);
    setError(null);

    try {
      // Parallel DELETE calls for efficiency
      const deletePromises = Array.from(selectedStudents).map(async (studentId) => {
        const response = await fetch(`${API_BASE_URL}/students/${studentId}`, {
          method: "DELETE",
        });
        if (!response.ok) {
          const result = await response.json();
          throw new Error(`Failed to delete ${studentId}: ${result.detail || result.error}`);
        }
        return studentId;
      });

      const results = await Promise.allSettled(deletePromises);
      
      // Check for failures
      const failures = results.filter(r => r.status === 'rejected');
      if (failures.length > 0) {
        throw new Error(`${failures.length}/${selectedStudents.size} deletions failed`);
      }

      // Success - refresh list and clear selection
      setSelectedStudents(new Set());
      setSelectAll(false);
      fetchStudents();
      
      // Success message
      window.alert(`✅ Successfully deleted ${selectedStudents.size} student(s)!\n\nFiles and database records have been permanently removed.`);
      
    } catch (error) {
      setError(`Bulk delete failed: ${error.message}`);
      console.error('Bulk delete error:', error);
    } finally {
      setBulkDeleting(false);
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
    setEnrollType(null); // Reset enrollment type selection
    setIsEnrollModalOpen(true);
  };

  const getAuthHeaders = () => {
    const token = localStorage.getItem("token");
    return token ? { Authorization: `Bearer ${token}` } : {};
  };

  const handleEnrollStudent = async (classId) => {
    if (!studentToEnroll) return;

    try {
      const response = await fetch(`${API_BASE_URL}/classes/${classId}/enroll`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...getAuthHeaders(),
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

  // Handle "Enroll to All Subjects" - bulk enrollment based on course/strand
  const handleEnrollToAllSubjects = (student) => {
    if (!student?.course || !student.course.trim()) {
      setStudentToEnrollAll(student);
      setEnrollAllResult({
        success: false,
        message: "Cannot enroll: student has no course/strand assigned."
      });
      setIsEnrollAllModalOpen(true);
      return;
    }

    setStudentToEnrollAll(student);
    setEnrollAllResult(null);
    setIsEnrollAllModalOpen(true);
  };

  const handleConfirmEnrollAll = async () => {
    if (!studentToEnrollAll) return;

    setEnrollAllLoading(true);
    setEnrollAllResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/enrollments/student/${studentToEnrollAll.id}/all-subjects`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ enrolled_by: "admin" }),
      });

      const result = await response.json();

      if (response.ok) {
        setEnrollAllResult({
          success: true,
          message: result.message,
          enrolled_count: result.enrolled_count,
          skipped_count: result.skipped_count,
          total_classes_found: result.total_classes_found,
          student_name: result.student_name,
          course: result.course,
          enrollments: result.enrollments || []
        });
      } else {
        setEnrollAllResult({
          success: false,
          message: result.detail || result.error || "Failed to enroll student to all subjects"
        });
      }
    } catch (error) {
      setEnrollAllResult({
        success: false,
        message: `Error enrolling student: ${error.message}`
      });
    } finally {
      setEnrollAllLoading(false);
    }
  };

  const closeModals = () => {
    setIsViewModalOpen(false);
    setIsEditModalOpen(false);
    setIsEnrollModalOpen(false);
    setIsEnrollAllModalOpen(false);
    setSelectedStudent(null);
    setStudentToEnroll(null);
    setStudentToEnrollAll(null);
    setEnrollAllLoading(false);
    setEnrollType(null);
  };

  // Remove old state setters - no longer used
  // setSearchTerm, setSelectedCourse replaced by filters

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
        <div className="metric-select-container">
          <label className="metric-select-label">
            <input
              type="checkbox"
              checked={selectAll}
              onChange={toggleSelectAll}
              className="select-all-checkbox"
            />
            <span className="select-all-text">
              {selectAll ? 'Deselect All' : 'Select All'} 
              ({currentStudents.length} visible)
            </span>
          </label>
          {selectedStudents.size > 0 && (
            <span className="selected-count">
              {selectedStudents.size} {selectedStudents.size === 1 ? 'selected' : 'selected'}
            </span>
          )}
        </div>
        
        <div className="metric-card">
          <div className="metric-icon">👥</div>
          <div className="metric-content">
            <div className="metric-value">{filteredStudentsState.length}</div>
            <div className="metric-label">Filtered Students</div>
          </div>
        </div>
      </div>

      {/* Advanced Filters (Copied from Analytics) */}
      <div className="filters-section">
        <div className="filters-grid">
          {/* Search */}
          <div className="filter-group">
            <label>Search</label>
            <input
              type="text"
              className="filter-select"
              placeholder="Name, course, or ID..."
              value={filters.search}
              onChange={(e) => handleFilterChange('search', e.target.value)}
            />
          </div>
          
          {/* Course */}
          <div className="filter-group">
            <label>Course</label>
            <select 
              className="filter-select" 
              value={filters.course} 
              onChange={(e) => handleFilterChange('course', e.target.value)}
              disabled={coursesLoading || coursesError}
            >
              <option value="All">All Courses</option>
              {courses.map((course) => (
                <option key={course.code} value={course.code}>
                  {course.code} - {course.name}
                </option>
              ))}
            </select>
            {coursesLoading && <span className="loading-text">Loading...</span>}
            {coursesError && <span className="error-text">{coursesError}</span>}
          </div>

          {/* Year Level */}
          <div className="filter-group">
            <label>Year Level</label>
            <select 
              className="filter-select" 
              value={filters.yearLevel} 
              onChange={(e) => handleFilterChange('yearLevel', e.target.value)}
            >
              <option value="All">All</option>
              {yearLevels.map(year => (
                <option key={year} value={year}>{year}</option>
              ))}
            </select>
          </div>

          {/* Sort */}
          <div className="filter-group">
            <label>Sort</label>
            <select
              className="filter-select"
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
            >
              <option value="name">Name</option>
              <option value="course">Course</option>
              <option value="year">Year</option>
              <option value="id">ID</option>
            </select>
          </div>

          {/* Apply Button */}
          <button 
            className="apply-filters-btn" 
            onClick={applyFilters}
            disabled={loading}
          >
            🔍 Apply Filters ({filteredStudentsState.length})
          </button>
        </div>
      </div>

      {/* Student Cards Grid */}
      <div className="students-grid">
        {currentStudents.length > 0 ? (
          currentStudents.map((student) => (
            <div key={student.id} className="student-card">
              <div className="student-header">
                <input
                  type="checkbox"
                  checked={selectedStudents.has(student.id)}
                  onChange={() => toggleSelection(student.id)}
                  className="student-checkbox"
                />
                <div className={`student-avatar ${selectedStudents.has(student.id) ? 'selected' : ''}`}>
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
          ))
        ) : filteredStudentsStateSorted.length > 0 ? (
          <div className="no-results">No students on this page</div>
        ) : (
          <div className="no-results">No students found matching filters</div>
        )}
      </div>

      {/* Pagination Controls */}
        {totalPages > 1 && (
          <div className="pagination-controls">
            <button 
              className="pagination-btn prev"
              onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
              disabled={currentPage === 1}
            >
              ← Previous
            </button>
            <span className="pagination-info">
              Page <strong>{currentPage}</strong> of <strong>{totalPages}</strong> 
              ({filteredStudentsStateSorted.length} total)
            </span>
            <button 
              className="pagination-btn next"
              onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
              disabled={currentPage === totalPages}
            >
              Next →
            </button>
          </div>
        )}


      {/* Page Actions */}
      <div className="page-actions">
        {selectedStudents.size > 0 && (
          <button 
            className="bulk-delete-btn" 
            onClick={handleBulkDelete}
            disabled={bulkDeleting}
          >
            <span className="btn-icon">🗑️</span>
            <span className="btn-text">
              {bulkDeleting ? 'Deleting...' : `Delete ${selectedStudents.size} Selected`}
            </span>
          </button>
        )}
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
              <div className="name-group">
                <div className="form-group">
                  <label>First Name:</label>
                  <input
                    type="text"
                    value={editForm.firstName || ''}
                    onChange={(e) => setEditForm({ ...editForm, firstName: e.target.value })}
                    placeholder="First name"
                  />
                </div>
                <div className="form-group">
                  <label>Middle Name:</label>
                  <input
                    type="text"
                    value={editForm.middleName || ''}
                    onChange={(e) => setEditForm({ ...editForm, middleName: e.target.value })}
                    placeholder="Middle name (optional)"
                  />
                </div>
                <div className="form-group">
                  <label>Last Name:</label>
                  <input
                    type="text"
                    value={editForm.lastName || ''}
                    onChange={(e) => setEditForm({ ...editForm, lastName: e.target.value })}
                    placeholder="Last name"
                  />
                </div>
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

      {/* Enroll Modal - with options for Single Class or All Subjects */}
      {isEnrollModalOpen && studentToEnroll && (
        <div className="modal-overlay" onClick={closeModals}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h2>Enroll Student</h2>
            <p>Enroll {studentToEnroll.name} (ID: {studentToEnroll.id})</p>
            
            {/* Enrollment Type Selection */}
            {enrollType === null && (
              <div className="enroll-options">
                <p>Choose enrollment type:</p>
                <button 
                  className="enroll-option-btn" 
                  onClick={() => setEnrollType('single')}
                >
                  <span className="btn-icon">📝</span>
                  <span className="btn-text">Enroll to Single Class</span>
                </button>
                <button 
                  className="enroll-option-btn" 
                  onClick={() => {
                    setEnrollType('all');
                    handleEnrollToAllSubjects(studentToEnroll);
                    setIsEnrollModalOpen(false);
                  }}
                >
                  <span className="btn-icon">📚</span>
                  <span className="btn-text">Enroll to All Subjects</span>
                </button>
              </div>
            )}

            {/* Single Class Enrollment */}
            {enrollType === 'single' && (
              <>
                <p>Select a class to enroll in:</p>
                <div className="classes-list">
                  {classesLoading ? (
                    <div>Loading classes...</div>
                  ) : classesError ? (
                    <div className="error">Error loading classes: {classesError}</div>
                  ) : (
                    classes.map((cls) => (
                      <div key={cls._id} className="class-item">
                        <div><strong>{cls.class_name}</strong> <span className="class-code">({cls.class_code})</span></div>
                        <div className="class-details">
                          <div className="class-teacher">Teacher: {cls.teacher_id}</div>
                          <div className="class-room">Room: {cls.room}</div>
                          <button
                            className="enroll-btn"
                            onClick={() => handleEnrollStudent(cls._id)}
                          >
                            Enroll
                          </button>
                        </div>
                      </div>
                    ))
                  )}
                </div>
                <div className="modal-actions">
                  <button className="btn-secondary" onClick={() => setEnrollType(null)}>Back</button>
                  <button className="btn-secondary" onClick={closeModals}>Cancel</button>
                </div>
              </>
            )}

            {/* If only showing initial options, just show cancel */}
            {enrollType === null && (
              <div className="modal-actions">
                <button className="btn-secondary" onClick={closeModals}>Cancel</button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Enroll To All Subjects Modal */}
      {isEnrollAllModalOpen && studentToEnrollAll && (
        <div className="modal-overlay" onClick={closeModals}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h2>Enroll to All Subjects</h2>
            <p>
              Student: <strong>{studentToEnrollAll.name}</strong> (ID: {studentToEnrollAll.id})
            </p>
            <p>
              Course/Strand: <strong>{studentToEnrollAll.course || "Not assigned"}</strong>
            </p>

            {!enrollAllResult && (
              <p>
                This will enroll the student in all classes mapped to their course/strand.
                Existing enrollments will be skipped.
              </p>
            )}

            {enrollAllResult && (
              <div className={`enroll-all-result ${enrollAllResult.success ? "success" : "error"}`}>
                <p>{enrollAllResult.message}</p>
                {enrollAllResult.success && (
                  <>
                    <p>Enrolled: {enrollAllResult.enrolled_count || 0}</p>
                    <p>Skipped (already enrolled): {enrollAllResult.skipped_count || 0}</p>
                    <p>Total classes found: {enrollAllResult.total_classes_found || 0}</p>
                  </>
                )}
              </div>
            )}

            <div className="modal-actions">
              <button className="btn-secondary" onClick={closeModals} disabled={enrollAllLoading}>
                {enrollAllResult ? "Close" : "Cancel"}
              </button>
              {!enrollAllResult && (
                <button
                  className="btn-primary"
                  onClick={handleConfirmEnrollAll}
                  disabled={enrollAllLoading || !studentToEnrollAll.course}
                >
                  {enrollAllLoading ? "Enrolling..." : "Confirm Enroll"}
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
