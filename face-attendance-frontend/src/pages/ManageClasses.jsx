import React, { useState, useEffect, useCallback } from "react";
import "../styles/ManageClasses.css";

const API_BASE_URL = "http://localhost:8000";

export default function ManageClasses({ onBack }) {
  const [classes, setClasses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [teachers, setTeachers] = useState([]);
  const [teacherSuggestions, setTeacherSuggestions] = useState([]);
  const [showTeacherSuggestions, setShowTeacherSuggestions] = useState(false);
  const [editTeacherSuggestions, setEditTeacherSuggestions] = useState([]);
  const [showEditTeacherSuggestions, setShowEditTeacherSuggestions] = useState(false);
// have filter for courses and days of the week
  const [courses, setCourses] = useState([]);
  const [newClass, setNewClass] = useState({
    name: "",
    teacher: "",
    teacherId: "",
    teacherDepartment: "",
    room: "",
    day: "Mon",
    startTime: "08:00",
    endTime: "09:00",
    courses: [],
  });

  const [isAddFormOpen, setIsAddFormOpen] = useState(false);
  const [selectedDay, setSelectedDay] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");

  const [selectedClass, setSelectedClass] = useState(null);
  const [isViewModalOpen, setIsViewModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [editForm, setEditForm] = useState({});
  const [isAddStudentModalOpen, setIsAddStudentModalOpen] = useState(false);
  const [classForStudentAdd, setClassForStudentAdd] = useState(null);
  const [allStudents, setAllStudents] = useState([]);
  const [studentsLoading, setStudentsLoading] = useState(false);
  const [studentSearchTerm, setStudentSearchTerm] = useState("");
  const [addStudentLoading, setAddStudentLoading] = useState(false);
  const [addStudentFeedback, setAddStudentFeedback] = useState(null);
  const resetNewClassForm = useCallback(() => {
    setNewClass({
      name: "",
      teacher: "",
      teacherId: "",
      teacherDepartment: "",
      room: "",
      day: "Mon",
      startTime: "08:00",
      endTime: "09:00",
      courses: [],
    });
    setTeacherSuggestions([]);
    setShowTeacherSuggestions(false);
  }, []);

  const closeAddClassModal = useCallback(() => {
    setIsAddFormOpen(false);
    resetNewClassForm();
  }, [resetNewClassForm]);

  const fetchClasses = useCallback(async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/classes`);
      if (!response.ok) {
        throw new Error('Failed to fetch classes');
      }
      const data = await response.json();
      // Transform API data to match component expectations
      const transformedClasses = data.classes.map(cls => {
        const teacherObj = teachers.find(t => t.id === cls.teacher_id);
        return {
          id: cls.class_code,
          name: cls.class_name,
          room: cls.room,
          teacher: teacherObj ? teacherObj.name : cls.teacher_id, // Show name if found, else ID
          teacherId: cls.teacher_id, // Keep ID for editing
          students: cls.enrolled_students?.length || 0,
          day: cls.schedule.split(' ')[0] || 'Mon',
          startTime: cls.schedule.split(' ')[1]?.split('-')[0] || '09:00',
          endTime: cls.schedule.split(' ')[1]?.split('-')[1] || '10:00',
          courses: cls.courses || [],
          enrolledStudents: cls.enrolled_students || [],
          _id: cls._id
        };
      });
      setClasses(transformedClasses);
      setError(null);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching classes:', err);
    } finally {
      setLoading(false);
    }
  }, [teachers]);

  // Fetch teachers and courses from API
  useEffect(() => {
    fetchTeachers();
    fetchCourses();
  }, []);

  // Fetch classes when teachers are loaded
  useEffect(() => {
    if (teachers.length > 0) {
      fetchClasses();
    }
  }, [teachers, fetchClasses]);

  const fetchTeachers = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/teachers`);
      if (!response.ok) {
        throw new Error('Failed to fetch teachers');
      }
      const data = await response.json();
      const transformedTeachers = data.teachers.map(teacher => ({
        id: teacher.teacher_id,
        name: `${teacher.first_name} ${teacher.last_name}`,
        department: teacher.department
      }));
      setTeachers(transformedTeachers);
    } catch (err) {
      console.error('Error fetching teachers:', err);
    }
  };

  const fetchCourses = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/courses`);
      if (!response.ok) {
        throw new Error('Failed to fetch courses');
      }
      const data = await response.json();
      setCourses(data.courses);
    } catch (err) {
      console.error('Error fetching courses:', err);
    }
  };

  const getAuthHeaders = () => {
    const token = localStorage.getItem("token");
    return token ? { Authorization: `Bearer ${token}` } : {};
  };

  const fetchStudentsForEnrollment = useCallback(async () => {
    try {
      setStudentsLoading(true);
      const response = await fetch(`${API_BASE_URL}/students`);
      if (!response.ok) {
        throw new Error("Failed to fetch students");
      }
      const data = await response.json();
      const transformedStudents = (data.students || []).map((student) => ({
        id: student.student_id,
        name: `${student.first_name || ""} ${student.last_name || ""}`.trim(),
        course: student.course || "",
        year: student.year || ""
      }));
      setAllStudents(transformedStudents);
    } catch (err) {
      setError(`Error fetching students: ${err.message}`);
    } finally {
      setStudentsLoading(false);
    }
  }, []);

const dayOptions = [
    { label: "Monday", value: "Mon" },
    { label: "Tuesday", value: "Tue" },
    { label: "Wednesday", value: "Wed" },
    { label: "Thursday", value: "Thu" },
    { label: "Friday", value: "Fri" },
    { label: "Saturday", value: "Sat" },
    { label: "Sunday", value: "Sun" }
  ];

  const handleAddClass = useCallback(async () => {
    if (newClass.name && newClass.teacher && newClass.room && newClass.day && newClass.startTime && newClass.endTime) {
      // Generate a unique class code that doesn't conflict with existing ones
      const existingCodes = new Set(classes.map(c => c.id));
      let classId = '';
      let counter = 1;
      
      while (true) {
        const candidateId = `CLS${String(counter).padStart(3, '0')}`;
        if (!existingCodes.has(candidateId)) {
          classId = candidateId;
          break;
        }
        counter++;
        // Safety check to prevent infinite loop
        if (counter > 1000) {
          // Fallback to timestamp-based ID if we can't find a unique ID
          classId = `CLS${Date.now()}`;
          break;
        }
      }

      try {
        const classData = {
          class_code: classId,
          class_name: newClass.name,
          teacher_id: newClass.teacherId,
          schedule: `${newClass.day} ${newClass.startTime}-${newClass.endTime}`,
          room: newClass.room,
          courses: newClass.courses
        };

        const response = await fetch(`${API_BASE_URL}/classes`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(classData),
        });

        const result = await response.json();

        if (response.ok) {
          closeAddClassModal();
          fetchClasses(); // Refresh the list
        } else {
          const errorMsg = result.detail || (typeof result.error === 'string' ? result.error : JSON.stringify(result.error));
          setError(`Failed to create class: ${errorMsg}`);
        }
      } catch (error) {
        setError(`Error creating class: ${error.message}`);
      }
    }
  }, [newClass, classes, fetchClasses, closeAddClassModal]);

  const handleTeacherInputChange = (e) => {
    const value = e.target.value;
    setNewClass({ ...newClass, teacher: value });

    if (value.length > 0) {
      const filtered = teachers.filter(teacher =>
        teacher.name.toLowerCase().includes(value.toLowerCase()) ||
        teacher.id.toLowerCase().includes(value.toLowerCase())
      );
      setTeacherSuggestions(filtered.slice(0, 5)); // Limit to 5 suggestions
      setShowTeacherSuggestions(true);
    } else {
      setTeacherSuggestions([]);
      setShowTeacherSuggestions(false);
    }
  };

  const handleTeacherSelect = (teacher) => {
    const normalizedDepartment = teacher.department === "College" ? "college" :
                                teacher.department === "SHS" ? "senior_high" :
                                teacher.department.toLowerCase();
    setNewClass({ ...newClass, teacher: teacher.name, teacherId: teacher.id, teacherDepartment: normalizedDepartment });
    setShowTeacherSuggestions(false);
  };

  const handleEditTeacherInputChange = (e) => {
    const value = e.target.value;
    setEditForm({ ...editForm, teacher: value });

    if (value.length > 0) {
      const filtered = teachers.filter(teacher =>
        teacher.name.toLowerCase().includes(value.toLowerCase()) ||
        teacher.id.toLowerCase().includes(value.toLowerCase())
      );
      setEditTeacherSuggestions(filtered.slice(0, 5)); // Limit to 5 suggestions
      setShowEditTeacherSuggestions(true);
    } else {
      setEditTeacherSuggestions([]);
      setShowEditTeacherSuggestions(false);
    }
  };

  const handleEditTeacherSelect = (teacher) => {
    const newDepartment = teacher.department === "College" ? "college" :
                         teacher.department === "SHS" ? "senior_high" :
                         teacher.department.toLowerCase();
    const allowedCourses = (editForm.courses || []).filter(courseCode => {
      const course = courses.find(c => (c.code || c) === courseCode);
      const level = course ? (course.level || "unknown") : "unknown";
      return !newDepartment || newDepartment === "both" || newDepartment === level;
    });
    setEditForm({ ...editForm, teacher: teacher.name, teacherId: teacher.id, teacherDepartment: newDepartment, courses: allowedCourses });
    setShowEditTeacherSuggestions(false);
  };

  const handleCourseChange = (courseCode, isChecked, isEdit = false) => {
    const target = isEdit ? editForm : newClass;
    const setTarget = isEdit ? setEditForm : setNewClass;
    const currentCourses = target.courses || [];
    if (isChecked) {
      setTarget({ ...target, courses: [...currentCourses, courseCode] });
    } else {
      setTarget({ ...target, courses: currentCourses.filter(c => c !== courseCode) });
    }
  };

  const handleView = (cls) => {
    setSelectedClass(cls);
    setIsViewModalOpen(true);
  };

  const handleOpenAddStudent = async (cls) => {
    setClassForStudentAdd(cls);
    setStudentSearchTerm("");
    setAddStudentFeedback(null);
    setIsAddStudentModalOpen(true);
    await fetchStudentsForEnrollment();
  };

  const handleAddStudentToClass = async (student) => {
    if (!classForStudentAdd || !student?.id) return;

    try {
      setAddStudentLoading(true);
      setAddStudentFeedback(null);

      const response = await fetch(`${API_BASE_URL}/classes/${classForStudentAdd._id}/enroll-student`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...getAuthHeaders()
        },
        body: JSON.stringify({ student_id: student.id })
      });

      const result = await response.json();
      if (!response.ok) {
        if (response.status === 401) {
          localStorage.removeItem("token");
          alert("Session expired. Please log in again.");
          window.location.href = "/login";
          return;
        }
        throw new Error(result.detail || result.error || "Failed to add student");
      }

      setAddStudentFeedback(result.message || "Student added successfully");
      setClassForStudentAdd((prev) => {
        if (!prev) return prev;
        const enrolled = new Set(prev.enrolledStudents || []);
        enrolled.add(student.id);
        return {
          ...prev,
          enrolledStudents: Array.from(enrolled),
          students: enrolled.size
        };
      });
      await fetchClasses();
    } catch (err) {
      setAddStudentFeedback(`Error: ${err.message}`);
    } finally {
      setAddStudentLoading(false);
    }
  };

  const handleEdit = (cls) => {
    setSelectedClass(cls);
    const teacher = teachers.find(t => t.id === cls.teacherId);
    const rawDepartment = teacher ? teacher.department : "";
    const teacherDepartment = rawDepartment === "College" ? "college" :
                             rawDepartment === "SHS" ? "senior_high" :
                             rawDepartment.toLowerCase();
    const allowedCourses = (cls.courses || []).filter(courseCode => {
      const course = courses.find(c => (c.code || c) === courseCode);
      const level = course ? (course.level || "unknown") : "unknown";
      return !teacherDepartment || teacherDepartment === "both" || teacherDepartment === level;
    });
    setEditForm({ ...cls, teacher: cls.teacher, teacherId: cls.teacherId, teacherDepartment, courses: allowedCourses });
    setIsEditModalOpen(true);
  };

  const handleSaveEdit = async () => {
    try {
      const updateData = {
        class_code: editForm.id,
        class_name: editForm.name,
        teacher_id: editForm.teacherId,
        schedule: `${editForm.day} ${editForm.startTime}-${editForm.endTime}`,
        room: editForm.room,
        courses: editForm.courses || []
      };

      const response = await fetch(`${API_BASE_URL}/classes/${selectedClass._id}`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(updateData),
      });

      const result = await response.json();

      if (response.ok) {
        setIsEditModalOpen(false);
        setSelectedClass(null);
        fetchClasses(); // Refresh the list
      } else {
        const errorMsg = result.detail || (typeof result.error === 'string' ? result.error : JSON.stringify(result.error));
        setError(`Failed to update class: ${errorMsg}`);
      }
    } catch (error) {
      setError(`Error updating class: ${error.message}`);
    }
  };

  const handleDelete = async (cls) => {
    const ok = window.confirm(`Delete class ${cls.name} (${cls.id})?`);
    if (!ok) return;

    try {
      const response = await fetch(`${API_BASE_URL}/classes/${cls._id}`, {
        method: "DELETE",
      });

      if (response.ok) {
        fetchClasses(); // Refresh the list
      } else {
        const result = await response.json();
        const errorMsg = result.detail || (typeof result.error === 'string' ? result.error : JSON.stringify(result.error));
        setError(`Failed to delete class: ${errorMsg}`);
      }
    } catch (error) {
      setError(`Error deleting class: ${error.message}`);
    }
  };

  const closeModals = () => {
    setIsViewModalOpen(false);
    setIsEditModalOpen(false);
    setIsAddStudentModalOpen(false);
    setSelectedClass(null);
    setClassForStudentAdd(null);
    setStudentSearchTerm("");
    setAddStudentFeedback(null);
  };

  const filteredClasses = classes.filter(cls => {
    const matchesDay = selectedDay === null || cls.day === selectedDay;
    const matchesSearch = searchTerm === "" || cls.name.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesDay && matchesSearch;
  });

  const formatSchedule = (c) => `${dayOptions.find(d => d.value === c.day)?.label || c.day} ${c.startTime} - ${c.endTime}`;
  const enrolledStudentIds = new Set(classForStudentAdd?.enrolledStudents || []);
  const filteredStudentsForAdd = allStudents.filter((student) => {
    const query = studentSearchTerm.trim().toLowerCase();
    if (!query) return true;
    return (
      student.name.toLowerCase().includes(query) ||
      student.id.toLowerCase().includes(query) ||
      student.course.toLowerCase().includes(query)
    );
  });

  if (loading) {
    return (
      <div className="manage-classes">
        <h1>Manage Classes</h1>
        <div className="loading">Loading classes...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="manage-classes">
        <h1>Manage Classes</h1>
        <div className="error">Error: {error}</div>
        <button onClick={fetchClasses} className="retry-btn">Retry</button>
      </div>
    );
  }

  return (
    <div className="manage-classes">
      <h1>Manage Classes</h1>

      {/* Back Button */}
      <div className="page-actions">
        <button type="button" className="back-btn" onClick={onBack}>
          <span className="btn-icon">←</span>
          <span className="btn-text">Back to Dashboard</span>
        </button>
      </div>

      <div className="controls-toolbar" style={{ display: "flex", justifyContent: "center", alignItems: "center", margin: "20px 0", gap: "20px" }}>
        <div className="filter-group" style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <label htmlFor="day-filter" style={{ fontWeight: "600", fontSize: "14px", color: "#444" }}>Filter by Day:</label>
          <select
            id="day-filter"
            className="day-filter-select"
            value={selectedDay || ''}
            onChange={(e) => setSelectedDay(e.target.value === '' ? null : e.target.value)}
            style={{
              padding: "8px 12px",
              borderRadius: "4px",
              border: "1px solid #ccc",
              fontSize: "14px",
              backgroundColor: "white",
              color: "#333",
              cursor: "pointer",
              minWidth: "150px"
            }}
          >
            <option value="">All Days</option>
            {dayOptions.map((day) => (
              <option key={day.value} value={day.value}>
                {day.label}
              </option>
            ))}
          </select>
        </div>

        <div className="search-group" style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <label htmlFor="class-search" style={{ fontWeight: "600", fontSize: "14px", color: "#444" }}>Search:</label>
          <input
            id="class-search"
            type="text"
            placeholder="Search class name..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            style={{
              padding: "8px 12px",
              borderRadius: "4px",
              border: "1px solid #ccc",
              fontSize: "14px",
              minWidth: "200px"
            }}
          />
        </div>

        <div className="add-class-button">
          <button className="primary" onClick={() => setIsAddFormOpen(true)}>Add Class</button>
        </div>
      </div>

      {isAddFormOpen && (
        <div className="modal-overlay" onClick={closeAddClassModal}>
          <div className="modal-content add-class-modal" onClick={(e) => e.stopPropagation()}>
            <h2>Add New Class</h2>
            <form className="add-class-form-modal">
              <div className="form-group">
                <label>Class Name:</label>
                <input
                  type="text"
                  placeholder="Class Name"
                  value={newClass.name}
                  onChange={(e) => setNewClass({ ...newClass, name: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label>Teacher:</label>
                <div className="teacher-input-container">
                  <input
                    type="text"
                    placeholder="Teacher Name or ID"
                    value={newClass.teacher}
                    onChange={handleTeacherInputChange}
                    onBlur={() => setTimeout(() => setShowTeacherSuggestions(false), 200)}
                    onFocus={() => {
                      if (newClass.teacher.length > 0 && teacherSuggestions.length > 0) {
                        setShowTeacherSuggestions(true);
                      }
                    }}
                  />
                  {showTeacherSuggestions && teacherSuggestions.length > 0 && (
                    <div className="teacher-suggestions">
                      {teacherSuggestions.map((teacher) => (
                        <div
                          key={teacher.id}
                          className="teacher-suggestion-item"
                          onClick={() => handleTeacherSelect(teacher)}
                        >
                          <div className="teacher-name">{teacher.name}</div>
                          <div className="teacher-id">{teacher.id}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
              <div className="form-group">
                <label>Room:</label>
                <input
                  type="text"
                  placeholder="Room"
                  value={newClass.room}
                  onChange={(e) => setNewClass({ ...newClass, room: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label>Day:</label>
                <select
                  value={newClass.day}
                  onChange={(e) => setNewClass({ ...newClass, day: e.target.value })}
                  className="day-select"
                >
                  {dayOptions.map((d) => (
                    <option key={d.value} value={d.value}>{d.label}</option>
                  ))}
                </select>
              </div>
              <div className="form-row-inline">
                <div className="form-group">
                  <label>Start Time:</label>
                  <input
                    type="time"
                    value={newClass.startTime}
                    onChange={(e) => setNewClass({ ...newClass, startTime: e.target.value })}
                  />
                </div>
                <div className="form-group">
                  <label>End Time:</label>
                  <input
                    type="time"
                    value={newClass.endTime}
                    onChange={(e) => setNewClass({ ...newClass, endTime: e.target.value })}
                  />
                </div>
              </div>
              <div className="form-group">
                <label>Courses/Strands:</label>
                <div className="course-checkboxes">
                  {courses.map((course) => {
                    const code = course.code || course;
                    const level = course.level || "unknown";
                    const isDisabled = !newClass.teacherDepartment ||
                      (newClass.teacherDepartment !== "both" && newClass.teacherDepartment !== level);
                    return (
                      <label key={code} className="course-checkbox">
                        <input
                          type="checkbox"
                          checked={(newClass.courses || []).includes(code)}
                          onChange={(e) => handleCourseChange(code, e.target.checked)}
                          disabled={isDisabled}
                        />
                        {code}
                      </label>
                    );
                  })}
                </div>
              </div>
            </form>
            <div className="modal-actions">
              <button className="btn-primary" onClick={handleAddClass}>Add Class</button>
              <button className="btn-secondary" onClick={closeAddClassModal}>Cancel</button>
            </div>
          </div>
        </div>
      )}

      <div className="list-container">
        <table className="classes-table">
          <thead>
            <tr>
              <th>Class ID</th>
              <th>Class Name</th>
              <th>Teacher</th>
              <th>Courses/Strands</th>
              <th>Students</th>
              <th>Room</th>
              <th>Schedule</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filteredClasses.map((cls) => (
              <tr key={cls.id} onClick={() => handleView(cls)} style={{ cursor: "pointer" }}>
                <td>{cls.id}</td>
                <td>{cls.name}</td>
                <td>{cls.teacher}</td>
                <td>{(cls.courses || []).join(', ')}</td>
                <td>{cls.students}</td>
                <td>{cls.room}</td>
                <td>{formatSchedule(cls)}</td>
                <td>
                  <button className="action-btn add-student" onClick={(e) => { e.stopPropagation(); handleOpenAddStudent(cls); }} style={{ backgroundColor: "#10b981", color: "white" }}>Add Student</button>
                  <button className="action-btn edit" onClick={(e) => { e.stopPropagation(); handleEdit(cls); }}>✏️ Edit</button>
                  <button className="action-btn delete" onClick={(e) => { e.stopPropagation(); handleDelete(cls); }}>🗑️ Delete</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {isAddStudentModalOpen && classForStudentAdd && (
        <div className="modal-overlay" onClick={closeModals}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h2>Add Student to {classForStudentAdd.name}</h2>
            <div className="form-group">
              <label>Search student</label>
              <input
                type="text"
                placeholder="Search by name, ID, or course..."
                value={studentSearchTerm}
                onChange={(e) => setStudentSearchTerm(e.target.value)}
              />
            </div>

            {addStudentFeedback && (
              <div className="detail-row">
                <strong>{addStudentFeedback}</strong>
              </div>
            )}

            <div className="classes-list">
              {studentsLoading ? (
                <div className="loading">Loading students...</div>
              ) : filteredStudentsForAdd.length === 0 ? (
                <div className="loading">No students found.</div>
              ) : (
                filteredStudentsForAdd.map((student) => {
                  const alreadyEnrolled = enrolledStudentIds.has(student.id);
                  return (
                    <div key={student.id} className="class-item">
                      <div><strong>{student.name || "Unnamed Student"}</strong> ({student.id})</div>
                      <div>{student.course} | Year {student.year}</div>
                      <div className="modal-actions" style={{ marginTop: "8px" }}>
                        <button
                          className="btn-primary"
                          disabled={alreadyEnrolled || addStudentLoading}
                          onClick={() => handleAddStudentToClass(student)}
                        >
                          {alreadyEnrolled ? "Already Enrolled" : "Add Student"}
                        </button>
                      </div>
                    </div>
                  );
                })
              )}
            </div>

            <div className="modal-actions">
              <button className="btn-secondary" onClick={closeModals}>Close</button>
            </div>
          </div>
        </div>
      )}

      {isViewModalOpen && selectedClass && (
        <div className="modal-overlay" onClick={closeModals}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h2>Class Details</h2>
            <div className="detail-row">
              <strong>ID:</strong> {selectedClass.id}
            </div>
            <div className="detail-row">
              <strong>Class Name:</strong> {selectedClass.name}
            </div>
            <div className="detail-row">
              <strong>Room:</strong> {selectedClass.room}
            </div>
            <div className="detail-row">
              <strong>Teacher:</strong> {selectedClass.teacher}
            </div>
            <div className="detail-row">
              <strong>Courses/Strands:</strong> {(selectedClass.courses || []).join(', ')}
            </div>
            <div className="detail-row">
              <strong>Students:</strong> {selectedClass.students}
            </div>
            <div className="detail-row">
              <strong>Schedule:</strong> {formatSchedule(selectedClass)}
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
            <h2>Edit Class</h2>
            <form className="edit-form">
              <div className="form-group">
                <label>Class Name:</label>
                <input
                  type="text"
                  value={editForm.name || ''}
                  onChange={(e) => setEditForm({ ...editForm, name: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label>Teacher:</label>
                <div className="teacher-input-container">
                  <input
                    type="text"
                    placeholder="Teacher Name or ID"
                    value={editForm.teacher || ''}
                    onChange={handleEditTeacherInputChange}
                    onBlur={() => setTimeout(() => setShowEditTeacherSuggestions(false), 200)}
                    onFocus={() => {
                      if (editForm.teacher && editForm.teacher.length > 0 && editTeacherSuggestions.length > 0) {
                        setShowEditTeacherSuggestions(true);
                      }
                    }}
                  />
                  {showEditTeacherSuggestions && editTeacherSuggestions.length > 0 && (
                    <div className="teacher-suggestions">
                      {editTeacherSuggestions.map((teacher) => (
                        <div
                          key={teacher.id}
                          className="teacher-suggestion-item"
                          onClick={() => handleEditTeacherSelect(teacher)}
                        >
                          <div className="teacher-name">{teacher.name}</div>
                          <div className="teacher-id">{teacher.id}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
              <div className="form-group">
                <label>Room:</label>
                <input
                  type="text"
                  value={editForm.room || ''}
                  onChange={(e) => setEditForm({ ...editForm, room: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label>Day:</label>
                <select
                  value={editForm.day || 'Mon'}
                  onChange={(e) => setEditForm({ ...editForm, day: e.target.value })}
                  className="day-select"
                >
                  {dayOptions.map((d) => (
                    <option key={d.value} value={d.value}>{d.label}</option>
                  ))}
                </select>
              </div>
              <div className="form-group">
                <label>Start Time:</label>
                <input
                  type="time"
                  value={editForm.startTime || '08:00'}
                  onChange={(e) => setEditForm({ ...editForm, startTime: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label>End Time:</label>
                <input
                  type="time"
                  value={editForm.endTime || '09:00'}
                  onChange={(e) => setEditForm({ ...editForm, endTime: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label>Courses:</label>
                <div className="course-checkboxes">
                  {courses.map((course) => {
                    const code = course.code || course;
                    const level = course.level || "unknown";
                    const isDisabled = !editForm.teacherDepartment ||
                      (editForm.teacherDepartment !== "both" && editForm.teacherDepartment !== level);
                    return (
                      <label key={code} className="course-checkbox">
                        <input
                          type="checkbox"
                          checked={(editForm.courses || []).includes(code)}
                          onChange={(e) => handleCourseChange(code, e.target.checked, true)}
                          disabled={isDisabled}
                        />
                        {code}
                      </label>
                    );
                  })}
                </div>
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
