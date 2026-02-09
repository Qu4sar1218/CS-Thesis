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
    classCode: "",
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

  const [selectedDay, setSelectedDay] = useState(null);
  const [selectedCourse, setSelectedCourse] = useState(null);

  const [selectedClass, setSelectedClass] = useState(null);
  const [isViewModalOpen, setIsViewModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [editForm, setEditForm] = useState({});

  const fetchClasses = useCallback(async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/classes/`);
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

  // Fetch classes when component mounts and when teachers or fetchClasses change
  useEffect(() => {
    fetchClasses();
  }, [teachers, fetchClasses]);

  const fetchTeachers = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/teachers/`);
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
      const response = await fetch(`${API_BASE_URL}/classes/courses`);
      if (!response.ok) {
        throw new Error('Failed to fetch courses');
      }
      const data = await response.json();
      setCourses(data.courses);
    } catch (err) {
      console.error('Error fetching courses:', err);
    }
  };

  const dayOptions = [
    { label: "Monday", value: "Mon" },
    { label: "Tuesday", value: "Tue" },
    { label: "Wednesday", value: "Wed" },
    { label: "Thursday", value: "Thu" },
    { label: "Friday", value: "Fri" },
    { label: "Saturday", value: "Sat" },

  ];

  const handleAddClass = async () => {
    if (!newClass.classCode || !newClass.name || !newClass.teacher || !newClass.room || !newClass.day || !newClass.startTime || !newClass.endTime) {
      setError("Please fill in all required fields");
      return;
    }

    try {
      const classData = {
        class_code: newClass.classCode,
        class_name: newClass.name,
        teacher_id: newClass.teacherId,
        schedule: `${newClass.day} ${newClass.startTime}-${newClass.endTime}`,
        room: newClass.room,
        courses: newClass.courses
      };

      const response = await fetch(`${API_BASE_URL}/classes/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(classData),
      });

      const result = await response.json();

      if (response.ok) {
        setNewClass({ classCode: "", name: "", teacher: "", teacherId: "", teacherDepartment: "", room: "", day: "Mon", startTime: "08:00", endTime: "09:00", courses: [] });
        setIsAddModalOpen(false);
        fetchClasses(); // Refresh the list
        setError(null);
      } else {
        const errorMsg = result.detail || (typeof result.error === 'string' ? result.error : JSON.stringify(result.error));
        setError(`Failed to create class: ${errorMsg}`);
      }
    } catch (error) {
      setError(`Error creating class: ${error.message}`);
    }
  };

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

  const handleEdit = (cls) => {
    setSelectedClass(cls);
    const teacher = teachers.find(t => t.id === cls.teacherId);
    const rawDepartment = teacher ? teacher.department : "both"; // Default to "both" if teacher not found
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

      const response = await fetch(`${API_BASE_URL}/classes/${selectedClass._id}/`, {
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
    const ok = window.confirm(`Are you sure you want to delete class "${cls.name}" (${cls.id})? This action cannot be undone.`);
    if (!ok) return;

    try {
      const response = await fetch(`${API_BASE_URL}/classes/${cls._id}/`, {
        method: "DELETE",
      });

      if (response.ok) {
        fetchClasses(); // Refresh the list
        setError(null);
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
    setIsAddModalOpen(false);
    setSelectedClass(null);
  };

  const filteredClasses = classes.filter(cls => {
    const dayMatch = selectedDay === null || cls.day === selectedDay;
    const courseMatch = selectedCourse === null || (cls.courses || []).includes(selectedCourse);
    return dayMatch && courseMatch;
  });

  const formatSchedule = (c) => `${dayOptions.find(d => d.value === c.day)?.label || c.day} ${c.startTime} - ${c.endTime}`;

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

      <div className="filter-buttons">
        <button
          className={`filter-btn ${selectedDay === null ? 'active' : ''}`}
          onClick={() => setSelectedDay(null)}
        >
          All
        </button>
        <button
          className={`filter-btn ${selectedDay === 'Mon' ? 'active' : ''}`}
          onClick={() => setSelectedDay('Mon')}
        >
          Monday
        </button>
        <button
          className={`filter-btn ${selectedDay === 'Tue' ? 'active' : ''}`}
          onClick={() => setSelectedDay('Tue')}
        >
          Tuesday
        </button>
        <button
          className={`filter-btn ${selectedDay === 'Wed' ? 'active' : ''}`}
          onClick={() => setSelectedDay('Wed')}
        >
          Wednesday
        </button>
        <button
          className={`filter-btn ${selectedDay === 'Thu' ? 'active' : ''}`}
          onClick={() => setSelectedDay('Thu')}
        >
          Thursday
        </button>
        <button
          className={`filter-btn ${selectedDay === 'Fri' ? 'active' : ''}`}
          onClick={() => setSelectedDay('Fri')}
        >
          Friday
        </button>
        <button
          className={`filter-btn ${selectedDay === 'Sat' ? 'active' : ''}`}
          onClick={() => setSelectedDay('Sat')}
        >
          Saturday
        </button>
      </div>

      <div className="course-filter">
        <label htmlFor="course-select">Filter by Course:</label>
        <select
          id="course-select"
          value={selectedCourse || ''}
          onChange={(e) => setSelectedCourse(e.target.value === '' ? null : e.target.value)}
        >
          <option value="">All Courses</option>
          {courses.map((course) => {
            const code = course.code || course;
            return (
              <option key={code} value={code}>
                {code}
              </option>
            );
          })}
        </select>
      </div>

      <div className="add-class-button">
        <button className="primary" onClick={() => setIsAddModalOpen(true)}>Add Class</button>
      </div>

      {isAddModalOpen && (
        <div className="modal-overlay" onClick={closeModals}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="add-class-form">
              <h2>Add New Class</h2>
              <div className="form-row">
                <input
                  type="text"
                  placeholder="Class Code"
                  value={newClass.classCode}
                  onChange={(e) => setNewClass({ ...newClass, classCode: e.target.value })}
                />
                <input
                  type="text"
                  placeholder="Class Name"
                  value={newClass.name}
                  onChange={(e) => setNewClass({ ...newClass, name: e.target.value })}
                />
                <div className="teacher-input-container">
                  <input
                    type="text"
                    placeholder="Teacher Name or ID"
                    value={newClass.teacher}
                    onChange={handleTeacherInputChange}
                    onBlur={() => setTimeout(() => setShowTeacherSuggestions(false), 200)}
                    onFocus={() => {
                      if (newClass.teacher.length === 0) {
                        setTeacherSuggestions(teachers.slice(0, 5));
                        setShowTeacherSuggestions(true);
                      } else if (newClass.teacher.length > 0 && teacherSuggestions.length > 0) {
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
                <input
                  type="text"
                  placeholder="Room"
                  value={newClass.room}
                  onChange={(e) => setNewClass({ ...newClass, room: e.target.value })}
                />
                <select
                  value={newClass.day}
                  onChange={(e) => setNewClass({ ...newClass, day: e.target.value })}
                  className="day-select"
                >
                  {dayOptions.map((d) => (
                    <option key={d.value} value={d.value}>{d.label}</option>
                  ))}
                </select>
                <input
                  type="time"
                  value={newClass.startTime}
                  onChange={(e) => setNewClass({ ...newClass, startTime: e.target.value })}
                />
                <input
                  type="time"
                  value={newClass.endTime}
                  onChange={(e) => setNewClass({ ...newClass, endTime: e.target.value })}
                />
              </div>
              <div className="course-selection">
                <label>Courses:</label>
                <div className="course-checkboxes">
                  {courses.length > 0 ? courses.map((course) => {
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
                  }) : <p>No courses available</p>}
                </div>
              </div>
              <div className="form-actions">
                <button className="primary" onClick={handleAddClass}>Add Class</button>
                <button className="secondary" onClick={() => {
                  setIsAddModalOpen(false);
                  setNewClass({ classCode: "", name: "", teacher: "", teacherId: "", teacherDepartment: "", room: "", day: "Mon", startTime: "08:00", endTime: "09:00", courses: [] });
                  setError(null);
                }}>Cancel</button>
              </div>
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
              <tr key={cls.id}>
                <td>{cls.id}</td>
                <td>{cls.name}</td>
                <td>{cls.teacher}</td>
                <td>{(cls.courses || []).join(', ')}</td>
                <td>{cls.students}</td>
                <td>{cls.room}</td>
                <td>{formatSchedule(cls)}</td>
                <td>
                  <button className="action-btn view" onClick={() => handleView(cls)}>👁️ View</button>
                  <button className="action-btn edit" onClick={() => handleEdit(cls)}>✏️ Edit</button>
                  <button className="action-btn delete" onClick={() => handleDelete(cls)}>🗑️ Delete</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>



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
                <label>Class Code:</label>
                <input
                  type="text"
                  value={editForm.id || ''}
                  onChange={(e) => setEditForm({ ...editForm, id: e.target.value })}
                />
              </div>
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
