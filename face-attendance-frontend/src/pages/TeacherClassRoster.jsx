import React, { useState, useEffect, useCallback } from "react";
import "../styles/TeacherDashboard.css";

export default function TeacherClassRoster({ onBack, userInfo }) {
  const [teacherClasses, setTeacherClasses] = useState([]);
  const [loadingClasses, setLoadingClasses] = useState(true);
  const [dayFilter, setDayFilter] = useState("");
  const [courseFilter, setCourseFilter] = useState("");
  const [teacherData, setTeacherData] = useState(null);
  const [selectedClass, setSelectedClass] = useState(null);
  const [enrolledStudents, setEnrolledStudents] = useState([]);
  const [loadingStudents, setLoadingStudents] = useState(false);
  const [courseFilterStudents, setCourseFilterStudents] = useState("");
  const [isStudentModalOpen, setIsStudentModalOpen] = useState(false);
  const [allStudents, setAllStudents] = useState([]);
  const [loadingAllStudents, setLoadingAllStudents] = useState(false);
  const [studentSearchTerm, setStudentSearchTerm] = useState("");
  const [addStudentLoading, setAddStudentLoading] = useState(false);
  const [addStudentFeedback, setAddStudentFeedback] = useState("");

  const dayOptions = [
    { value: "", label: "All Days" },
    { value: "Monday", label: "Monday" },
    { value: "Tuesday", label: "Tuesday" },
    { value: "Wednesday", label: "Wednesday" },
    { value: "Thursday", label: "Thursday" },
    { value: "Friday", label: "Friday" },
    { value: "Saturday", label: "Saturday" },
  ];

  const dayAbbrevToFull = {
    'M': 'Monday',
    'T': 'Tuesday',
    'W': 'Wednesday',
    'H': 'Thursday',
    'F': 'Friday',
    'S': 'Saturday',
    'Mon': 'Monday',
    'Tue': 'Tuesday',
    'Wed': 'Wednesday',
    'Thu': 'Thursday',
    'Fri': 'Friday',
    'Sat': 'Saturday',
  };

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:8000";

  const fetchTeacherData = useCallback(async () => {
    if (!userInfo?.user_id) return;

    try {
      const response = await fetch(`${BACKEND_URL}/teachers/${userInfo.user_id}`);
      if (response.ok) {
        const data = await response.json();
        setTeacherData(data);
      }
    } catch (error) {
      console.error("Error fetching teacher data:", error);
    }
  }, [userInfo?.user_id, BACKEND_URL]);

  const fetchClasses = useCallback(async () => {
    if (!userInfo?.user_id) {
      console.error("No teacher ID available");
      return;
    }

    setLoadingClasses(true);
    try {
      const response = await fetch(`${BACKEND_URL}/classes/teacher/${userInfo.user_id}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch classes: ${response.status}`);
      }
      const data = await response.json();
      setTeacherClasses(data.classes || []);
    } catch (error) {
      console.error("Error fetching classes:", error);
      setTeacherClasses([]);
    } finally {
      setLoadingClasses(false);
    }
  }, [userInfo?.user_id, BACKEND_URL]);

  const fetchEnrolledStudents = useCallback(async (classId) => {
    setLoadingStudents(true);
    try {
      // First get the class details to get enrolled student IDs
      const classResponse = await fetch(`${BACKEND_URL}/classes/${classId}`);
      if (!classResponse.ok) {
        throw new Error(`Failed to fetch class details: ${classResponse.status}`);
      }
      const classData = await classResponse.json();

      // Get student details for enrolled students
      const enrolledStudentIds = classData.enrolled_students || [];
      const studentPromises = enrolledStudentIds.map(studentId =>
        fetch(`${BACKEND_URL}/students/${studentId}`).then(res => res.ok ? res.json() : null)
      );

      const studentResults = await Promise.all(studentPromises);
      const validStudents = studentResults.filter(student => student !== null);

      setEnrolledStudents(validStudents);
    } catch (error) {
      console.error("Error fetching enrolled students:", error);
      setEnrolledStudents([]);
    } finally {
      setLoadingStudents(false);
    }
  }, [BACKEND_URL]);

  const getAuthHeaders = () => {
    const token = localStorage.getItem("token");
    return token ? { Authorization: `Bearer ${token}` } : {};
  };

  const fetchAllStudents = useCallback(async () => {
    try {
      setLoadingAllStudents(true);
      const response = await fetch(`${BACKEND_URL}/students`);
      if (!response.ok) {
        throw new Error(`Failed to fetch students: ${response.status}`);
      }
      const data = await response.json();
      const transformedStudents = (data.students || []).map((student) => ({
        id: student.student_id,
        name: `${student.first_name || ""} ${student.last_name || ""}`.trim(),
        course: student.course || "",
        year: student.year || ""
      }));
      setAllStudents(transformedStudents);
    } catch (error) {
      console.error("Error fetching students:", error);
      setAllStudents([]);
    } finally {
      setLoadingAllStudents(false);
    }
  }, [BACKEND_URL]);

  const handleClassClick = (cls) => {
    setSelectedClass(cls);
    fetchEnrolledStudents(cls._id);
    fetchAllStudents();
    setStudentSearchTerm("");
    setAddStudentFeedback("");
    setIsStudentModalOpen(true);
  };

  const handleBackToClasses = () => {
    setSelectedClass(null);
    setEnrolledStudents([]);
    setCourseFilterStudents("");
  };

  const closeStudentModal = () => {
    setIsStudentModalOpen(false);
    setAddStudentFeedback("");
    handleBackToClasses();
  };

  useEffect(() => {
    fetchClasses();
    fetchTeacherData();
  }, [fetchClasses, fetchTeacherData]);

  const handleAddStudentToClass = async (studentId) => {
    if (!selectedClass || !studentId) return;

    try {
      setAddStudentLoading(true);
      setAddStudentFeedback("");
      const response = await fetch(`${BACKEND_URL}/classes/${selectedClass._id}/enroll-student`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...getAuthHeaders()
        },
        body: JSON.stringify({ student_id: studentId })
      });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.detail || result.error || "Failed to add student");
      }
      setAddStudentFeedback(result.message || "Student added successfully");
      await fetchEnrolledStudents(selectedClass._id);
      await fetchClasses();
    } catch (error) {
      setAddStudentFeedback(`Error: ${error.message}`);
    } finally {
      setAddStudentLoading(false);
    }
  };

  const parseSchedule = (schedule) => {
    // Parse schedule string like "MWF 9:00-10:00" into days, startTime, endTime
    const parts = schedule.split(' ');
    if (parts.length >= 2) {
      const days = parts[0];
      const timeRange = parts.slice(1).join(' ');
      const [startTime, endTime] = timeRange.split('-');
      return { days, startTime, endTime };
    }
    return { days: schedule, startTime: 'N/A', endTime: 'N/A' };
  };

  const getFullDays = (days) => {
    if (days.length === 3 && /^[MTWFSH]+$/.test(days)) {
      return days.split('').map(d => dayAbbrevToFull[d]).filter(Boolean);
    } else if (dayAbbrevToFull[days]) {
      return [dayAbbrevToFull[days]];
    }
    return [];
  };

  // Filter classes based on selected filters
  const filteredClasses = teacherClasses.filter((cls) => {
    const { days } = parseSchedule(cls.schedule);
    const fullDays = getFullDays(days);
    const matchesDay = !dayFilter || fullDays.includes(dayFilter);
    const matchesCourse = !courseFilter || cls.class_name.toLowerCase().includes(courseFilter.toLowerCase());
    return matchesDay && matchesCourse;
  });

  const enrolledStudentIdSet = new Set(enrolledStudents.map((student) => student.student_id));
  const filteredStudentsForAdd = allStudents.filter((student) => {
    const query = studentSearchTerm.trim().toLowerCase();
    if (!query) return true;
    return (
      student.name.toLowerCase().includes(query) ||
      student.id.toLowerCase().includes(query) ||
      student.course.toLowerCase().includes(query)
    );
  });

  return (
    <div className="teacher-dashboard-wrapper">
      <main className="teacher-main-content">
        <div className="content-header">
          <h1>My Subjects</h1>
          <p>View your assigned subjects.</p>
        </div>

        {/* Filters */}
        <div className="filters-section" style={{ marginBottom: '20px', display: 'flex', gap: '15px', flexWrap: 'wrap' }}>
          <div className="filter-group">
            <label htmlFor="day-filter" style={{ color: 'white', marginRight: '10px' }}>Filter by Day:</label>
            <select
              id="day-filter"
              value={dayFilter}
              onChange={(e) => setDayFilter(e.target.value)}
              style={{
                padding: '8px 12px',
                borderRadius: '8px',
                border: '1px solid rgba(255,255,255,0.2)',
                background: 'rgba(255, 255, 255, 0.05)',
                color: 'white'
              }}
            >
              {dayOptions.map(option => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </div>
          {teacherData && teacherData.department && (
            <div className="filter-group">
              <label htmlFor="course-filter" style={{ color: 'white', marginRight: '10px' }}>
                Filter by {teacherData.department.toLowerCase().includes('college') ? 'Course' : teacherData.department.toLowerCase().includes('shs') ? 'Strand' : 'Subject'}:
              </label>
              <input
                id="course-filter"
                type="text"
                value={courseFilter}
                onChange={(e) => setCourseFilter(e.target.value)}
                placeholder={`Search ${teacherData.department.toLowerCase().includes('college') ? 'courses' : teacherData.department.toLowerCase().includes('shs') ? 'strands' : 'subjects'}...`}
                style={{
                  padding: '8px 12px',
                  borderRadius: '8px',
                  border: '1px solid rgba(255,255,255,0.2)',
                  background: 'rgba(255,255,255,0.1)',
                  color: 'white'
                }}
              />
            </div>
          )}
        </div>

        <div className="teacher-class-roster">
          {loadingClasses ? (
            <div className="loading-classes">Loading classes...</div>
          ) : teacherClasses.length === 0 ? (
            <div className="no-classes">No classes assigned to you yet.</div>
          ) : (
            <div className="class-roster-cards" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '20px' }}>
              {filteredClasses.map((cls) => {
                const { days, startTime, endTime } = parseSchedule(cls.schedule);
                return (
                  <div key={cls._id} className="class-card" style={{
                    background: 'rgba(255, 255, 255, 0.05)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '12px',
                    padding: '20px',
                    backdropFilter: 'blur(10px)',
                    transition: 'transform 0.2s',
                    cursor: 'pointer'
                  }}
                  onClick={() => handleClassClick(cls)}
                  onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-2px)'}
                  onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0)'}
                  >
                    <div className="card-header" style={{ marginBottom: '15px' }}>
                      <h3 style={{ color: '#10b981', margin: '0 0 5px 0', fontSize: '18px', fontWeight: '600' }}>
                        {cls.class_name}
                      </h3>
                      <p style={{ color: 'rgba(255,255,255,0.7)', margin: '0', fontSize: '14px' }}>
                        {cls.class_code}
                      </p>
                      {cls.courses && cls.courses.length > 0 && (
                        <div style={{ marginTop: '10px' }}>
                          <p style={{ color: 'rgba(255,255,255,0.8)', margin: '0 0 5px 0', fontSize: '12px', fontWeight: '500' }}>
                            {teacherData?.department?.toLowerCase().includes('college') ? 'Courses:' : teacherData?.department?.toLowerCase().includes('shs') ? 'Strands:' : 'Courses/Strands:'}
                          </p>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '5px' }}>
                            {cls.courses.map((course, index) => (
                              <span key={index} style={{
                                background: 'rgba(16, 185, 129, 0.2)',
                                color: '#10b981',
                                padding: '2px 8px',
                                borderRadius: '12px',
                                fontSize: '11px',
                                fontWeight: '500'
                              }}>
                                {course}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                    <div className="card-details" style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                      <div className="detail-item" style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'rgba(255,255,255,0.6)' }}>Days:</span>
                        <span style={{ color: 'white', fontWeight: '500' }}>{days}</span>
                      </div>
                      <div className="detail-item" style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'rgba(255,255,255,0.6)' }}>Room:</span>
                        <span style={{ color: 'white', fontWeight: '500' }}>{cls.room}</span>
                      </div>
                      <div className="detail-item" style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'rgba(255,255,255,0.6)' }}>Time:</span>
                        <span style={{ color: 'white', fontWeight: '500' }}>{startTime} - {endTime}</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>



        <div className="manage-classes-form-buttons">
          <button type="button" className="manage-classes-secondary" onClick={onBack} style={{ padding: '8px 16px', fontSize: '13px', marginTop: '20px' }}>
            Back to Dashboard
          </button>
        </div>
      </main>

      {/* Student Modal */}
      {isStudentModalOpen && selectedClass && (
        <div className="modal-overlay" onClick={closeStudentModal}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="content-header" style={{ marginBottom: '20px' }}>
              <h2>Students in {selectedClass.class_name}</h2>
              <p>View enrolled students filtered by courses/strands.</p>
            </div>

            <div style={{
              marginBottom: '20px',
              padding: '12px',
              border: '1px solid rgba(255,255,255,0.15)',
              borderRadius: '8px',
              background: 'rgba(255,255,255,0.03)'
            }}>
              <h3 style={{ color: '#10b981', marginTop: 0, marginBottom: '10px', fontSize: '16px' }}>Add Student</h3>
              <input
                type="text"
                value={studentSearchTerm}
                onChange={(e) => setStudentSearchTerm(e.target.value)}
                placeholder="Search by name, ID, or course..."
                style={{
                  width: '100%',
                  marginBottom: '10px',
                  padding: '8px 12px',
                  borderRadius: '8px',
                  border: '1px solid rgba(255,255,255,0.2)',
                  background: 'rgba(255,255,255,0.08)',
                  color: 'white'
                }}
              />

              {addStudentFeedback && (
                <p style={{ margin: '0 0 10px 0', color: addStudentFeedback.startsWith('Error:') ? '#f87171' : '#34d399' }}>
                  {addStudentFeedback}
                </p>
              )}

              <div style={{ maxHeight: '220px', overflowY: 'auto', display: 'grid', gap: '8px' }}>
                {loadingAllStudents ? (
                  <div className="loading-classes">Loading students...</div>
                ) : filteredStudentsForAdd.length === 0 ? (
                  <div className="no-classes">No students found.</div>
                ) : (
                  filteredStudentsForAdd.map((student) => {
                    const alreadyEnrolled = enrolledStudentIdSet.has(student.id);
                    return (
                      <div key={student.id} style={{
                        border: '1px solid rgba(255,255,255,0.1)',
                        borderRadius: '8px',
                        padding: '8px 10px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        gap: '10px'
                      }}>
                        <div>
                          <div style={{ color: 'white', fontSize: '13px', fontWeight: '600' }}>{student.name || 'Unnamed Student'} ({student.id})</div>
                          <div style={{ color: 'rgba(255,255,255,0.7)', fontSize: '12px' }}>{student.course} | Year {student.year}</div>
                        </div>
                        <button
                          type="button"
                          className="action-btn primary"
                          disabled={alreadyEnrolled || addStudentLoading}
                          onClick={() => handleAddStudentToClass(student.id)}
                        >
                          {alreadyEnrolled ? 'Already Enrolled' : 'Add Student'}
                        </button>
                      </div>
                    );
                  })
                )}
              </div>
            </div>

            {/* Course/Strand Filter for Students */}
            {selectedClass.courses && selectedClass.courses.length > 0 && (
              <div className="filters-section" style={{ marginBottom: '20px' }}>
                <div className="filter-group">
                  <label htmlFor="course-filter-students" style={{ color: 'white', marginRight: '10px' }}>
                    Filter by {teacherData?.department?.toLowerCase().includes('college') ? 'Course' : teacherData?.department?.toLowerCase().includes('shs') ? 'Strand' : 'Course/Strand'}:
                  </label>
                  <select
                    id="course-filter-students"
                    value={courseFilterStudents}
                    onChange={(e) => setCourseFilterStudents(e.target.value)}
                    style={{
                      padding: '8px 12px',
                      borderRadius: '8px',
                      border: '1px solid rgba(255,255,255,0.2)',
                      background: 'rgba(255, 255, 255, 0.05)',
                      color: 'white'
                    }}
                  >
                    <option value="">All {teacherData?.department?.toLowerCase().includes('college') ? 'Courses' : teacherData?.department?.toLowerCase().includes('shs') ? 'Strands' : 'Courses/Strands'}</option>
                    {selectedClass.courses.map((course, index) => (
                      <option key={index} value={course}>{course}</option>
                    ))}
                  </select>
                </div>
              </div>
            )}

            <div className="students-list">
              {loadingStudents ? (
                <div className="loading-classes">Loading students...</div>
              ) : enrolledStudents.length === 0 ? (
                <div className="no-classes">No students enrolled in this class yet.</div>
              ) : (
                <div className="students-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: '15px' }}>
                  {enrolledStudents
                    .filter(student => !courseFilterStudents || (student.course && student.course.trim().toLowerCase().includes(courseFilterStudents.trim().toLowerCase())))
                    .map((student) => (
                      <div key={student.student_id} className="student-card" style={{
                        background: 'rgba(255, 255, 255, 0.05)',
                        border: '1px solid rgba(255, 255, 255, 0.1)',
                        borderRadius: '8px',
                        padding: '15px',
                        backdropFilter: 'blur(10px)'
                      }}>
                        <div className="student-header" style={{ marginBottom: '10px' }}>
                          <h4 style={{ color: '#10b981', margin: '0 0 5px 0', fontSize: '16px', fontWeight: '600' }}>
                            {student.first_name} {student.last_name}
                          </h4>
                          <p style={{ color: 'rgba(255,255,255,0.7)', margin: '0', fontSize: '12px' }}>
                            ID: {student.student_id}
                          </p>
                        </div>
                        <div className="student-details" style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
                          <div className="detail-item" style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span style={{ color: 'rgba(255,255,255,0.6)', fontSize: '12px' }}>
                              {teacherData?.department?.toLowerCase().includes('college') ? 'Course:' : teacherData?.department?.toLowerCase().includes('shs') ? 'Strand:' : 'Course/Strand:'}
                            </span>
                            <span style={{ color: 'white', fontWeight: '500', fontSize: '12px' }}>{student.course}</span>
                          </div>
                          <div className="detail-item" style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span style={{ color: 'rgba(255,255,255,0.6)', fontSize: '12px' }}>Year:</span>
                            <span style={{ color: 'white', fontWeight: '500', fontSize: '12px' }}>{student.year}</span>
                          </div>
                        </div>
                      </div>
                    ))}
                </div>
              )}
            </div>

            <div className="manage-classes-form-buttons" style={{ marginTop: '20px' }}>
              <button type="button" className="manage-classes-secondary" onClick={closeStudentModal} style={{ padding: '8px 16px', fontSize: '13px' }}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
