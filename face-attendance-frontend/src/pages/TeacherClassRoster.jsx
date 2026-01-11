import React, { useState, useEffect, useCallback } from "react";
import "../styles/TeacherDashboard.css";

export default function TeacherClassRoster({ onBack, userInfo }) {
  const [teacherClasses, setTeacherClasses] = useState([]);
  const [loadingClasses, setLoadingClasses] = useState(true);
  const [dayFilter, setDayFilter] = useState("");
  const [courseFilter, setCourseFilter] = useState("");
  const [teacherData, setTeacherData] = useState(null);

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

  useEffect(() => {
    fetchClasses();
    fetchTeacherData();
  }, [fetchClasses, fetchTeacherData]);

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
    </div>
  );
}
