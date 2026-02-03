import React, { useState, useEffect } from 'react';
import '../styles/StudentSchedule.css';

export default function StudentSchedule({ onBack, userInfo }) {
  const [schedule, setSchedule] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:8000";

  useEffect(() => {
    const fetchSchedule = async () => {
      if (!userInfo?.user_id) {
        setError('User information not available');
        setLoading(false);
        return;
      }

      try {
        setLoading(true);
        const response = await fetch(`${BACKEND_URL}/api/student/schedule/${userInfo.user_id}`);

        if (!response.ok) {
          throw new Error(`Failed to fetch schedule: ${response.status}`);
        }

        const data = await response.json();
        setSchedule(data.schedule || []);
      } catch (err) {
        console.error('Error fetching schedule:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchSchedule();
  }, [userInfo?.user_id, BACKEND_URL]);

  // Group schedule by day
  const groupedSchedule = schedule.reduce((acc, item) => {
    if (!acc[item.day]) {
      acc[item.day] = [];
    }
    acc[item.day].push(item);
    return acc;
  }, {});

  // Sort days in order: Monday to Sunday
  const dayOrder = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
  const sortedDays = Object.keys(groupedSchedule).sort((a, b) => {
    const indexA = dayOrder.indexOf(a);
    const indexB = dayOrder.indexOf(b);
    return (indexA === -1 ? 7 : indexA) - (indexB === -1 ? 7 : indexB);
  });

  if (loading) {
    return (
      <div className="student-schedule-container">
        <div className="schedule-header">
          <button className="back-button" onClick={onBack}>
            ← Back
          </button>
          <h1>My Subject Schedule</h1>
        </div>
        <div className="loading-state">
          <div className="loading-spinner"></div>
          <p>Loading your schedule...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="student-schedule-container">
        <div className="schedule-header">
          <button className="back-button" onClick={onBack}>
            ← Back
          </button>
          <h1>My Subject Schedule</h1>
        </div>
        <div className="error-state">
          <p>❌ Error loading schedule: {error}</p>
          <button className="retry-button" onClick={() => window.location.reload()}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="student-schedule-container">
      <div className="schedule-header">
        <button className="back-button" onClick={onBack}>
          ← Back
        </button>
        <h1>My Subject Schedule</h1>
      </div>

      {schedule.length === 0 ? (
        <div className="empty-state">
          <p>📅 No subjects assigned to your schedule yet.</p>
          <p>Please contact your administrator if you believe this is an error.</p>
        </div>
      ) : (
        <div className="schedule-content">
          {sortedDays.map(day => (
            <div key={day} className="day-section">
              <h2 className="day-title">{day}</h2>
              <div className="subjects-grid">
                {groupedSchedule[day]
                  .sort((a, b) => a.start_time.localeCompare(b.start_time))
                  .map((subject, index) => (
                    <div key={`${subject.subject_code}-${index}`} className="subject-card">
                      <div className="subject-header">
                        <h3 className="subject-name">{subject.subject_name}</h3>
                        <span className="subject-code">{subject.subject_code}</span>
                      </div>
                      <div className="subject-details">
                        <div className="subject-time">
                          <span className="time-icon">🕐</span>
                          {subject.start_time} - {subject.end_time}
                        </div>
                        <div className="subject-instructor">
                          <span className="instructor-icon">👨‍🏫</span>
                          {subject.instructor}
                        </div>
                        <div className="subject-room">
                          <span className="room-icon">🏫</span>
                          {subject.room}
                        </div>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
