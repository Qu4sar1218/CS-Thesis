import React, { useState, useEffect } from 'react';
import '../styles/AttendanceInsightsPanel.css';

const normalizeStatus = (status) => {
  const normalized = (status || '').toString().trim().toLowerCase();
  if (normalized === 'late') return 'Late';
  if (normalized === 'absent') return 'Absent';
  return 'Present';
};

const AttendanceInsightsPanel = ({ studentId }) => {
  const [insightsData, setInsightsData] = useState(null);
  const [eventAttendance, setEventAttendance] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchInsights = async () => {
      if (!studentId) {
        setLoading(false);
        return;
      }

      try {
        const [insightsResponse, attendanceResponse] = await Promise.all([
          fetch(`http://127.0.0.1:8000/analytics/student/${studentId}/insights`),
          fetch(`http://127.0.0.1:8000/analytics/student/${studentId}`)
        ]);

        if (insightsResponse.ok) {
          const data = await insightsResponse.json();
          setInsightsData(data);
        } else {
          setError('Failed to fetch attendance insights');
        }

        if (attendanceResponse.ok) {
          const attendancePayload = await attendanceResponse.json();
          const rawRecords = attendancePayload.attendance || [];
          const eventsOnly = rawRecords
            .filter((record) => record.mode === 'events' || Boolean(record.event_id))
            .map((record) => ({
              date: record.date || '',
              event: record.event_name || record.subject || 'Unknown Event',
              status: normalizeStatus(record.status),
              time: record.check_in_time || record.time || 'N/A'
            }))
            .sort((a, b) => {
              const aDate = `${a.date || ''} ${a.time || '00:00:00'}`;
              const bDate = `${b.date || ''} ${b.time || '00:00:00'}`;
              return bDate.localeCompare(aDate);
            });
          setEventAttendance(eventsOnly);
        } else {
          setEventAttendance([]);
        }
      } catch (err) {
        setError('Error fetching attendance insights');
        console.error('Error:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchInsights();
  }, [studentId]);

  if (loading) {
    return (
      <div className="insights-panel">
        <div className="loading">
          <div className="spinner"></div>
          <p>Loading attendance insights...</p>
        </div>
      </div>
    );
  }

  if (error || !insightsData) {
    return (
      <div className="insights-panel">
        <div className="error-state">
          <p>{error || 'No data available'}</p>
        </div>
      </div>
    );
  }

  const { attendance_summary, subject_breakdown, face_recognition_logs, smart_feedback } = insightsData;
  const eventTotal = eventAttendance.length;
  const eventPresent = eventAttendance.filter((a) => a.status === 'Present').length;
  const eventLate = eventAttendance.filter((a) => a.status === 'Late').length;
  const eventAbsent = eventAttendance.filter((a) => a.status === 'Absent').length;
  const eventRate = eventTotal > 0 ? (((eventPresent + eventLate) / eventTotal) * 100).toFixed(1) : '0.0';

  return (
    <div className="insights-panel">
      <h2 className="panel-title">Attendance Insights & Identity Activity Panel</h2>

      {/* Attendance Summary Cards */}
      <div className="summary-cards">
        <div className="summary-card">
          <div className="card-icon">📚</div>
          <div className="card-content">
            <span className="card-label">Total Sessions</span>
            <span className="card-value">{attendance_summary.total_sessions}</span>
          </div>
        </div>
        <div className="summary-card">
          <div className="card-icon">❌</div>
          <div className="card-content">
            <span className="card-label">Total Absences</span>
            <span className="card-value">{attendance_summary.absent_count}</span>
          </div>
        </div>
        <div className="summary-card">
          <div className="card-icon">📊</div>
          <div className="card-content">
            <span className="card-label">Attendance Rate</span>
            <span className="card-value">{attendance_summary.attendance_percentage}%</span>
          </div>
        </div>
        <div className={`summary-card status-${attendance_summary.status.toLowerCase().replace(' ', '-')}`}>
          <div className="card-icon">⚡</div>
          <div className="card-content">
            <span className="card-label">Status</span>
            <span className="card-value">{attendance_summary.status}</span>
          </div>
        </div>
      </div>

      {/* Overview Columns - Side by Side */}
      <div className="overview-columns">
        {/* Event Attendance Overview */}
        <div className="overview-column">
          <div className="section-header">
            <span className="section-icon">🎉</span>
            <h3 className="section-title">Events Overview</h3>
          </div>
          <div className="summary-cards">
            <div className="summary-card">
              <div className="card-icon">📅</div>
              <div className="card-content">
                <span className="card-label">Total Events</span>
                <span className="card-value">{eventTotal}</span>
              </div>
            </div>
            <div className="summary-card">
              <div className="card-icon">✅</div>
              <div className="card-content">
                <span className="card-label">Present</span>
                <span className="card-value">{eventPresent}</span>
              </div>
            </div>
            <div className="summary-card">
              <div className="card-icon">⏰</div>
              <div className="card-content">
                <span className="card-label">Late</span>
                <span className="card-value">{eventLate}</span>
              </div>
            </div>
            <div className="summary-card">
              <div className="card-icon">❌</div>
              <div className="card-content">
                <span className="card-label">Absent</span>
                <span className="card-value">{eventAbsent}</span>
              </div>
            </div>
            <div className="summary-card">
              <div className="card-icon">📈</div>
              <div className="card-content">
                <span className="card-label">Attendance Rate</span>
                <span className="card-value">{eventRate}%</span>
              </div>
            </div>
          </div>
        </div>

        {/* Subject-Based Attendance Breakdown */}
        <div className="overview-column">
          <div className="section-header">
            <span className="section-icon">📚</span>
            <h3 className="section-title">Subject Attendance</h3>
          </div>
          {subject_breakdown.length > 0 ? (
            <div className="subject-grid">
              {subject_breakdown.map((subject, index) => (
                <div key={index} className="subject-card">
                  <div className="subject-header">
                    <h4 className="subject-name">{subject.subject}</h4>
                    <span className="subject-percentage">{subject.attendance_percentage}%</span>
                  </div>
                  <div className="subject-stats">
                    <div className="stat-item">
                      <span className="stat-label">Present:</span>
                      <span className="stat-value present">{subject.present_count}</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Absent:</span>
                      <span className="stat-value absent">{subject.absent_count}</span>
                    </div>
                  </div>
                  <div className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{ width: `${subject.attendance_percentage}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-state-icon">📭</div>
              <p>No subject data available</p>
            </div>
          )}
        </div>
      </div>

      {/* Recent Event Attendance */}
      <div className="bottom-section">
        <div className="section-header">
          <span className="section-icon">📋</span>
          <h3 className="section-title">Recent Event Attendance</h3>
        </div>
        {eventAttendance.length > 0 ? (
          <div className="event-attendance-table">
            <div className="event-table-header">
              <span>Date</span>
              <span>Time</span>
              <span>Event</span>
              <span>Status</span>
            </div>
            {eventAttendance.slice(0, 5).map((record, index) => (
              <div key={index} className="event-table-row">
                <span>{record.date}</span>
                <span>{record.time}</span>
                <span>{record.event}</span>
                <span className={`result-badge ${record.status.toLowerCase()}`}>
                  {record.status}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-state">
            <div className="empty-state-icon">📭</div>
            <p>No event attendance records found</p>
          </div>
        )}
      </div>

      {/* Face Recognition Activity Log */}
      <div className="bottom-section">
        <div className="section-header">
          <span className="section-icon">👤</span>
          <h3 className="section-title">Face Recognition Activity</h3>
        </div>
        {face_recognition_logs.length > 0 ? (
          <div className="activity-table">
            <div className="table-header">
              <span>Date</span>
              <span>Time</span>
              <span>Subject</span>
              <span>Recognition Result</span>
            </div>
            {face_recognition_logs.map((log, index) => (
              <div key={index} className="table-row">
                <span>{log.date}</span>
                <span>{log.time}</span>
                <span>{log.subject}</span>
                <span className={`result-badge ${log.result.toLowerCase()}`}>
                  {log.result}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-state">
            <div className="empty-state-icon">📭</div>
            <p>No recognition activity found</p>
          </div>
        )}
      </div>

      {/* Smart Attendance Feedback */}
      <div className="bottom-section">
        <div className="section-header">
          <span className="section-icon">💡</span>
          <h3 className="section-title">Attendance Feedback</h3>
        </div>
        <div className="feedback-card">
          <div className="feedback-icon">
            {attendance_summary.status === 'Good Standing' ? '🎉' :
             attendance_summary.status === 'Warning' ? '⚠️' : '🚨'}
          </div>
          <p className="feedback-message">{smart_feedback}</p>
        </div>
      </div>
    </div>
  );
};

export default AttendanceInsightsPanel;
