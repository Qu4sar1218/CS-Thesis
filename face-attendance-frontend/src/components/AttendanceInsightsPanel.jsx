import React, { useState, useEffect } from 'react';
import '../styles/AttendanceInsightsPanel.css';



const AttendanceInsightsPanel = ({ studentId }) => {
  const [insightsData, setInsightsData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchInsights = async () => {
      if (!studentId) {
        setLoading(false);
        return;
      }

      try {
        const insightsResponse = await fetch(`http://127.0.0.1:8000/analytics/student/${studentId}/insights`);

        if (insightsResponse.ok) {
          const data = await insightsResponse.json();
          setInsightsData(data);
        } else {
          setError('Failed to fetch attendance insights');
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

  const { attendance_summary, subject_breakdown } = insightsData;

  return (
    <div className="insights-panel">
      <h2 className="panel-title">Attendance Insights</h2>

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

      {/* Subject-Based Attendance Breakdown */}
      <div className="overview-columns">
        <div className="overview-column full-width">
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
    </div>
  );
};

export default AttendanceInsightsPanel;

