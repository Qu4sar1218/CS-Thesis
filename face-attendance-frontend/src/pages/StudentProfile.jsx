import React, { useState, useEffect } from 'react';
import '../styles/StudentProfile.css';
import AttendanceInsightsPanel from '../components/AttendanceInsightsPanel';

export default function StudentProfile({ onBack, userInfo }) {
  const [isEditing, setIsEditing] = useState(false);
  const [attendanceData, setAttendanceData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [profileData, setProfileData] = useState({
    full_name: userInfo?.full_name || '',
    email: userInfo?.email || '',
    student_id: userInfo?.user_id || '',
    course: userInfo?.course || '',
    year: userInfo?.year || '',
  });

  useEffect(() => {
    const fetchAttendance = async () => {
      if (!userInfo?.student_id) {
        setLoading(false);
        return;
      }

      try {
        const response = await fetch(`http://127.0.0.1:8000/analytics/student/${userInfo.student_id}`);
        if (response.ok) {
          const data = await response.json();
          const transformedData = data.attendance.map(record => ({
            date: record.date,
            subject: record.subject || record.class_id || 'Unknown Subject',
            status: record.status,
            time: record.check_in_time || 'N/A',
            room: 'N/A'
          }));
          setAttendanceData(transformedData);
        } else {
          console.error('Failed to fetch attendance data');
          setAttendanceData([]);
        }
      } catch (error) {
        console.error('Error fetching attendance data:', error);
        setAttendanceData([]);
      } finally {
        setLoading(false);
      }
    };

    fetchAttendance();
    const intervalId = setInterval(fetchAttendance, 5000);
    return () => clearInterval(intervalId);
  }, [userInfo]);

  const handleSave = () => {
    alert('Profile updated successfully!');
    setIsEditing(false);
  };

  const handleChange = (e) => {
    setProfileData({ ...profileData, [e.target.name]: e.target.value });
  };

  const totalClasses = attendanceData.length;
  const presentCount = attendanceData.filter(a => a.status === 'Present').length;
  const absentCount = attendanceData.filter(a => a.status === 'Absent').length;
  const attendanceRate = totalClasses > 0 ? ((presentCount / totalClasses) * 100).toFixed(1) : '0.0';

  return (
    <div className="student-profile-page">
      <button className="back-btn" onClick={onBack}>← Back</button>

      <div className="profile-main-card">
        {/* Profile Header */}
        <div className="profile-header">
          <div className="avatar-section">
            <div className="avatar-circle">
              {profileData.full_name.charAt(0).toUpperCase()}
            </div>
          </div>
          <div className="header-info">
            <h1 className="student-name">{profileData.full_name}</h1>
            <p className="student-id">{profileData.student_id}</p>
            <p className="student-email">{profileData.email}</p>
          </div>
          {!isEditing && (
            <button className="edit-profile-btn" onClick={() => setIsEditing(true)}>Edit Profile</button>
          )}
        </div>

        {/* Student Information Section */}
        <div className="student-info-section">
          <h2 className="section-title">Student Information</h2>
          <div className="info-grid">
            <div className="info-item">
              <label className="info-label">Year Level</label>
              <span className="info-value">{profileData.year}</span>
            </div>
            <div className="info-item">
              <label className="info-label">Course</label>
              <span className="info-value">{profileData.course}</span>
            </div>
            <div className="info-item">
              <label className="info-label">Email</label>
              {isEditing ? (
                <input
                  type="email"
                  name="email"
                  value={profileData.email}
                  onChange={handleChange}
                  className="info-input"
                />
              ) : (
                <span className="info-value">{profileData.email}</span>
              )}
            </div>
          </div>

          {isEditing && (
            <div className="edit-actions">
              <button className="save-btn" onClick={handleSave}>Save</button>
              <button className="cancel-btn" onClick={() => setIsEditing(false)}>Cancel</button>
            </div>
          )}
        </div>

        {/* Attendance Insights & Identity Activity Panel */}
        <AttendanceInsightsPanel studentId={userInfo?.student_id} />

        {/* Attendance Section */}
        <div className="attendance-section">
          <h2 className="section-title">Attendance Overview</h2>

          {loading ? (
            <div className="loading">
              <div className="spinner"></div>
              <p>Loading attendance data...</p>
            </div>
          ) : (
            <>
              <div className="attendance-stats-grid">
                <div className="stat-card">
                  <div className="stat-icon">📚</div>
                  <div className="stat-content">
                    <span className="stat-label">Total Classes</span>
                    <span className="stat-value">{totalClasses}</span>
                  </div>
                </div>
                <div className="stat-card">
                  <div className="stat-icon">✅</div>
                  <div className="stat-content">
                    <span className="stat-label">Present</span>
                    <span className="stat-value">{presentCount}</span>
                  </div>
                </div>
                <div className="stat-card">
                  <div className="stat-icon">❌</div>
                  <div className="stat-content">
                    <span className="stat-label">Absent</span>
                    <span className="stat-value">{absentCount}</span>
                  </div>
                </div>
                <div className="stat-card">
                  <div className="stat-icon">📊</div>
                  <div className="stat-content">
                    <span className="stat-label">Attendance Rate</span>
                    <span className="stat-value">{attendanceRate}%</span>
                  </div>
                </div>
              </div>

              {/* Recent Attendance */}
              <div className="recent-attendance">
                <h3 className="subsection-title">Recent Attendance</h3>
                <div className="attendance-table">
                  <div className="table-header">
                    <span>Date</span>
                    <span>Subject</span>
                    <span>Status</span>
                  </div>
                  {attendanceData.slice(0, 5).map((record, index) => (
                    <div key={index} className="table-row">
                      <span>{record.date}</span>
                      <span>{record.subject}</span>
                      <span className={`status-badge ${record.status.toLowerCase()}`}>
                        {record.status}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
