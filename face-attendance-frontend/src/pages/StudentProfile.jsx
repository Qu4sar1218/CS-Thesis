import React, { useState, useEffect } from 'react';
import '../styles/StudentProfile.css';
import AttendanceInsightsPanel from '../components/AttendanceInsightsPanel';

// Helper function to format student names as: Firstname Lastname M.
const formatStudentName = (firstName, middleName, lastName) => {
  if (!firstName || !lastName) {
    return `${firstName || ''} ${lastName || ''}`.trim();
  }

  // Capitalize each word in firstName and lastName
  const capitalize = (str) => str.split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()).join(' ');

  const capitalizedFirst = capitalize(firstName);
  const capitalizedLast = capitalize(lastName);

  const lastParts = capitalizedLast.split();
  if (lastParts.length > 1) {
    // Handle multiple words in last name
    const middleInitial = middleName ? ` ${middleName.charAt(0).toUpperCase()}.` : '';
    return `${capitalizedFirst} ${lastParts[0]}${middleInitial} ${lastParts.slice(1).join(' ')}`.trim();
  } else {
    // Standard format
    const middleInitial = middleName ? ` ${middleName.charAt(0).toUpperCase()}.` : '';
    return `${capitalizedFirst}${middleInitial} ${capitalizedLast}`.trim();
  }
};

export default function StudentProfile({ onBack, userInfo }) {
  const [isEditing, setIsEditing] = useState(false);
  const [attendanceData, setAttendanceData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [profileLoading, setProfileLoading] = useState(true);
  const [profileData, setProfileData] = useState({
    full_name: userInfo?.full_name || '',
    email: userInfo?.email || '',
    student_id: userInfo?.user_id || '',
    course: userInfo?.course || '',
    year: userInfo?.year || '',
    password: '',
  });

  useEffect(() => {
    const fetchStudentProfile = async () => {
      if (!userInfo?.user_id) {
        setProfileLoading(false);
        return;
      }

      try {
        const response = await fetch(`http://127.0.0.1:8000/students/${userInfo.user_id}`);
        if (response.ok) {
          const studentData = await response.json();
          // Use formatted name from backend if available, otherwise construct it
          const formattedName = studentData.formatted_name || formatStudentName(
            studentData.first_name || '',
            studentData.middle_name || '',
            studentData.last_name || ''
          );
          setProfileData({
            full_name: formattedName,
            email: studentData.email || '',
            student_id: studentData.student_id || '',
            course: studentData.course || '',
            year: studentData.year || '',
            password: '',
          });
        } else {
          console.error('Failed to fetch student profile data');
          // Fallback to userInfo if API fails
          setProfileData({
            full_name: userInfo?.full_name || '',
            email: userInfo?.email || '',
            student_id: userInfo?.user_id || '',
            course: userInfo?.course || '',
            year: userInfo?.year || '',
            password: '',
          });
        }
      } catch (error) {
        console.error('Error fetching student profile data:', error);
        // Fallback to userInfo if API fails
        setProfileData({
          full_name: userInfo?.full_name || '',
          email: userInfo?.email || '',
          student_id: userInfo?.user_id || '',
          course: userInfo?.course || '',
          year: userInfo?.year || '',
          password: '',
        });
      } finally {
        setProfileLoading(false);
      }
    };

    const fetchAttendance = async () => {
      if (!userInfo?.user_id) {
        setLoading(false);
        return;
      }

      try {
        const response = await fetch(`http://127.0.0.1:8000/analytics/student/${userInfo.user_id}`);
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

    fetchStudentProfile();
    fetchAttendance();
    const intervalId = setInterval(fetchAttendance, 5000);
    return () => clearInterval(intervalId);
  }, [userInfo]);

  const handleSave = async () => {
    try {
      // Prepare update data
      const updateData = {};

      // Only include email if it changed
      if (profileData.email !== userInfo?.email) {
        updateData.email = profileData.email;
      }

      // Include password if provided
      if (profileData.password.trim()) {
        updateData.password = profileData.password;
      }

      // Only send request if there's something to update
      if (Object.keys(updateData).length > 0) {
        const response = await fetch(`http://127.0.0.1:8000/students/${profileData.student_id}`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(updateData),
        });

        if (response.ok) {
          alert('Profile updated successfully!');
          // Reset password field
          setProfileData(prev => ({ ...prev, password: '' }));
        } else {
          const errorData = await response.json();
          alert(`Failed to update profile: ${errorData.detail || 'Unknown error'}`);
          return;
        }
      } else {
        alert('No changes to save.');
      }

      setIsEditing(false);
    } catch (error) {
      console.error('Error updating profile:', error);
      alert('An error occurred while updating the profile.');
    }
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
            <h1 className="profile-student-name">{profileData.full_name}</h1>
            <p className="profile-student-id">{profileData.student_id}</p>
            <p className="student-email">{profileData.email}</p>
          </div>
          {!isEditing && (
            <button className="edit-profile-btn" onClick={() => setIsEditing(true)}>Edit Profile</button>
          )}
        </div>

        {/* Student Information Section */}
        <div className="student-info-section">
          <h2 className="section-title">Student Information</h2>
          {profileLoading ? (
            <div className="loading">
              <div className="spinner"></div>
              <p>Loading profile data...</p>
            </div>
          ) : (
            <>
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
                {isEditing && (
                  <div className="info-item">
                    <label className="info-label">New Password</label>
                    <input
                      type="password"
                      name="password"
                      value={profileData.password}
                      onChange={handleChange}
                      className="info-input"
                      placeholder="Enter new password (leave empty to keep current)"
                    />
                  </div>
                )}
              </div>

              {isEditing && (
                <div className="edit-actions">
                  <button className="save-btn" onClick={handleSave}>Save</button>
                  <button className="cancel-btn" onClick={() => setIsEditing(false)}>Cancel</button>
                </div>
              )}
            </>
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
