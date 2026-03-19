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

  const lastParts = capitalizedLast.split(' ');
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
  const [profileLoading, setProfileLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('profile'); // 'profile', 'classes', 'events'
  const [attendanceLoading, setAttendanceLoading] = useState(false);
  const [attendanceLogs, setAttendanceLogs] = useState({ classes: [], events: [], hallway: [] });
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

    const fetchAttendanceHistory = async () => {
      if (!userInfo?.user_id) return;
      setAttendanceLoading(true);
      try {
        const response = await fetch(`http://127.0.0.1:8000/analytics/student/${userInfo.user_id}`);
        if (response.ok) {
          const data = await response.json();
          const allLogs = data.attendance || [];
          setAttendanceLogs({
            classes: allLogs.filter(log => log.mode === 'class' || !log.mode),
            events: allLogs.filter(log => log.mode === 'events')
          });
        }
      } catch (error) {
        console.error('Error fetching attendance history:', error);
      } finally {
        setAttendanceLoading(false);
      }
    };

    fetchStudentProfile();
    fetchAttendanceHistory();
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

  return (
    <div className="student-profile-page">
      <button className="back-btn" onClick={onBack}>? Back</button>

      {/* Profile Header Card */}
      <div className="profile-header-card">
        <div className="avatar-section">
          <div className="avatar-circle">
            {profileData.full_name.charAt(0).toUpperCase()}
          </div>
        </div>
        <div className="header-info">
          <h1 className="profile-student-name">{profileData.full_name}</h1>
          <p className="profile-student-id">Student ID: {profileData.student_id}</p>
          <p className="student-meta">{profileData.course} • {profileData.year}</p>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="profile-tabs">
        <button 
          className={`tab-btn ${activeTab === 'profile' ? 'active' : ''}`}
          onClick={() => setActiveTab('profile')}
        >
          Profile & Insights
        </button>

        <button 
          className={`tab-btn ${activeTab === 'classes' ? 'active' : ''}`}
          onClick={() => setActiveTab('classes')}
        >
          Class Attendance
        </button>
        <button 
          className={`tab-btn ${activeTab === 'events' ? 'active' : ''}`}
          onClick={() => setActiveTab('events')}
        >
          Event Attendance
        </button>
        <button 
          className={`tab-btn ${activeTab === 'hallway' ? 'active' : ''}`}
          onClick={() => setActiveTab('hallway')}
        >
          Hallway Attendance
        </button>
      </div>

      {/* Tab Content */}
      <div className="profile-content">
        
        {/* PROFILE TAB */}
        {activeTab === 'profile' && (
          <div className="profile-main-layout">
            {/* Full Width Details since Insights simplified */}
            <div className="profile-details-card full-width">
              <div className="card-header">
                <h2>Personal Details</h2>
                {!isEditing && (
                  <button className="edit-link" onClick={() => setIsEditing(true)}>Edit</button>
                )}
              </div>
              
              {profileLoading ? (
                <div className="loading-state">Loading details...</div>
              ) : (
                <div className="info-form">
                  <div className="info-group">
                    <label>Full Name</label>
                    <div className="static-value">{profileData.full_name}</div>
                  </div>
                  <div className="info-group">
                    <label>Course & Year</label>
                    <div className="static-value">{profileData.course} - {profileData.year}</div>
                  </div>
                  <div className="info-group">
                    <label>Email Address</label>
                    {isEditing ? (
                      <input
                        type="email"
                        name="email"
                        value={profileData.email}
                        onChange={handleChange}
                        className="edit-input"
                      />
                    ) : (
                      <div className="static-value">{profileData.email || 'No email set'}</div>
                    )}
                  </div>
                  
                  {isEditing && (
                    <div className="info-group">
                      <label>New Password</label>
                      <input
                        type="password"
                        name="password"
                        value={profileData.password}
                        onChange={handleChange}
                        className="edit-input"
                        placeholder="Leave blank to keep current"
                      />
                    </div>
                  )}

                  {isEditing && (
                    <div className="form-actions">
                      <button className="save-btn" onClick={handleSave}>Save Changes</button>
                      <button className="cancel-btn" onClick={() => setIsEditing(false)}>Cancel</button>
                    </div>
                  )}
                </div>
              )}

              {/* Simplified Insights below details */}
              <AttendanceInsightsPanel studentId={userInfo?.user_id || userInfo?.student_id} />
            </div>
          </div>
        )}

        {/* ATTENDANCE TABLES (Shared Structure) */}
        {(activeTab === 'classes' || activeTab === 'events' || activeTab === 'hallway') && (
          <div className="attendance-history-card">
            <div className="card-header">
              <h2>{activeTab === 'classes' ? 'Class Attendance History' : activeTab === 'events' ? 'Event Participation History' : 'Hallway Sessions'}</h2>
            </div>
            
            {attendanceLoading ? (
              <div className="loading-state">Loading attendance records...</div>
            ) : (
              <div className="table-responsive">
                <table className="attendance-table">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Time</th>
                      <th>{activeTab === 'classes' ? 'Subject/Class' : activeTab === 'events' ? 'Event' : 'Session ID'}</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {attendanceLogs[activeTab]?.length > 0 ? (
                      attendanceLogs[activeTab].map((record, idx) => (
                        <tr key={record._id || idx}>
                          <td>{record.date}</td>
                          <td>{record.check_in_time || record.time || '-'}</td>
                          <td>{activeTab === 'classes' ? (record.subject || record.class_name || 'Class') : 
                                activeTab === 'events' ? (record.event_name || 'Event') : 
                                (record.session_id || record.source_collection || 'Hallway')}</td>
                          <td>
                            <span className={`status-badge ${record.status?.toLowerCase()}`}>
                              {record.status}
                            </span>
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan="4" className="empty-state">No attendance records found.</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
