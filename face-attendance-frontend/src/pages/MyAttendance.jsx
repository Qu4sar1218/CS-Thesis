import React, { useState, useEffect } from 'react';
import '../styles/MyAttendance.css';

export default function MyAttendance({ onBack, userInfo }) {
  const [attendanceData, setAttendanceData] = useState([]);
  const [filteredData, setFilteredData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [subjectFilter, setSubjectFilter] = useState('');
  const [dateFilter, setDateFilter] = useState('');
  const [sortConfig, setSortConfig] = useState({ key: 'date', direction: 'desc' });
  const [selectedRecord, setSelectedRecord] = useState(null);
  const [showModal, setShowModal] = useState(false);

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
          // Transform the data to match the expected format
          const transformedData = data.attendance.map(record => ({
            date: record.date,
            subject: record.subject || record.class_id || 'Unknown Subject',
            status: record.status,
            time: record.check_in_time || 'N/A',
            room: 'N/A' // Room info not available in current API
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

    // Initial fetch
    fetchAttendance();

    // Set up polling every 5 seconds for real-time updates
    const intervalId = setInterval(fetchAttendance, 5000);

    // Cleanup interval on unmount
    return () => clearInterval(intervalId);
  }, [userInfo]);

  // Get unique subjects for filter dropdown
  const uniqueSubjects = [...new Set(attendanceData.map(record => record.subject))];

  // Handle sorting
  const handleSort = (key) => {
    let direction = 'asc';
    if (sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
  };

  // Handle modal
  const openModal = (record) => {
    setSelectedRecord(record);
    setShowModal(true);
  };

  const closeModal = () => {
    setShowModal(false);
    setSelectedRecord(null);
  };

  // Filter and sort data
  useEffect(() => {
    let filtered = attendanceData.filter(record => {
      const matchesSearch = record.subject.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           record.date.includes(searchTerm);
      const matchesSubject = subjectFilter === '' || record.subject === subjectFilter;
      const matchesDate = dateFilter === '' || record.date === dateFilter;

      return matchesSearch && matchesSubject && matchesDate;
    });

    // Sort data
    filtered.sort((a, b) => {
      if (a[sortConfig.key] < b[sortConfig.key]) {
        return sortConfig.direction === 'asc' ? -1 : 1;
      }
      if (a[sortConfig.key] > b[sortConfig.key]) {
        return sortConfig.direction === 'asc' ? 1 : -1;
      }
      return 0;
    });

    setFilteredData(filtered);
  }, [attendanceData, searchTerm, subjectFilter, dateFilter, sortConfig]);

  return (
    <div className="my-attendance-container">
      <button className="back-btn" onClick={onBack}>← Back</button>

      <div className="attendance-header">
        <h1>My Attendance</h1>
      </div>

      {loading ? (
        <div className="loading">
          <div className="spinner"></div>
          <p>Loading attendance data...</p>
        </div>
      ) : (
        <div className="attendance-content">
          <div className="attendance-summary">
            <div className="summary-card total">
              <div className="card-icon">📚</div>
              <div className="card-content">
                <h3>Total Classes</h3>
                <p className="card-value">{attendanceData.length}</p>
              </div>
            </div>
            <div className="summary-card present">
              <div className="card-icon">✅</div>
              <div className="card-content">
                <h3>Present</h3>
                <p className="card-value">{attendanceData.filter(a => a.status === 'Present').length}</p>
              </div>
            </div>
            <div className="summary-card absent">
              <div className="card-icon">❌</div>
              <div className="card-content">
                <h3>Absent</h3>
                <p className="card-value">{attendanceData.filter(a => a.status === 'Absent').length}</p>
              </div>
            </div>
            <div className="summary-card rate">
              <div className="card-icon">📊</div>
              <div className="card-content">
                <h3>Attendance Rate</h3>
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${(attendanceData.filter(a => a.status === 'Present').length / attendanceData.length) * 100}%` }}></div>
                </div>
                <p className="card-value">{((attendanceData.filter(a => a.status === 'Present').length / attendanceData.length) * 100).toFixed(1)}%</p>
              </div>
            </div>
          </div>

          <div className="attendance-table-section">
            <div className="table-header">
              <h2>Attendance Records</h2>
              <div className="filters-container">
                <div className="search-container">
                  <input
                    type="text"
                    placeholder="Search by subject or date..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="search-input"
                  />
                </div>
                <div className="filter-container">
                  <select
                    value={subjectFilter}
                    onChange={(e) => setSubjectFilter(e.target.value)}
                    className="filter-select"
                  >
                    <option value="">All Subjects</option>
                    {uniqueSubjects.map(subject => (
                      <option key={subject} value={subject}>{subject}</option>
                    ))}
                  </select>
                </div>
                <div className="filter-container">
                  <input
                    type="date"
                    value={dateFilter}
                    onChange={(e) => setDateFilter(e.target.value)}
                    className="filter-date"
                  />
                </div>
              </div>
            </div>
            <div className="table-container">
              <table className="attendance-table">
                <thead>
                  <tr>
                    <th onClick={() => handleSort('date')} className="sortable">
                      Date {sortConfig.key === 'date' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                    </th>
                    <th onClick={() => handleSort('subject')} className="sortable">
                      Subject {sortConfig.key === 'subject' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                    </th>
                    <th onClick={() => handleSort('status')} className="sortable">
                      Status {sortConfig.key === 'status' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                    </th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredData.map((record, index) => (
                    <tr key={index}>
                      <td>{record.date}</td>
                      <td>{record.subject}</td>
                      <td>
                        <span className={`status-badge ${record.status.toLowerCase()}`}>
                          {record.status}
                        </span>
                      </td>
                      <td>
                        <button
                          className="view-details-btn"
                          onClick={() => openModal(record)}
                        >
                          View Details
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {showModal && selectedRecord && (
            <div className="modal-overlay" onClick={closeModal}>
              <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                <div className="modal-header">
                  <h3>Attendance Details</h3>
                  <button className="modal-close" onClick={closeModal}>×</button>
                </div>
                <div className="modal-body">
                  <div className="detail-row">
                    <span className="detail-label">Date:</span>
                    <span className="detail-value">{selectedRecord.date}</span>
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">Subject:</span>
                    <span className="detail-value">{selectedRecord.subject}</span>
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">Time:</span>
                    <span className="detail-value">{selectedRecord.time}</span>
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">Room:</span>
                    <span className="detail-value">{selectedRecord.room}</span>
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">Status:</span>
                    <span className={`detail-value status-${selectedRecord.status.toLowerCase()}`}>
                      {selectedRecord.status}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
