import React, { useState, useEffect, useCallback } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  PointElement,
  LineElement,
  RadialLinearScale,
} from 'chart.js';
import { Doughnut, Bar, Radar, PolarArea, Line } from 'react-chartjs-2';
import '../styles/Analytics.css';

const BACKEND_URL = 'http://127.0.0.1:8000';

export default function Analytics({ onBack }) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [rawAttendance, setRawAttendance] = useState([]);
  const [summary, setSummary] = useState({}); 
  const [filteredData, setFilteredData] = useState([]);
  const [courses, setCourses] = useState([]);
  const [yearLevels, setYearLevels] = useState([]);
  const [filters, setFilters] = useState({
    course: 'All',
    yearLevel: 'All',
    status: 'All',
    startDate: '',
    endDate: ''
  });
  const [showTable, setShowTable] = useState(false);

  // State for chart data
  const [overallPieData, setOverallPieData] = useState({ datasets: [] });
  const [courseBarData, setCourseBarData] = useState({ datasets: [] });
  const [courseRadarData, setCourseRadarData] = useState({ datasets: [] });
  const [yearPolarData, setYearPolarData] = useState({ datasets: [] });
  const [yearLineData, setYearLineData] = useState({ datasets: [] });

  // Helper to get computed CSS variables
  const getCssVar = (name) => {
    if (typeof window === 'undefined') return '';
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  };

  // Chart options can be defined once
  const chartOptions = {
    overallPie: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } } },
    courseBar: { responsive: true, maintainAspectRatio: false, indexAxis: 'y', scales: { x: { beginAtZero: true } }, plugins: { legend: { display: false } } },
    courseRadar: { responsive: true, maintainAspectRatio: false, scales: { r: { beginAtZero: true } } },
    yearPolar: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } } },
    yearLine: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, max: 100 } }, plugins: { legend: { display: false } } }
  };

  // Register Chart.js components
  ChartJS.register(
    CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, ArcElement,
    PointElement, LineElement, RadialLinearScale
  );
  
  // Icons (SVG for professional look)
  const icons = {
    present: <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>,
    late: <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10 10-4.5 10-10S17.5 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z"/></svg>,
    absent: <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>,
    chart: <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM8 17l-4-4 1.41-1.41L8 14.17l7.59-7.59L19 9l-11 11z"/></svg>,
    table: <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M20 5H14v1.17L16.17 11H20V5zm-2 13H10v-1.17l3.17 3.17H18v-6h2v6zM6 17H4v-2h2v2zm0-6H4v-2h2v2zM8 5h8V3H8v2zm0 14v-2H4v2h4z"/></svg>,
    download: <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg>
  };

  const normalizeStatus = (status) => {
    const normalized = (status || '').toString().trim().toUpperCase();
    if (['PRESENT'].includes(normalized)) return 'Present';
    if (['LATE'].includes(normalized)) return 'Late';
    return 'Absent';
  };

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const coursesRes = await fetch(`${BACKEND_URL}/courses`);
      const coursesData = await coursesRes.json();
      setCourses(coursesData.courses || []);

      const studentsRes = await fetch(`${BACKEND_URL}/students`);
      const studentsData = await studentsRes.json();
      const studentsMapData = {};
      const yearSet = new Set();
      studentsData.students?.forEach(s => {
        studentsMapData[s.student_id] = {
          name: `${s.first_name || ''} ${s.last_name || ''}`.trim() || s.student_id,
          course: s.course || 'Unknown',
          year_level: s.year || 'Unknown'
        };
        yearSet.add(s.year || 'Unknown');
      });
      setYearLevels(Array.from(yearSet).sort());

      const attendanceRes = await fetch(`${BACKEND_URL}/attendance-db`);
      const attendanceData = await attendanceRes.json();
      let enriched = attendanceData.attendance?.map(record => ({
        ...record,
        name: studentsMapData[record.student_id]?.name || record.name || record.student_id || 'Unknown',
        course: studentsMapData[record.student_id]?.course || record.course || 'Unknown', 
        year_level: studentsMapData[record.student_id]?.year_level || record.year || 'Unknown',
        attendance_status: normalizeStatus(record.status)
      })) || [];
      setRawAttendance(enriched);
    } catch (err) {
      setError(`Failed to load data: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, []);



  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // When raw data is fetched, populate the dashboard with the full, unfiltered dataset.
  useEffect(() => {
    if (rawAttendance.length > 0) {
        setFilteredData(rawAttendance);
        const total = rawAttendance.length;
        const present = rawAttendance.filter(d => d.attendance_status === 'Present').length;
        const late = rawAttendance.filter(d => d.attendance_status === 'Late').length;
        const absent = total - present - late;
        setSummary({ total, present, late, absent });
    }
  }, [rawAttendance]);

  const applyFilters = () => {
    let data = [...rawAttendance]; // Always filter from the original raw data
    if (filters.course !== 'All') data = data.filter(d => d.course === filters.course);
    if (filters.yearLevel !== 'All') data = data.filter(d => d.year_level === filters.yearLevel);
    if (filters.status !== 'All') data = data.filter(d => d.attendance_status === filters.status);
    if (filters.startDate) data = data.filter(d => d.date >= filters.startDate);
    if (filters.endDate) data = data.filter(d => d.date <= filters.endDate);

    const total = data.length;
    const present = data.filter(d => d.attendance_status === 'Present').length;
    const late = data.filter(d => d.attendance_status === 'Late').length;
    const absent = total - present - late;

    setFilteredData(data);
    setSummary({ total, present, late, absent });
  };

  // This useEffect hook reactively updates chart data whenever the filtered data changes.
  useEffect(() => {
    const data = filteredData;

    
    if (!data.length) {
      // Fallback empty charts with valid structure to prevent Legend error
      setOverallPieData({
        labels: ['Present', 'Late', 'Absent'],
        datasets: [{ data: [0, 0, 0], backgroundColor: ['#16a34a', '#f59e0b', '#ef4444'] }]
      });
      setCourseBarData({
        labels: ['No Data'],
        datasets: [{ 
          label: 'Students', 
          data: [0], 
          backgroundColor: ['#3b82f6']
        }]
      });
      setCourseRadarData({
        labels: ['No Data'],
datasets: [
          { label: 'Present', data: [0], borderColor: '#16a34a', backgroundColor: 'rgba(16, 185, 129, 0.2)' },
          { label: 'Late', data: [0], borderColor: '#f59e0b', backgroundColor: 'rgba(245, 158, 11, 0.2)' },
          { label: 'Absent', data: [0], borderColor: '#ef4444', backgroundColor: 'rgba(239, 68, 68, 0.2)' }
        ]
      });
      setYearPolarData({
        labels: ['No Data'],
        datasets: [{ data: [0], backgroundColor: ['#3b82f6'] }]
      });
      setYearLineData({
        labels: ['No Data'],
        datasets: [{ label: 'Attendance %', data: [0], borderColor: '#3b82f6', backgroundColor: 'rgba(37, 99, 235, 0.1)' }]
      });
      return;
    };

    // Resolve CSS variables to actual colors for Chart.js
    const colors = {
        primary: getCssVar('--primary') || '#3b82f6',
        success: getCssVar('--success') || '#16a34a',
        warning: getCssVar('--warning') || '#f59e0b',
        danger: getCssVar('--danger') || '#ef4444',
        successLight: 'rgba(16, 185, 129, 0.2)',
        warningLight: 'rgba(245, 158, 11, 0.2)',
        dangerLight: 'rgba(239, 68, 68, 0.2)',
        primaryLight: 'rgba(37, 99, 235, 0.1)',
    };

    // Create a vibrant, consistent color palette for charts
    const colorPalette = [
        colors.primary,
        colors.success,
        '#ec4899', // pink-500
        '#8b5cf6', // violet-500
        '#06b6d4', // cyan-500
        colors.warning,
        '#6366f1', // indigo-500
        '#84cc16', // lime-500
    ];

    const { present, late, absent } = summary;

    // Overall Status (Doughnut)
    setOverallPieData({
      labels: ['Present', 'Late', 'Absent'],
      datasets: [{ data: [present, late, absent], backgroundColor: [colors.success, colors.warning, colors.danger] }]
    });

    // Courses Overview (Bar)
    const courseTotals = {};
    data.forEach(d => courseTotals[d.course] = (courseTotals[d.course] || 0) + 1);
    setCourseBarData({
      labels: Object.keys(courseTotals),
      datasets: [{ 
          label: 'Students', 
          data: Object.values(courseTotals), 
          backgroundColor: Object.keys(courseTotals).map((_, i) => colorPalette[i % colorPalette.length])
      }]
    });

    // Course Patterns (Radar)
    const courseStats = {};
    data.forEach(d => {
      if (!courseStats[d.course]) courseStats[d.course] = { Present: 0, Late: 0, Absent: 0 };
      courseStats[d.course][d.attendance_status]++;
    });
    const courseStatLabels = Object.keys(courseStats);
    setCourseRadarData({
      labels: courseStatLabels,
datasets: [
        { label: 'Present', data: courseStatLabels.map(l => courseStats[l].Present), borderColor: colors.success, backgroundColor: colors.successLight },
        { label: 'Late', data: courseStatLabels.map(l => courseStats[l].Late), borderColor: colors.warning, backgroundColor: colors.warningLight },
        { label: 'Absent', data: courseStatLabels.map(l => courseStats[l].Absent), borderColor: colors.danger, backgroundColor: colors.dangerLight }
      ]
    });

    // Year Distribution (PolarArea)
    const yearTotals = {};
    data.forEach(d => yearTotals[d.year_level] = (yearTotals[d.year_level] || 0) + 1);
    setYearPolarData({
      labels: Object.keys(yearTotals),
      datasets: [{ 
          data: Object.values(yearTotals), 
          backgroundColor: Object.keys(yearTotals).map((_, i) => colorPalette[i % colorPalette.length])
      }]
    });

    // Attendance Trends (Line)
    const yearStats = {};
    data.forEach(d => {
      if (!yearStats[d.year_level]) yearStats[d.year_level] = { total: 0, attended: 0 };
      yearStats[d.year_level].total++;
      if (d.attendance_status !== 'Absent') yearStats[d.year_level].attended++;
    });
    const yearStatLabels = Object.keys(yearStats);
    const rates = yearStatLabels.map(l => Math.round((yearStats[l].attended / yearStats[l].total) * 100 || 0));
    setYearLineData({
      labels: yearStatLabels,
      datasets: [{ label: 'Attendance %', data: rates, borderColor: colors.primary, backgroundColor: colors.primaryLight, tension: 0.4 }]
    });

  }, [filteredData, summary]);

  const handleFilterChange = (key, value) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  const getUniqueOptions = (field) => {
    // Use rawAttendance to ensure all options are always available
    const options = Array.from(new Set(rawAttendance.map(d => d[field] || 'Unknown'))).sort();
    return ['All', ...options];
  };

  const exportData = () => {
    const csv = 'Student ID,Name,Course,Year,Status,Date\n' + 
      filteredData.slice(0, 1000).map(r => 
        [r.student_id, `"${r.name}"`, r.course, r.year_level, r.attendance_status, r.date].join(',')
      ).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `attendance-${new Date().toISOString().slice(0,10)}.csv`;
    a.click();
  };

  if (loading) {
    return (
      <div className="analytics">
        <h1>📊 Analytics Dashboard</h1>
        <div className="loading-spinner">
          <div />
          <p>Loading attendance data...</p>
        </div>
        <div className="analytics-actions">
          <button className="analytics-btn analytics-btn-secondary" onClick={onBack}>← Back</button>
        </div>
      </div>
    );
  }

  return (
    <div className="analytics">
      <div className="analytics-header">
        <h1>{icons.chart} Analytics Dashboard</h1>
        <button className="analytics-btn analytics-btn-secondary" onClick={onBack}>← Back to Dashboard</button>
      </div>

      {error && <div className="error-message">{error}</div>}

      {/* Filters */}
      <div className="filters-section">
        <div className="filters-grid">
          <div className="filter-group">
            <label>Course</label>
            <select className="filter-select" value={filters.course} onChange={(e) => handleFilterChange('course', e.target.value)}>
              <option value="All">All Courses</option>
              {courses.map(course => (
                <option key={course.code} value={course.code}>{course.name} ({course.level || 'N/A'})</option>
              ))}
            </select>
          </div>
          <div className="filter-group">
            <label>Year Level</label>
            <select className="filter-select" value={filters.yearLevel} onChange={(e) => handleFilterChange('yearLevel', e.target.value)}>
              {['All', ...yearLevels].map(opt => <option key={opt} value={opt}>{opt}</option>)}
            </select>
          </div>
          <div className="filter-group">
            <label>Status</label>
            <select className="filter-select" value={filters.status} onChange={(e) => handleFilterChange('status', e.target.value)}>
              <option value="All">All</option>
              <option value="Present">Present</option>
              <option value="Late">Late</option>
              <option value="Absent">Absent</option>
            </select>
          </div>
          <div className="filter-group">
            <label>Start Date</label>
            <input type="date" className="filter-date" value={filters.startDate} onChange={(e) => handleFilterChange('startDate', e.target.value)} />
          </div>
          <div className="filter-group">
            <label>End Date</label>
            <input type="date" className="filter-date" value={filters.endDate} onChange={(e) => handleFilterChange('endDate', e.target.value)} />
          </div>
          <button className="analytics-btn analytics-btn-primary" onClick={applyFilters} disabled={loading}>
            🔍 Apply Filters ({filteredData.length})
          </button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="analytics-grid">
        <div className="analytics-card">
          <h3>{icons.chart} Total Records</h3>
          <div className="metric">{summary.total || filteredData.length}</div>
        </div>
        <div className="analytics-card">
          <h3>{icons.present} Present</h3>
          <div className="metric status-present">{summary.present || 0}</div>
        </div>
        <div className="analytics-card">
          <h3>{icons.late} Late</h3>
          <div className="metric status-late">{summary.late || 0}</div>
        </div>
        <div className="analytics-card">
          <h3>{icons.absent} Absent</h3>
          <div className="metric status-absent">{summary.absent || 0}</div>
        </div>
      </div>

      {/* Charts */}
      <div className="charts-grid">
        <div className="chart-section">
          <h2>{icons.chart} Status Distribution</h2>
          <div className="chart-container"><Doughnut data={overallPieData} options={chartOptions.overallPie} /></div>
        </div>
        <div className="chart-section">
          <h2>{icons.chart} Courses Overview</h2>
          <div className="chart-container"><Bar data={courseBarData} options={chartOptions.courseBar} /></div>
        </div>
        <div className="chart-section">
          <h2>{icons.chart} Course Patterns</h2>
          <div className="chart-container"><Radar data={courseRadarData} options={chartOptions.courseRadar} /></div>
        </div>
        <div className="chart-section">
          <h2>{icons.chart} Year Distribution</h2>
          <div className="chart-container"><PolarArea data={yearPolarData} options={chartOptions.yearPolar} /></div>
        </div>
        <div className="chart-section">
          <h2>{icons.chart} Attendance Trends</h2>
          <div className="chart-container"><Line data={yearLineData} options={chartOptions.yearLine} /></div>
        </div>
      </div>

      {/* Recent Data Table Toggle */}
      <div className="chart-section">
        <div className="table-toggle-header">
          <h2>{icons.table} Recent Records</h2>
          <button className="analytics-btn analytics-btn-secondary" onClick={() => setShowTable(!showTable)}>
            {showTable ? 'Hide' : 'Show'} Table
          </button>
        </div>
        {showTable && (
          <div className="table-container">
            <table className="analytics-table">
              <thead>
                <tr>
                  <th>Student</th>
                  <th>Course</th>
                  <th>Status</th>
                  <th>Date</th>
                </tr>
              </thead>
              <tbody>
                {filteredData.slice(0, 10).map((record, i) => (
                  <tr key={record.id || i}>
                    <td>{record.name}</td>
                    <td>{record.course}</td>
                    <td>
                      <span className={`status-${record.attendance_status.toLowerCase()}`}>
                        {record.attendance_status}
                      </span>
                    </td>
                    <td>{record.date}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Action Buttons */}
      <div className="analytics-actions">
        <button className="analytics-btn analytics-btn-success" onClick={exportData}>
          {icons.download} Export CSV ({filteredData.length} records)
        </button>
        <button className="analytics-btn analytics-btn-secondary" onClick={onBack}>← Back</button>
      </div>
    </div>
  );
}
