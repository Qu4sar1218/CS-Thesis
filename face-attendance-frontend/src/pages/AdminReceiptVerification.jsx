import React, { useState, useEffect, useCallback } from "react";
import axios from "axios";
import "../styles/AdminReceiptVerification.css";

// Helper function to format student names as: Firstname Lastname M.
const formatStudentName = (firstName, middleName, lastName) => {
  if (!firstName || !lastName) {
    return `${firstName || ''} ${lastName || ''}`.trim();
  }

  // Get middle initial if middle name exists
  const middleInitial = middleName ? ` ${middleName[0]}.` : '';

  return `${firstName} ${lastName}${middleInitial}`.trim();
};

//add filter for receipts per course
function AdminReceiptVerification({ onBack }) {
  const [receipts, setReceipts] = useState([]);
  const [filteredReceipts, setFilteredReceipts] = useState([]);
  const [selectedReceipt, setSelectedReceipt] = useState(null);
  const [loading, setLoading] = useState(true);
  const [verifying, setVerifying] = useState(false);
  const [message, setMessage] = useState("");
  const [filters, setFilters] = useState({
    course: "",
    year: "",
    gradeLevel: ""
  });
  const [showModal, setShowModal] = useState(false);
  const [allCourses, setAllCourses] = useState([]);
  const [allYears, setAllYears] = useState([]);

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:8000";

  const fetchReceipts = useCallback(async () => {
    try {
      const receiptsData = (await axios.get(`${BACKEND_URL}/receipts`)).data.receipts || [];
      const studentsData = (await axios.get(`${BACKEND_URL}/students`)).data.students || [];
      const coursesData = (await axios.get(`${BACKEND_URL}/courses`)).data.courses || [];

      // Create a map of student_id to student data
      const studentsMap = {};
      studentsData.forEach(student => {
        studentsMap[student.student_id] = student;
      });

      // Merge student data into receipts
      const receiptsWithStudents = receiptsData.map(receipt => ({
        ...receipt,
        student_name: studentsMap[receipt.student_id] ?
          formatStudentName(
            studentsMap[receipt.student_id].first_name,
            studentsMap[receipt.student_id].middle_name,
            studentsMap[receipt.student_id].last_name
          ) : 'Unknown',
        student_course: studentsMap[receipt.student_id]?.course || 'Unknown',
        student_year: studentsMap[receipt.student_id]?.year || 'Unknown'
      }));

      setReceipts(receiptsWithStudents);

      // Set all available courses
      setAllCourses([...new Set(coursesData.map(c => c.code))]);

      // Set all available years from students
      setAllYears([...new Set(studentsData.map(s => s.year).filter(y => y))]);
    } catch (error) {
      console.error("Error fetching data:", error);
      setMessage("Failed to load data. Please try again.");
    } finally {
      setLoading(false);
    }
  }, [BACKEND_URL]);

  useEffect(() => {
    fetchReceipts();
  }, [fetchReceipts]);

  // Filter receipts based on filters
  useEffect(() => {
    let filtered = receipts;

    if (filters.course) {
      filtered = filtered.filter(receipt => receipt.student_course === filters.course);
    }

    if (filters.year) {
      filtered = filtered.filter(receipt => receipt.student_year === filters.year);
    }

    if (filters.gradeLevel) {
      filtered = filtered.filter(receipt => receipt.student_year === filters.gradeLevel);
    }

    setFilteredReceipts(filtered);
  }, [receipts, filters]);

  const handleVerify = async (status) => {
    if (!selectedReceipt) return;

    setVerifying(true);
    setMessage("");

    try {
      await axios.put(`${BACKEND_URL}/receipts/${selectedReceipt._id}/verify`, {
        status: status,
        verified_by: "admin" // In a real app, this would be the current admin's ID
      });

      setMessage(`Receipt ${status} successfully!`);
      // Refresh receipts
      await fetchReceipts();
      setSelectedReceipt(null);
      setShowModal(false);
    } catch (error) {
      console.error("Error verifying receipt:", error);
      setMessage("Failed to verify receipt. Please try again.");
    } finally {
      setVerifying(false);
    }
  };

  const handleDelete = async () => {
    if (!selectedReceipt) return;

    if (!window.confirm("Are you sure you want to delete this receipt? This action cannot be undone.")) {
      return;
    }

    setVerifying(true);
    setMessage("");

    try {
      await axios.delete(`${BACKEND_URL}/receipts/${selectedReceipt._id}`);
      setMessage("Receipt deleted successfully!");
      // Refresh receipts
      await fetchReceipts();
      setSelectedReceipt(null);
      setShowModal(false);
    } catch (error) {
      console.error("Error deleting receipt:", error);
      setMessage("Failed to delete receipt. Please try again.");
    } finally {
      setVerifying(false);
    }
  };



  const handleSelectReceipt = (receipt) => {
    setSelectedReceipt(receipt);
    setShowModal(true);
  };

  const handleCloseModal = () => {
    setSelectedReceipt(null);
    setShowModal(false);
  };

  const getStatusColor = (status) => {
    switch (status) {
      case "pending": return "pending";
      case "verified": return "verified";
      case "rejected": return "rejected";
      default: return "pending";
    }
  };

  if (loading) {
    return (
      <div className="receipt-verification">
        <div className="loading">Loading receipts...</div>
      </div>
    );
  }

  return (
    <div className="receipt-verification">
      <header className="verification-header">
        <h1>Receipt Verification</h1>
        <p>Review and verify student receipt submissions for events.</p>
      </header>

      {message && (
        <div className={`message ${message.includes("successfully") ? "success" : "error"}`}>
          {message}
        </div>
      )}

      <div className="verification-content">
        <section className="filters-section">
          <h3>Filter Receipts</h3>
          <div className="filters">
            <select
              value={filters.course}
              onChange={(e) => setFilters({...filters, course: e.target.value})}
            >
              <option value="">All Courses</option>
              {allCourses.map(course => (
                <option key={course} value={course}>{course}</option>
              ))}
            </select>
            
            <select
              value={filters.gradeLevel}
              onChange={(e) => setFilters({...filters, gradeLevel: e.target.value})}
            >
              <option value="">All Year Levels</option>
              {allYears.map(grade => (
                <option key={grade} value={grade}>{grade}</option>
              ))}
            </select>
          </div>
        </section>
        <section className="receipts-list">
          <h2>Pending Receipts ({filteredReceipts.length})</h2>
          {filteredReceipts.length > 0 ? (
            <div className="receipts-grid">
              {filteredReceipts.map((receipt) => (
                <div
                  key={receipt._id}
                  className={`receipt-card ${selectedReceipt && selectedReceipt._id === receipt._id ? "selected" : ""}`}
                  onClick={() => handleSelectReceipt(receipt)}
                >
                  <div className="receipt-header">
                    <h3>Receipt #{receipt._id.slice(-6)}</h3>
                    <span className={`status ${getStatusColor(receipt.status)}`}>
                      {receipt.status}
                    </span>
                  </div>
                  <div className="receipt-info">
                    <p><strong>Student:</strong> {receipt.student_name}</p>
                    <p><strong>Student ID:</strong> {receipt.student_id}</p>
                    <p><strong>Course:</strong> {receipt.student_course}</p>
                    <p><strong>Event ID:</strong> {receipt.event_id}</p>
                    <p><strong>Transaction:</strong> {receipt.transaction_id}</p>
                    <p><strong>Submitted:</strong> {new Date(receipt.submitted_at).toLocaleDateString()}</p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-state">
              <p>No receipts found matching the filters.</p>
            </div>
          )}
        </section>
      </div>

      {showModal && selectedReceipt && (
        <div className="modal-overlay" onClick={handleCloseModal}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={handleCloseModal}>×</button>
            <h2>Receipt Details</h2>
            <div className="detail-content">
              <div className="detail-info">
                <p><strong>Receipt ID:</strong> {selectedReceipt._id}</p>
                <p><strong>Student Name:</strong> {selectedReceipt.student_name}</p>
                <p><strong>Student ID:</strong> {selectedReceipt.student_id}</p>
                <p><strong>Course:</strong> {selectedReceipt.student_course}</p>
                <p><strong>Year:</strong> {selectedReceipt.student_year}</p>
                <p><strong>Event ID:</strong> {selectedReceipt.event_id}</p>
                <p><strong>Transaction ID:</strong> {selectedReceipt.transaction_id}</p>
                <p><strong>Status:</strong> {selectedReceipt.status}</p>
                <p><strong>Submitted At:</strong> {new Date(selectedReceipt.submitted_at).toLocaleString()}</p>
                {selectedReceipt.verified_at && (
                  <p><strong>Verified At:</strong> {new Date(selectedReceipt.verified_at).toLocaleString()}</p>
                )}
                {selectedReceipt.verified_by && (
                  <p><strong>Verified By:</strong> {selectedReceipt.verified_by}</p>
                )}
              </div>

              <div className="receipt-image">
                <h3>Receipt Image</h3>
                <img
                  src={selectedReceipt.receipt_image}
                  alt="Receipt"
                  className="receipt-preview"
                />
              </div>

              <div className="verification-actions">
                {selectedReceipt.status === "pending" && (
                  <>
                    <button
                      className="btn verify"
                      onClick={() => handleVerify("verified")}
                      disabled={verifying}
                    >
                      {verifying ? "Verifying..." : "Verify Receipt"}
                    </button>
                    <button
                      className="btn reject"
                      onClick={() => handleVerify("rejected")}
                      disabled={verifying}
                    >
                      {verifying ? "Rejecting..." : "Reject Receipt"}
                    </button>
                  </>
                )}
                <button
                  className="btn delete"
                  onClick={handleDelete}
                  disabled={verifying}
                >
                  {verifying ? "Deleting..." : "🗑️ Delete Receipt"}
                </button>
                <button
                  className="btn cancel"
                  onClick={handleCloseModal}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="page-actions">
        <button type="button" className="back-btn" onClick={onBack}>
          <span className="btn-icon">←</span>
          <span className="btn-text">Back to Dashboard</span>
        </button>
      </div>
    </div>
  );
}

export default AdminReceiptVerification;
