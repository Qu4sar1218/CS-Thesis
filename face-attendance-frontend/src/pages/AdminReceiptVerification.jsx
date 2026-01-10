import React, { useState, useEffect, useCallback } from "react"; 
import axios from "axios";
import "../styles/AdminReceiptVerification.css";
//add filter for receipts per course 
function AdminReceiptVerification({ onBack }) {
  const [receipts, setReceipts] = useState([]);
  const [selectedReceipt, setSelectedReceipt] = useState(null);
  const [loading, setLoading] = useState(true);
  const [verifying, setVerifying] = useState(false);
  const [message, setMessage] = useState("");

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:8000";

  const fetchReceipts = useCallback(async () => {
    try {
      setReceipts((await axios.get(`${BACKEND_URL}/receipts`)).data.receipts || []);
    } catch (error) {
      console.error("Error fetching receipts:", error);
      setMessage("Failed to load receipts. Please try again.");
    } finally {
      setLoading(false);
    }
  }, [BACKEND_URL]);

  useEffect(() => {
    fetchReceipts();
  }, [fetchReceipts]);

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
    } catch (error) {
      console.error("Error verifying receipt:", error);
      setMessage("Failed to verify receipt. Please try again.");
    } finally {
      setVerifying(false);
    }
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
        <section className="receipts-list">
          <h2>Pending Receipts</h2>
          {receipts.length > 0 ? (
            <div className="receipts-grid">
              {receipts.map((receipt) => (
                <div
                  key={receipt._id}
                  className={`receipt-card ${selectedReceipt && selectedReceipt._id === receipt._id ? "selected" : ""}`}
                  onClick={() => setSelectedReceipt(receipt)}
                >
                  <div className="receipt-header">
                    <h3>Receipt #{receipt._id.slice(-6)}</h3>
                    <span className={`status ${getStatusColor(receipt.status)}`}>
                      {receipt.status}
                    </span>
                  </div>
                  <div className="receipt-info">
                    <p><strong>Student ID:</strong> {receipt.student_id}</p>
                    <p><strong>Event ID:</strong> {receipt.event_id}</p>
                    <p><strong>Transaction:</strong> {receipt.transaction_id}</p>
                    <p><strong>Submitted:</strong> {new Date(receipt.submitted_at).toLocaleDateString()}</p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-state">
              <p>No receipts found.</p>
            </div>
          )}
        </section>

        {selectedReceipt && (
          <section className="receipt-detail">
            <h2>Receipt Details</h2>
            <div className="detail-content">
              <div className="detail-info">
                <p><strong>Receipt ID:</strong> {selectedReceipt._id}</p>
                <p><strong>Student ID:</strong> {selectedReceipt.student_id}</p>
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

              {selectedReceipt.status === "pending" && (
                <div className="verification-actions">
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
                  <button
                    className="btn cancel"
                    onClick={() => setSelectedReceipt(null)}
                  >
                    Cancel
                  </button>
                </div>
              )}
            </div>
          </section>
        )}
      </div>

      <div className="toolbar">
        <button className="btn btn-secondary" onClick={onBack}>
          Back to Dashboard
        </button>
      </div>
    </div>
  );
}

export default AdminReceiptVerification;
