import React, { useState, useRef, useEffect, useCallback } from "react";
import axios from "axios";
import "../styles/StudentReceiptSubmission.css";
import "../styles/WebcamModal.css";

function StudentReceiptSubmission({ studentId, onBack }) {
  const [selectedEvent, setSelectedEvent] = useState("");
  const [transactionId, setTransactionId] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [message, setMessage] = useState("");
  const [capturedImages, setCapturedImages] = useState([]);
  const [showWebcam, setShowWebcam] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [captureStatus, setCaptureStatus] = useState("");
  const [events, setEvents] = useState([]);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:8000";

  const fetchEvents = useCallback(async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/events`);
      setEvents(response.data.events || []);
    } catch (error) {
      console.error("Error fetching events:", error);
      setMessage("Failed to load events. Please try again.");
    }
  }, [BACKEND_URL]);

  // Fetch events on component mount
  useEffect(() => {
    fetchEvents();
  }, [fetchEvents]);

  // Cleanup webcam on unmount
  useEffect(() => {
    return () => {
      stopWebcam();
    };
  }, []);

  const startWebcam = async () => {
    setCameraReady(false);
    setShowWebcam(true);
    setCaptureStatus("🔄 Initializing camera...");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user" // Use front camera (same as face recognition)
        },
        audio: false
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;

        // Wait for video to be ready
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play().then(() => {
            setCameraReady(true);
            setCaptureStatus("📸 Position your receipt in the camera and capture a clear photo.");
          }).catch((err) => {
            console.error("Error playing video:", err);
            setCaptureStatus("❌ Error starting video playback");
          });
        };
      }

      setCapturedImages([]);
    } catch (error) {
      console.error("Webcam error:", error);
      setMessage("❌ Error accessing webcam: " + error.message);
      setShowWebcam(false);
    }
  };

  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setShowWebcam(false);
    setCameraReady(false);
  };

  const captureImage = () => {
    if (!videoRef.current || !canvasRef.current || !cameraReady) {
      setCaptureStatus("⚠️ Camera not ready yet, please wait...");
      return;
    }

    const canvas = canvasRef.current;
    const video = videoRef.current;

    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas to blob
    canvas.toBlob((blob) => {
      if (blob) {
        setCapturedImages([blob]); // Only need one image for receipt
        setCaptureStatus("✅ Receipt captured successfully! You can now close the camera and submit.");
      }
    }, "image/jpeg", 0.95);
  };

  const retakePhoto = () => {
    setCapturedImages([]);
    setCaptureStatus("📸 Position your receipt in the camera and capture a clear photo.");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!selectedEvent || !transactionId || capturedImages.length === 0) {
      setMessage("Please fill in all fields and take a photo of your receipt.");
      return;
    }

    setIsSubmitting(true);
    setMessage("");

    try {
      // Convert image to base64
      const reader = new FileReader();
      reader.onload = async (e) => {
        const base64Image = e.target.result;

        const receiptData = {
          student_id: studentId,
          event_id: selectedEvent,
          transaction_id: transactionId,
          receipt_image: base64Image
        };

        try {
          await axios.post(`${BACKEND_URL}/receipts`, receiptData);
          setMessage("Receipt submitted successfully! It will be reviewed by an administrator.");
          // Reset form
          setSelectedEvent("");
          setTransactionId("");
          setCapturedImages([]);
          stopWebcam();
        } catch (error) {
          console.error("Error submitting receipt:", error);
          if (error.response && error.response.data && error.response.data.detail) {
            setMessage(error.response.data.detail);
          } else {
            setMessage("Failed to submit receipt. Please try again.");
          }
        } finally {
          setIsSubmitting(false);
        }
      };
      reader.readAsDataURL(capturedImages[0]);
    } catch (error) {
      console.error("Error processing image:", error);
      setMessage("Failed to process image. Please try again.");
      setIsSubmitting(false);
    }
  };

  return (
    <div className="receipt-submission">
      <header className="submission-header">
        <button className="back-btn" onClick={onBack}>← Back to Dashboard</button>
        <h1>Submit Receipt</h1>
        <p>Enter your transaction ID and take a photo of your receipt for verification.</p>
      </header>

      <form onSubmit={handleSubmit} className="submission-form">
        <div className="form-group">
          <label htmlFor="event">Select Event:</label>
          <select
            id="event"
            value={selectedEvent}
            onChange={(e) => setSelectedEvent(e.target.value)}
            required
          >
            <option value="">Choose an event...</option>
            {events.map((event) => (
              <option key={event._id} value={event._id}>
                {event.name} - {event.date}
              </option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="transactionId">Transaction ID (6 digits):</label>
          <input
            type="text"
            id="transactionId"
            value={transactionId}
            onChange={(e) => {
              const value = e.target.value.replace(/\D/g, '').slice(0, 6);
              setTransactionId(value);
            }}
            placeholder="Enter 6-digit transaction ID"
            pattern="\d{6}"
            title="Transaction ID must be exactly 6 digits"
            required
          />
        </div>

        <div className="form-group">
          <label>Receipt Photo:</label>
          <button
            type="button"
            onClick={startWebcam}
            className="camera-btn"
            disabled={capturedImages.length > 0}
          >
            📷 Take Photo
          </button>
          {capturedImages.length > 0 && (
            <div className="image-preview">
              <p>✅ Receipt captured successfully!</p>
              <button
                type="button"
                onClick={retakePhoto}
                className="retake-btn"
              >
                🔄 Retake Photo
              </button>
            </div>
          )}
        </div>

        {message && (
          <div className={`message ${message.includes("successfully") ? "success" : "error"}`}>
            {message}
          </div>
        )}

        <button
          type="submit"
          disabled={isSubmitting}
          className="submit-btn"
        >
          {isSubmitting ? "Submitting..." : "Submit Receipt"}
        </button>
      </form>

      {/* Webcam Modal */}
      {showWebcam && (
        <div className="webcam-overlay">
          <div className="webcam-modal">
            <h2>Receipt Photo Capture</h2>

            <div className="video-container">
              {!cameraReady && (
                <div className="camera-loading">
                  <p>🔄 Starting camera...</p>
                </div>
              )}
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="webcam-video"
              />
              <canvas ref={canvasRef} className="hidden-canvas" />
            </div>

            <p className="capture-status">{captureStatus}</p>

            <div className="modal-buttons">
              <button
                onClick={captureImage}
                disabled={!cameraReady || capturedImages.length > 0}
                className="btn-capture"
              >
                📸 Capture Receipt
              </button>

              <button
                onClick={() => {
                  stopWebcam();
                  setCaptureStatus("");
                }}
                disabled={capturedImages.length === 0}
                className="btn-save"
              >
                💾 Save Photo
              </button>

              <button
                onClick={() => {
                  stopWebcam();
                  setCapturedImages([]);
                  setCaptureStatus("");
                }}
                className="btn-cancel"
              >
                ❌ Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default StudentReceiptSubmission;
