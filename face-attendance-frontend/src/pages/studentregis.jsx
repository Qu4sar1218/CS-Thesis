import React, { useState, useRef, useEffect } from "react";
import "../styles/RegisterStudent.css";
import "../styles/WebcamModal.css";

export default function StudentRegis({ onBack }) {
  const [formData, setFormData] = useState({
    firstName: "",
    middleName: "",
    lastName: "",
    birthday: "",
    address: "",
    guardianContact: "",
    course: "",
    yearLevel: "",
    section: "",
  });

  const [message, setMessage] = useState("");
  const [studentId, setStudentId] = useState("");
  const [showWebcam, setShowWebcam] = useState(false);
  const [capturedImages, setCapturedImages] = useState([]);
  const [trainingStatus, setTrainingStatus] = useState("");
  const [isTraining, setIsTraining] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    const { firstName, lastName, birthday, course } = formData;

    // Basic validation
    if (!firstName || !lastName || !birthday || !course) {
      setMessage("⚠️ Please fill out all required fields before submitting.");
      return;
    }

    // Generate student ID (you can replace this with your own logic)
    const generatedId = `STU${Date.now()}`;
    setStudentId(generatedId);

    setMessage(
      `✅ Student "${firstName} ${lastName}" registered successfully! Student ID: ${generatedId}. Now train face recognition data.`
    );

    // Don't reset form yet - keep data for face training
  };

  const startWebcam = async () => {
    if (!studentId) {
      setMessage("⚠️ Please register the student first before training face data.");
      return;
    }

    setCameraReady(false);
    setShowWebcam(true);
    setTrainingStatus("🔄 Initializing camera...");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user"
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
            setTrainingStatus("📸 Position your face in the camera. Capture 5-10 images from different angles.");
          }).catch((err) => {
            console.error("Error playing video:", err);
            setTrainingStatus("❌ Error starting video playback");
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
      setTrainingStatus("⚠️ Camera not ready yet, please wait...");
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
        setCapturedImages((prev) => [...prev, blob]);
        const newCount = capturedImages.length + 1;
        setTrainingStatus(
          `✅ Captured ${newCount} image(s). ${
            newCount >= 5
              ? "You can now submit for training!"
              : `Capture ${5 - newCount} more.`
          }`
        );
      }
    }, "image/jpeg", 0.95);
  };

  const sendImagesToBackend = async () => {
    if (capturedImages.length === 0) {
      setTrainingStatus("⚠️ Please capture at least one image first.");
      return;
    }

    setIsTraining(true);
    setTrainingStatus("🔄 Uploading and training face data...");

    let successCount = 0;
    let failCount = 0;

    try {
      // Send each image to the backend
      for (let i = 0; i < capturedImages.length; i++) {
        const formData = new FormData();
        formData.append("student_id", studentId);
        formData.append("image", capturedImages[i], `face_${i + 1}.jpg`);

        try {
          const response = await fetch("http://localhost:5000/train-face", {
            method: "POST",
            body: formData,
          });

          const result = await response.json();

          if (response.ok) {
            successCount++;
            setTrainingStatus(
              `✅ Processing image ${i + 1}/${capturedImages.length}...`
            );
          } else {
            failCount++;
            console.error(`Image ${i + 1} failed:`, result.error);
          }
        } catch (error) {
          failCount++;
          console.error(`Error uploading image ${i + 1}:`, error);
        }
      }

      // Final status
      if (successCount > 0) {
        setTrainingStatus(
          `🎉 Training complete! ${successCount} image(s) processed successfully. ${
            failCount > 0 ? `${failCount} failed.` : ""
          }`
        );
        setMessage(
          `✅ Face recognition training completed for "${formData.firstName} ${formData.lastName}"!`
        );

        // Reset form after successful training
        setTimeout(() => {
          resetForm();
          stopWebcam();
        }, 3000);
      } else {
        setTrainingStatus(
          "❌ Training failed. Please ensure your face is clearly visible and try again."
        );
      }
    } catch (error) {
      setTrainingStatus("❌ Error during training: " + error.message);
    } finally {
      setIsTraining(false);
    }
  };

  const resetForm = () => {
    setFormData({
      firstName: "",
      middleName: "",
      lastName: "",
      birthday: "",
      address: "",
      guardianContact: "",
      course: "",
      yearLevel: "",
      section: "",
    });
    setStudentId("");
    setCapturedImages([]);
    setTrainingStatus("");
    setMessage("");
  };

  const handleFaceTraining = () => {
    if (!studentId) {
      setMessage("⚠️ Please register the student first before training face data.");
      return;
    }
    startWebcam();
  };

  // Cleanup webcam on unmount
  useEffect(() => {
    return () => {
      stopWebcam();
    };
  }, []);

  return (
    <div className="register-student">
      <h1>Register Student</h1>

      <form className="register-form" onSubmit={handleSubmit}>
        <label>
          First Name:
          <input
            type="text"
            name="firstName"
            value={formData.firstName}
            onChange={handleChange}
            placeholder="Enter first name"
            required
          />
        </label>

        <label>
          Middle Name:
          <input
            type="text"
            name="middleName"
            value={formData.middleName}
            onChange={handleChange}
            placeholder="Enter middle name"
          />
        </label>

        <label>
          Last Name:
          <input
            type="text"
            name="lastName"
            value={formData.lastName}
            onChange={handleChange}
            placeholder="Enter last name"
            required
          />
        </label>

        <label>
          Birthday:
          <input
            type="date"
            name="birthday"
            value={formData.birthday}
            onChange={handleChange}
            required
          />
        </label>

        <label>
          Address:
          <input
            type="text"
            name="address"
            value={formData.address}
            onChange={handleChange}
            placeholder="Enter address"
          />
        </label>

        <label>
          Guardian/Parent Contact (Phone or Email):
          <input
            type="text"
            name="guardianContact"
            value={formData.guardianContact}
            onChange={handleChange}
            placeholder="Enter contact info"
          />
        </label>

        <label>
          Course/Strand:
          <input
            type="text"
            name="course"
            value={formData.course}
            onChange={handleChange}
            placeholder="Enter course (e.g. BSIT)"
            required
          />
        </label>

        <label>
          Grade/Year Level:
          <input
            type="text"
            name="yearLevel"
            value={formData.yearLevel}
            onChange={handleChange}
            placeholder="Enter year level (e.g. 1st Year)"
          />
        </label>

        <label>
          Section:
          <input
            type="text"
            name="section"
            value={formData.section}
            onChange={handleChange}
            placeholder="Enter section"
          />
        </label>

        <div className="form-buttons">
          <button type="submit" className="primary">
            Register
          </button>
          <button type="button" className="secondary" onClick={onBack}>
            Back
          </button>
        </div>

        <button
          type="button"
          className="primary train-face-btn"
          onClick={handleFaceTraining}
          disabled={!studentId}
        >
          🧠 Train Face Recognition Data
        </button>
      </form>

      {message && <p className="message-text">{message}</p>}

      {/* Webcam Modal */}
      {showWebcam && (
        <div className="webcam-overlay">
          <div className="webcam-modal">
            <h2>Face Recognition Training</h2>
            <p className="student-id-display">
              Student ID: <strong>{studentId}</strong>
            </p>

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

            <p className="training-status">{trainingStatus}</p>

            <div className="captured-count">
              <strong>Captured: {capturedImages.length} image(s)</strong>
            </div>

            <div className="modal-buttons">
              <button
                onClick={captureImage}
                disabled={isTraining || !cameraReady}
                className="btn-capture"
              >
                📸 Capture Image
              </button>

              <button
                onClick={sendImagesToBackend}
                disabled={capturedImages.length === 0 || isTraining}
                className="btn-submit"
              >
                {isTraining ? "⏳ Training..." : "✅ Submit for Training"}
              </button>

              <button
                onClick={() => {
                  stopWebcam();
                  setCapturedImages([]);
                  setTrainingStatus("");
                }}
                disabled={isTraining}
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