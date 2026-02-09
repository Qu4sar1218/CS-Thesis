import React, { useState, useRef, useEffect } from "react";
import "../styles/RegisterStudent.css";
import "../styles/WebcamModal.css";

const API_BASE_URL = "http://127.0.0.1:8000";

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

  const [courseOptions, setCourseOptions] = useState([]);
  const [allCourses, setAllCourses] = useState([]);
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
    if (name === 'yearLevel') {
      setFormData((prev) => ({ ...prev, course: '' }));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const { firstName, middleName, lastName, birthday, course, yearLevel, address, guardianContact, section } = formData;

    // Basic validation
    if (!firstName || !lastName || !birthday || !course) {
      setMessage("⚠️ Please fill out all required fields before submitting.");
      return;
    }

    // Generate student ID in format 11XXXX (11 + 4 random digits)
    const randomDigits = Math.floor(1000 + Math.random() * 9000).toString();
    const generatedId = `11${randomDigits}`;

    try {
      const studentData = {
        student_id: generatedId,
        first_name: firstName,
        last_name: lastName,
        middle_name: middleName || null,
        email: "", // Not collected in form
        course: course,
        year: yearLevel,
        // Optional fields
        ...(address && { address }),
        ...(guardianContact && { guardian_contact: guardianContact }),
        ...(section && { section })
      };

      const response = await fetch(`${API_BASE_URL}/students`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(studentData),
      });

      const result = await response.json();

      if (response.ok) {
        setStudentId(generatedId);
        const displayName = middleName ? `${firstName} ${middleName.charAt(0)}. ${lastName}` : `${firstName} ${lastName}`;
        setMessage(
          `✅ Student "${displayName}" registered successfully! Student ID: STU ${generatedId}. Now train face recognition data.`
        );
      } else {
        setMessage(`❌ Registration failed: ${result.detail || result.error}`);
      }
    } catch (error) {
      setMessage(`❌ Error registering student: ${error.message}`);
    }
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
      // Send each image to the face training server
      for (let i = 0; i < capturedImages.length; i++) {
        const formData = new FormData();
        formData.append("student_id", studentId);
        formData.append("image", capturedImages[i], `face_${i + 1}.jpg`);

        try {
          const response = await fetch(`http://127.0.0.1:5000/train-face`, {
            method: "POST",
            body: formData,
          });

          const result = await response.json();

          if (response.ok && result.encodings_saved) {
            successCount++;
            setTrainingStatus(
              `✅ Processing image ${i + 1}/${capturedImages.length}...`
            );
          } else {
            failCount++;
            console.error(`Image ${i + 1} failed:`, result.error || result.detail);
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

  const fetchCourses = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/classes/courses`);
      if (!response.ok) {
        throw new Error('Failed to fetch courses');
      }
      const data = await response.json();
      setAllCourses(data.courses); // Now contains objects with code, name, level
    } catch (err) {
      console.error('Error fetching courses:', err);
    }
  };

  // Fetch courses on component mount
  useEffect(() => {
    fetchCourses();
  }, []);

  // Update course options based on year level and fetched courses
  useEffect(() => {
    if (allCourses.length > 0) {
      if (formData.yearLevel === 'Grade 11' || formData.yearLevel === 'Grade 12') {
        // For SHS, filter strands by level
        setCourseOptions(allCourses.filter(course => course.level === 'senior_high'));
      } else if (formData.yearLevel.includes('Year College')) {
        // For College, filter courses by level
        setCourseOptions(allCourses.filter(course => course.level === 'college'));
      } else {
        setCourseOptions([]);
      }
    }
  }, [formData.yearLevel, allCourses]);

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
          Middle Name (Optional):
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
          Grade/Year Level:
          <select
            name="yearLevel"
            value={formData.yearLevel}
            onChange={handleChange}
            required
          >
            <option value="">Select Year Level</option>
            <option value="Grade 11">Grade 11</option>
            <option value="Grade 12">Grade 12</option>
            <option value="1st Year College">1st Year College</option>
            <option value="2nd Year College">2nd Year College</option>
            <option value="3rd Year College">3rd Year College</option>
            <option value="4th Year College">4th Year College</option>
          </select>
        </label>

        <label>
          Course/Strand:
          <select
            name="course"
            value={formData.course}
            onChange={handleChange}
            required
            disabled={!formData.yearLevel}
          >
            <option value="">Select Course/Strand</option>
            {courseOptions.map((course) => (
              <option key={course.code} value={course.code}>
                {course.name}
              </option>
            ))}
          </select>
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
              Student ID: <strong>STU {studentId}</strong>
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
              <div className="oval-guide"></div>
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