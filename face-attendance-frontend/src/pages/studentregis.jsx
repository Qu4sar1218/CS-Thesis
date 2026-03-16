import React, { useState, useRef, useEffect } from "react";
import "../styles/RegisterStudent.css";
import "../styles/WebcamModal.css";

const API_BASE_URL = "http://localhost:8000";

// Valid positions for face training
const FACE_POSITIONS = [
  { id: 'front', label: 'Center (Front)', guidance: 'Look straight at the camera' },
  { id: 'left', label: 'Left', guidance: 'Turn your face to the left' },
  { id: 'right', label: 'Right', guidance: 'Turn your face to the right' },
  { id: 'up', label: 'Up', guidance: 'Look up towards the camera' },
  { id: 'down', label: 'Down', guidance: 'Look down' }
];

// Required positions for complete registration
const REQUIRED_POSITIONS = ['front', 'left', 'right', 'up', 'down'];

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
  const [showWebcam, setShowWebcam] = useState(false);
  const [capturedImages, setCapturedImages] = useState([]);
  const [trainingStatus, setTrainingStatus] = useState("");
  const [isTraining, setIsTraining] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [studentId, setStudentId] = useState("");
  
  // Popup modal state
  const [isPopupVisible, setIsPopupVisible] = useState(false);
  const [popupTitle, setPopupTitle] = useState("");
  const [popupMessage, setPopupMessage] = useState("");
  const [popupType, setPopupType] = useState("error"); // "error" or "success"
  
  // Position-based training state
  const [currentPosition, setCurrentPosition] = useState('front');
  const [capturedPositions, setCapturedPositions] = useState({
    front: false,
    left: false,
    right: false,
    up: false,
    down: false
  });
  
  // Track images per position
  const [positionImageCounts, setPositionImageCounts] = useState({
    front: 0,
    left: 0,
    right: 0,
    up: 0,
    down: 0
  });

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

  // Helper function to show popup
  const displayPopup = (title, message, type = "error") => {
    setPopupTitle(title);
    setPopupMessage(message);
    setPopupType(type);
    setIsPopupVisible(true);
  };

  const closePopup = () => {
    setIsPopupVisible(false);
  };

  // Start face training with generated student ID
  const startFaceTraining = () => {
    // Generate student ID first (required for face training)
    const randomDigits = Math.floor(1000 + Math.random() * 9000).toString();
    const generatedId = `11${randomDigits}`;
    setStudentId(generatedId);
    
    // Now start webcam for face training
    startWebcam();
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const { firstName, middleName, lastName, birthday, course, yearLevel, address, guardianContact, section } = formData;

    // Basic validation
    if (!firstName || !lastName || !birthday || !course) {
      setMessage("⚠️ Please fill out all required fields before submitting.");
      return;
    }

    // Generate student ID if not already generated
    let finalStudentId = studentId;
    if (!finalStudentId) {
      const randomDigits = Math.floor(1000 + Math.random() * 9000).toString();
      finalStudentId = `11${randomDigits}`;
      setStudentId(finalStudentId);
    }

    // CHECK IF FACE TRAINING IS COMPLETE - REQUIRED FOR REGISTRATION
    const allPositionsCaptured = REQUIRED_POSITIONS.every(
      pos => capturedPositions[pos] && positionImageCounts[pos] > 0
    );
    
    if (!allPositionsCaptured) {
      const missingPositions = REQUIRED_POSITIONS.filter(
        pos => !capturedPositions[pos] || positionImageCounts[pos] === 0
      );
      displayPopup("Face Training Required", `Please capture all 5 positions before registration: ${missingPositions.join(", ")}`, "error");
      return; // BLOCK REGISTRATION
    }

    if (capturedImages.length === 0) {
      displayPopup("No Face Images", "Please capture face images before registration.", "error");
      return; // BLOCK REGISTRATION
    }

      // Upload face images and verify they are valid
    setIsTraining(true);
    setTrainingStatus("🔄 Uploading and validating face data...");
    
    let successCount = 0;
    let failedPositions = [];
    
    try {
      for (let i = 0; i < capturedImages.length; i++) {
        const formDataImg = new FormData();
        const imageData = capturedImages[i];
        const position = imageData.position || 'front';
        const blob = imageData.blob || imageData;
        formDataImg.append("image", blob, `face_${i + 1}.jpg`);
        formDataImg.append("position", position);
        
        const uploadUrl = `${API_BASE_URL}/students/${finalStudentId}/face-encodings`;
        console.log(`Uploading to: ${uploadUrl}, position: ${position}`);
        
        try {
          const response = await fetch(uploadUrl, {
            method: "POST",
            body: formDataImg,
          });

          const result = await response.json();
          console.log(`Upload response for ${position}:`, response.status, result);
          
          // Consider success if response is OK (status 200-299)
          // Even if face detection fails, the image is saved
          if (response.ok) {
            successCount++;
            console.log(`Successfully uploaded image ${i + 1} for position ${position}`);
          } else {
            failedPositions.push(`${position}: ${result.detail || result.error || 'Server error'}`);
          }
        } catch (error) {
          console.error(`Network error uploading image ${i + 1}:`, error);
          failedPositions.push(`${position}: Network error - ${error.message}`);
        }
      }

      // Check if any uploads succeeded
      if (successCount === 0) {
        displayPopup("Upload Failed", `All face image uploads failed. Please check if the backend server is running on port 8000.`, "error");
        setIsTraining(false);
        return; // BLOCK REGISTRATION
      }

      // Even if some failed, we can still proceed if at least some images were saved
      if (failedPositions.length > 0) {
        setTrainingStatus(`⚠️ ${successCount}/${capturedImages.length} images uploaded. Some positions may need recapture.`);
      } else {
        setTrainingStatus(`✅ All ${successCount} face images uploaded successfully!`);
      }

    } catch (error) {
      console.error("Error uploading face images:", error);
      displayPopup("Upload Error", `Error uploading face images: ${error.message}`, "error");
      setIsTraining(false);
      return; // BLOCK REGISTRATION
    }

    // Only proceed to registration if face training is complete and valid
    // Now create the student
    try {
      const studentData = {
        student_id: finalStudentId,
        first_name: firstName,
        last_name: lastName,
        middle_name: middleName || null,
        email: "",
        course: course,
        year: yearLevel,
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
        displayPopup("Registration Successful!", `Student "${firstName} ${lastName}" registered successfully! Student ID: STU ${finalStudentId}.`, "success");
        // Reset form
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
        setCapturedImages([]);
        setCapturedPositions({
          front: false,
          left: false,
          right: false,
          up: false,
          down: false
        });
        setPositionImageCounts({
          front: 0,
          left: 0,
          right: 0,
          up: 0,
          down: 0
        });
        setStudentId("");
      } else {
        setMessage(`❌ Registration failed: ${result.detail || result.error}`);
      }
    } catch (error) {
      setMessage(`❌ Error registering student: ${error.message}`);
    }
  };

  const startWebcam = async () => {
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

        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play().then(() => {
            setCameraReady(true);
            setTrainingStatus("📸 Position your face in the camera. Capture images from all 5 positions.");
          }).catch((err) => {
            console.error("Error playing video:", err);
            setTrainingStatus("❌ Error starting video playback");
          });
        };
      }

      setCapturedImages([]);
      // Reset position tracking
      setCapturedPositions({
        front: false,
        left: false,
        right: false,
        up: false,
        down: false
      });
      setPositionImageCounts({
        front: 0,
        left: 0,
        right: 0,
        up: 0,
        down: 0
      });
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

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
      if (blob) {
        // Store blob with position info as an object
        setCapturedImages((prev) => [...prev, { blob, position: currentPosition }]);
        
        const newCount = positionImageCounts[currentPosition] + 1;
        setPositionImageCounts(prev => ({
          ...prev,
          [currentPosition]: newCount
        }));
        
        if (newCount === 1) {
          setCapturedPositions(prev => ({
            ...prev,
            [currentPosition]: true
          }));
        }
        
        const totalCaptured = Object.values(positionImageCounts).reduce((a, b) => a + b, 0) + 1;
        setTrainingStatus(
          `✅ Captured ${currentPosition} (${newCount} img). Total: ${totalCaptured}/5 positions. ${
            totalCaptured >= 5
              ? "All positions captured!"
              : `Next: ${REQUIRED_POSITIONS.filter(p => !capturedPositions[p] || positionImageCounts[p] === 0).join(", ")}`
          }`
        );
      }
    }, "image/jpeg", 0.95);
  };

  const handleFaceTraining = () => {
    startFaceTraining();
  };

  const finishFaceTraining = () => {
    // Check if all positions are captured
    const allPositionsCaptured = REQUIRED_POSITIONS.every(
      pos => capturedPositions[pos] && positionImageCounts[pos] > 0
    );
    
    if (!allPositionsCaptured) {
      const missingPositions = REQUIRED_POSITIONS.filter(
        pos => !capturedPositions[pos] || positionImageCounts[pos] === 0
      );
      setTrainingStatus(`⚠️ Face training incomplete. Please capture all positions. Missing: ${missingPositions.join(", ")}`);
      setMessage(`⚠️ Face training incomplete. Please finish face capture.`);
      return;
    }

    stopWebcam();
    setTrainingStatus("✅ Face training complete! You can now register the student.");
  };

  const fetchCourses = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/courses`);
      if (!response.ok) {
        throw new Error('Failed to fetch courses');
      }
      const data = await response.json();
      setAllCourses(data.courses);
    } catch (err) {
      console.error('Error fetching courses:', err);
    }
  };

  useEffect(() => {
    fetchCourses();
  }, []);

  useEffect(() => {
    if (allCourses.length > 0) {
      if (formData.yearLevel === 'Grade 11' || formData.yearLevel === 'Grade 12') {
        setCourseOptions(allCourses.filter(course => course.level === 'senior_high'));
      } else if (formData.yearLevel.includes('Year College')) {
        setCourseOptions(allCourses.filter(course => course.level === 'college'));
      } else {
        setCourseOptions([]);
      }
    }
  }, [formData.yearLevel, allCourses]);

  useEffect(() => {
    return () => {
      stopWebcam();
    };
  }, []);

  return (
    <div className="register-student">
      <h1>Register Student</h1>

      {/* Student Registration Form - Always Visible */}
      <form className="register-form" onSubmit={handleSubmit}>
        {/* Student ID Display (if generated) */}
        {studentId && (
          <div className="student-id-display">
            <span className="student-id-label">Student ID:</span>
            <span className="student-id-value">STU {studentId}</span>
          </div>
        )}
        
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
          <button 
            type="submit" 
            className="primary"
            disabled={!studentId || capturedImages.length === 0}
            title={!studentId || capturedImages.length === 0 ? "Please complete face training first" : "Register Student"}
          >
            {(!studentId || capturedImages.length === 0) ? "🔒 Complete Face Training First" : "Register Student"}
          </button>
          <button type="button" className="secondary" onClick={onBack}>
            Back
          </button>
        </div>
      </form>

      {/* Face Training Section - Also Visible */}
      <div className="face-training-section">
        <div className="training-prompt">
          <div className="training-icon">📸</div>
          <h2>Face Recognition Training</h2>
          <p>Capture your face from 5 different angles for attendance recognition.</p>
          <p className="training-instructions">
            Required positions: <strong>Center, Left, Right, Up, Down</strong>
          </p>
          {!showWebcam ? (
            <button
              type="button"
              className="primary train-face-btn"
              onClick={handleFaceTraining}
            >
              📸 {studentId ? "Continue Face Training" : "Start Face Training"}
            </button>
          ) : null}
        </div>
        
        {/* Show training status and progress if started */}
        {(trainingStatus || showWebcam) && (
          <div className="training-status-container">
            {trainingStatus && !showWebcam && (
              <p className={`training-status ${trainingStatus.includes('complete') ? 'success' : ''}`}>
                {trainingStatus}
              </p>
            )}
          </div>
        )}
      </div>

      {message && <p className="message-text">{message}</p>}

      {/* Webcam Modal */}
      {showWebcam && (
        <div className="webcam-overlay">
          <div className="webcam-modal">
            <h2>Face Recognition Training</h2>

            {/* Position Selector */}
            <div className="position-selector">
              <p className="position-label">Select Face Position:</p>
              <div className="position-buttons">
                {FACE_POSITIONS.map((pos) => (
                  <button
                    key={pos.id}
                    type="button"
                    className={`position-btn ${currentPosition === pos.id ? 'active' : ''} ${capturedPositions[pos.id] ? 'captured' : ''}`}
                    onClick={() => setCurrentPosition(pos.id)}
                    disabled={isTraining}
                  >
                    {pos.label}
                    {capturedPositions[pos.id] && positionImageCounts[pos.id] > 0 && (
                      <span className="position-check">✓</span>
                    )}
                  </button>
                ))}
              </div>
              <p className="position-guidance">
                {FACE_POSITIONS.find(p => p.id === currentPosition)?.guidance}
              </p>
            </div>

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
                className="webcam-video mirrored"
              />
              <div className="oval-guide"></div>
              <canvas ref={canvasRef} className="hidden-canvas" />
            </div>

            <p className="training-status">{trainingStatus}</p>

            {/* Position Progress */}
            <div className="captured-count">
              <strong>Position Progress:</strong>
              <div className="position-progress">
                {FACE_POSITIONS.map((pos) => (
                  <div key={pos.id} className={`progress-item ${capturedPositions[pos.id] ? 'completed' : ''}`}>
                    <span className="progress-label">{pos.label}</span>
                    <span className="progress-count">
                      {positionImageCounts[pos.id] > 0 ? `${positionImageCounts[pos.id]} img` : '-'}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div className="modal-buttons">
              <button
                onClick={captureImage}
                disabled={isTraining || !cameraReady}
                className="btn-capture"
              >
                📸 Capture {FACE_POSITIONS.find(p => p.id === currentPosition)?.label}
              </button>

              <button
                onClick={finishFaceTraining}
                disabled={capturedImages.length === 0 || isTraining}
                className="btn-submit"
              >
                {isTraining ? "⏳ Processing..." : "✅ Done - Continue to Registration"}
              </button>

              <button
                onClick={() => {
                  stopWebcam();
                  setCapturedImages([]);
                  setTrainingStatus("");
                  setCapturedPositions({
                    front: false,
                    left: false,
                    right: false,
                    up: false,
                    down: false
                  });
                  setPositionImageCounts({
                    front: 0,
                    left: 0,
                    right: 0,
                    up: 0,
                    down: 0
                  });
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
      {/* Popup Modal */}
      {isPopupVisible && (
        <div className="popup-overlay">
          <div className={`popup-modal ${popupType}`}>
            <div className="popup-header">
              <h3>{popupType === 'error' ? '⚠️ Error' : '✅ Success'}</h3>
              <button className="popup-close" onClick={closePopup}>×</button>
            </div>
            <div className="popup-content">
              <h4>{popupTitle}</h4>
              <p>{popupMessage}</p>
            </div>
            <div className="popup-footer">
              <button className="popup-btn" onClick={closePopup}>OK</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

