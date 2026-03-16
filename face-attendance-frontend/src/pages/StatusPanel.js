import React, { useEffect, useState, useRef, useCallback } from "react";
import axios from "axios";
import "../styles/StatusPanel.css";

function StatusPanel({ mode, subject }) {
  const [status, setStatus] = useState({});
  const [streamConnected, setStreamConnected] = useState(false);
  const [isStreamLoading, setIsStreamLoading] = useState(true);
  const [isRecognitionActive, setIsRecognitionActive] = useState(false);
  const [imageSrc, setImageSrc] = useState(null);
  const [useSnapshotFallback, setUseSnapshotFallback] = useState(false);
  const [recentlyRecognized, setRecentlyRecognized] = useState(null);
  const [recognizedPaymentStatus, setRecognizedPaymentStatus] = useState(null);
  
  // Confirmation overlay for events mode
  const [showConfirmation, setShowConfirmation] = useState(false);
  const confirmationTimeoutRef = useRef(null);
  const previousRecognizedKeyRef = useRef(null);

  // Camera configuration
  const cameraIndex = 0;

  // Attendance mode (IN/OUT)
  const [attendanceMode, setAttendanceMode] = useState("IN");
  const [isSwitchingMode, setIsSwitchingMode] = useState(false);
  const [message, setMessage] = useState(null);
  const [messageType, setMessageType] = useState("");

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:8000";
  
  const snapshotIntervalRef = useRef(null);
  const statusIntervalRef = useRef(null);
  const streamTimeoutRef = useRef(null);
  const cameraStatusIntervalRef = useRef(null);
  const imageSrcRef = useRef("");
  const snapshotEndpointRef = useRef("/snapshot");
  const snapshot404CountRef = useRef(0);
  const mjpegRetryCountRef = useRef(0);
  const streamStartTimeoutMs = 10000; // Reduced from 25000ms for faster navigation
  const maxMjpegRetries = 5;

  const normalizeRecognized = useCallback((raw) => {
    if (!raw) return null;
    const firstName = raw.first_name || raw.firstName || "";
    const middleName = raw.middle_name || raw.middleName || "";
    const lastName = raw.last_name || raw.lastName || "";
    const nameFromParts = [firstName, middleName, lastName].filter(Boolean).join(" ").trim();
    return {
      name: raw.name || raw.student_name || nameFromParts || "Unknown Student",
      student_id: raw.student_id || raw.studentId || raw.studentID || "",
      event_id: raw.event_id || raw.eventId || null,
      time: raw.time || raw.scan_time || raw.check_in_time || raw.check_out_time || "",
      date: raw.date || raw.scan_date || "",
      course: raw.course || raw.program || "",
      year: raw.year || raw.year_level || "",
      status: raw.status || raw.attendance_status || "",
      message: raw.message || raw.reason || "",
      attendance_mode: raw.attendance_mode || raw.attendanceMode || "",
      attendance_type: raw.attendance_type || raw.attendanceType || ""
    };
  }, []);

  const updateImageSrc = (url) => {
    try {
      if (imageSrcRef.current && imageSrcRef.current.startsWith && 
          imageSrcRef.current.startsWith('blob:') && imageSrcRef.current !== url) {
        URL.revokeObjectURL(imageSrcRef.current);
      }
    } catch (e) {
      // ignore cleanup errors
    }
    imageSrcRef.current = url;
    setImageSrc(url || null);
  };

  // Fetch attendance mode from backend
  const fetchAttendanceMode = useCallback(async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/attendance-mode`);
      setAttendanceMode(response.data.mode || "IN");
    } catch (err) {
      console.error("Error fetching attendance mode:", err);
    }
  }, [BACKEND_URL]);

  // Switch attendance mode
  const switchAttendanceMode = useCallback(async () => {
    const newMode = attendanceMode === "IN" ? "OUT" : "IN";
    setIsSwitchingMode(true);
    try {
      await axios.post(`${BACKEND_URL}/attendance-mode`, { mode: newMode });
      setAttendanceMode(newMode);
      setMessage(`Switched to ${newMode} mode`);
      setMessageType("success");
      setTimeout(() => setMessage(null), 3000);
    } catch (err) {
      console.error("Error switching attendance mode:", err);
      setMessage("Failed to switch mode");
      setMessageType("error");
      setTimeout(() => setMessage(null), 3000);
    } finally {
      setIsSwitchingMode(false);
    }
  }, [attendanceMode, BACKEND_URL]);

  const fetchPaymentStatuses = useCallback(async (attendanceRecords, currentEventId) => {
    if (mode !== 'events' || !currentEventId) return;

    const statuses = {};
    for (const record of attendanceRecords) {
      try {
        const response = await axios.get(`${BACKEND_URL}/students/${record.student_id}/payment-status/${currentEventId}`);
        // Store the full response to access receipt_status
        statuses[record.student_id] = response.data;
      } catch (err) {
        console.error(`Error fetching payment status for ${record.student_id}:`, err);
        statuses[record.student_id] = { paid: false, receipt_status: "not_found" };
      }
    }
    setPaymentStatuses(statuses);
  }, [BACKEND_URL, mode]);

  // Fetch payment status for the recently recognized student in events mode
  const fetchRecognizedPaymentStatus = useCallback(async (studentId, eventId) => {
    if (mode !== 'events' || !studentId || !eventId) return;
    
    try {
      const response = await axios.get(`${BACKEND_URL}/students/${studentId}/payment-status/${eventId}`);
      setRecognizedPaymentStatus(response.data);
    } catch (err) {
      console.error(`Error fetching payment status for recognized student ${studentId}:`, err);
      setRecognizedPaymentStatus({ paid: false });
    }
  }, [BACKEND_URL, mode]);

  

  const fetchStatusData = useCallback(async () => {
    try {
      // Run multiple API calls in parallel for better performance
      const [statusRes, attendanceRes, recognizedRes] = await Promise.allSettled([
        axios.get(`${BACKEND_URL}/status`),
        mode === "class" 
          ? axios.get(`${BACKEND_URL}/attendance/class/${subject?.classId || subject?.id}/today`)
          : axios.get(`${BACKEND_URL}/attendance`),
        axios.get(`${BACKEND_URL}/recently-recognized`)
      ]);

      const statusOk = statusRes.status === "fulfilled";
      let statusData = {};
      if (statusOk) {
        statusData = statusRes.value.data;
        setStatus(statusData);
        setIsRecognitionActive(Boolean(statusData.recognition_running));
      }

      // If recognition is already running (e.g., started elsewhere), auto-attach live feed.
      if (statusData?.recognition_running && !useSnapshotFallback) {
        const currentSrc = imageSrcRef.current || "";
        if (!currentSrc || currentSrc.startsWith("blob:")) {
          const streamUrl = `${BACKEND_URL}/video?t=${Date.now()}`;
          updateImageSrc(streamUrl);
          setIsStreamLoading(true);
        }
      }

      // Skip attendance fetch since UI removed

      let recognized = null;
      let recognitionKey = null;
      let previousRecognitionKey = previousRecognizedKeyRef.current;
      if (recognizedRes.status === "fulfilled") {
        const recognizedData = recognizedRes.value.data || {};
        const recognizedRaw =
          recognizedData.recently_recognized ||
          recognizedData.recentlyRecognized ||
          recognizedData.recent ||
          (recognizedData.student_id || recognizedData.studentId ? recognizedData : null);
        recognized = normalizeRecognized(recognizedRaw);
        recognitionKey = recognized
          ? `${recognized.student_id || ""}|${recognized.time || ""}|${recognized.date || ""}|${recognized.status || ""}|${recognized.message || ""}`
          : null;
        previousRecognizedKeyRef.current = recognitionKey;
        setRecentlyRecognized(recognized);
      }

      // Trigger confirmation overlay in events mode when a new student is recognized
      if (mode === 'events' && recognitionKey && recognitionKey !== previousRecognitionKey) {
        // Clear any existing timeout
        if (confirmationTimeoutRef.current) {
          clearTimeout(confirmationTimeoutRef.current);
        }
        setShowConfirmation(true);
        // Auto-dismiss after 3 seconds
        confirmationTimeoutRef.current = setTimeout(() => {
          setShowConfirmation(false);
        }, 3000);
      }

      // Fetch payment status for the recognized student in events mode
      const eventId = statusData?.current_event_id || subject?.id || recognized?.event_id;
      if (mode === 'events' && recognized?.student_id && eventId) {
        await fetchRecognizedPaymentStatus(recognized.student_id, eventId);
      }

      // Fetch attendance mode
      await fetchAttendanceMode();

      // Show message from recently recognized if any
      if (recognized?.message) {
        setMessage(recognized.message);
        setMessageType(recognized.status?.includes('already') || recognized.status === 'no_time_in' ? "error" : "success");
        setTimeout(() => setMessage(null), 5000);
      }

      // Payment statuses fetch removed (no UI usage)

      // Keep connection state in sync when recognition is stopped.
      if (statusOk && statusData && !statusData.recognition_running) {
        setStreamConnected(false);
        setIsStreamLoading(false);
      }
    } catch (err) {
      console.error("Error fetching status:", err);
    }
  }, [BACKEND_URL, mode, subject, fetchPaymentStatuses, fetchAttendanceMode, fetchRecognizedPaymentStatus, normalizeRecognized, useSnapshotFallback]);

  const configureMode = useCallback(async () => {
    if (mode === "class") {
      const classId = subject?.classId || subject?.id;
      if (!classId) {
        console.error("Missing class ID for class mode");
        setMessage("Error: No class selected. Please go back and select a class.");
        setMessageType("error");
        return true;
      }

      try {
        console.log("Configuring class mode for classId:", classId);
        
        // First, set the mode
        const modeRes = await axios.post(`${BACKEND_URL}/set-mode`, {
          mode: "class",
          event_id: classId,
        });
        console.log("Set mode response:", modeRes.data);
        
        // Check if class is not scheduled today (403 from backend)
        if (modeRes.data?.detail?.includes("not scheduled for today")) {
          console.warn("Class is not scheduled for today");
          setMessage("This class is not scheduled for today. Please select a class that is scheduled for today.");
          setMessageType("error");
          setTimeout(() => setMessage(null), 10000);
          return false;
        }
        
        // Initialize attendance for the class
        if (subject?.monitoring?.enabled && subject?.monitoring?.standalone) {
          console.log("Starting standalone monitoring mode");
          await axios.post(`${BACKEND_URL}/attendance/monitoring/standalone-start`, {
            class_id: classId,
            fallback_status: "ABSENT"
          });
        } else if (subject?.monitoring?.enabled && subject?.monitoring?.previousClassId) {
          console.log("Starting monitoring mode with previous class");
          await axios.post(`${BACKEND_URL}/attendance/monitoring/start`, {
            class_id: classId,
            previous_class_id: subject.monitoring.previousClassId,
            fallback_status: "ABSENT"
          });
        } else {
          console.log("Initializing regular class attendance");
          const initRes = await axios.post(`${BACKEND_URL}/attendance/initialize-class/${classId}`);
          console.log("Initialize class response:", initRes.data);
        }
        return true;
      } catch (err) {
        console.error("Failed to configure class mode:", err);
        // Show more specific error message
        const errorMessage = err.response?.data?.detail || err.message || "Unknown error";
        setMessage(`Failed to configure class mode: ${errorMessage}`);
        setMessageType("error");
        setTimeout(() => setMessage(null), 10000);
        return false;
      }
    }

    if (mode === "events") {
      try {
        await axios.post(`${BACKEND_URL}/set-mode`, {
          mode: "events",
          event_id: subject?.id || null,
        });
      } catch (err) {
        console.error("Failed to configure events mode:", err);
        setMessage(`Failed to configure events mode: ${err.response?.data?.detail || err.message}`);
        setMessageType("error");
        setTimeout(() => setMessage(null), 5000);
      }
    }

    if (mode === "hallway") {
      try {
        await axios.post(`${BACKEND_URL}/set-mode`, {
          mode: "hallway",
          event_id: null,
        });
        await axios.post(`${BACKEND_URL}/attendance/initialize-inout`);
      } catch (err) {
        console.error("Failed to configure hallway mode:", err);
        setMessage(`Failed to configure hallway mode: ${err.response?.data?.detail || err.message}`);
        setMessageType("error");
        setTimeout(() => setMessage(null), 5000);
      }
    }

    return true;
  }, [BACKEND_URL, mode, subject]);

  const startStream = useCallback(async () => {
    console.log("Starting camera stream with camera index:", cameraIndex);
    setIsStreamLoading(true);
    setUseSnapshotFallback(false);
    mjpegRetryCountRef.current = 0;

    // First, stop any existing recognition
    try {
      await axios.post(`${BACKEND_URL}/stop`);
    } catch (err) {
      console.log("No existing recognition to stop");
    }

    // Wait a moment for camera to be released
    await new Promise(resolve => setTimeout(resolve, 500));

    // Set the camera index on the backend before starting
    try {
      const selectRes = await axios.post(`${BACKEND_URL}/camera/select`, {
        camera_index: cameraIndex
      });
      console.log("Camera selected:", selectRes.data);
    } catch (err) {
      console.warn("Could not set camera index on backend:", err);
    }

    // Wait a moment after selecting camera
    await new Promise(resolve => setTimeout(resolve, 300));

    const startPollingForFrame = () => {
      let waited = 0;
      const baseInterval = 200;
      if (cameraStatusIntervalRef.current) clearInterval(cameraStatusIntervalRef.current);
      
      cameraStatusIntervalRef.current = setInterval(async () => {
        try {
          const res = await axios.get(`${BACKEND_URL}/camera_status`);
          if (res.data && res.data.recognition_running === false) {
            console.warn('Recognition stopped before first frame was produced');
            setUseSnapshotFallback(true);
            setIsStreamLoading(false);
            setStreamConnected(false);
            clearInterval(cameraStatusIntervalRef.current);
            cameraStatusIntervalRef.current = null;
            return;
          }

          if (res.data && res.data.has_frame) {
            const streamUrl = `${BACKEND_URL}/video`;
            updateImageSrc(streamUrl);
            setStreamConnected(true);
            setIsStreamLoading(false);
            clearInterval(cameraStatusIntervalRef.current);
            cameraStatusIntervalRef.current = null;
          } else {
            waited += baseInterval;
            if (waited >= streamStartTimeoutMs) {
              console.warn('Timed out waiting for camera frame - enabling snapshot fallback');
              setUseSnapshotFallback(true);
              setIsStreamLoading(false);
              clearInterval(cameraStatusIntervalRef.current);
              cameraStatusIntervalRef.current = null;
            }
          }
        } catch (err) {
          waited += baseInterval;
          if (waited >= streamStartTimeoutMs) {
            console.warn('camera_status failed repeatedly - enabling snapshot fallback', err);
            setUseSnapshotFallback(true);
            setIsStreamLoading(false);
            setStreamConnected(false);
            clearInterval(cameraStatusIntervalRef.current);
            cameraStatusIntervalRef.current = null;
          }
        }
      }, baseInterval);
    };

    const tryStart = async () => {
      try {
        const startRes = await axios.post(`${BACKEND_URL}/start`);
        if (startRes?.data?.status === 'failed') {
          throw new Error(startRes?.data?.error || 'Backend failed to start recognition');
        }
        setIsRecognitionActive(true);
        startPollingForFrame();
        return;
      } catch (e) {
        console.error("Failed to start recognition:", e);
        setUseSnapshotFallback(true);
        setIsStreamLoading(false);
        setStreamConnected(false);
      }
    };

    await tryStart();

    if (streamTimeoutRef.current) clearTimeout(streamTimeoutRef.current);
    streamTimeoutRef.current = setTimeout(() => {
      console.warn("Stream timeout - enabling snapshot fallback");
      setIsStreamLoading(false);
      setUseSnapshotFallback(true);
    }, streamStartTimeoutMs);
  }, [BACKEND_URL, cameraIndex, streamStartTimeoutMs]);

  const handleStartRecognition = useCallback(async () => {
    startStream();
  }, [startStream]);

  const handleStopRecognition = useCallback(async () => {
    try {
      setIsStreamLoading(true);
      await axios.post(`${BACKEND_URL}/stop`);
      setIsRecognitionActive(false);
    } catch (err) {
      console.error("Error stopping recognition:", err);
    } finally {
      setIsStreamLoading(false);
      setStreamConnected(false);
    }
  }, [BACKEND_URL]);

  const stopStream = useCallback((stopBackend = true) => {
    if (snapshotIntervalRef.current) {
      clearInterval(snapshotIntervalRef.current);
      snapshotIntervalRef.current = null;
    }
    if (cameraStatusIntervalRef.current) {
      clearInterval(cameraStatusIntervalRef.current);
      cameraStatusIntervalRef.current = null;
    }

    updateImageSrc(null);

    if (stopBackend) {
      axios.post(`${BACKEND_URL}/stop`).catch(err => {
        console.error("Error stopping backend:", err);
      });
    }
  }, [BACKEND_URL]);

  const handleImageLoad = useCallback(() => {
    if (streamTimeoutRef.current) {
      clearTimeout(streamTimeoutRef.current);
      streamTimeoutRef.current = null;
    }
    if (cameraStatusIntervalRef.current) {
      clearInterval(cameraStatusIntervalRef.current);
      cameraStatusIntervalRef.current = null;
    }
    setIsStreamLoading(false);
    // Keep snapshot polling alive when source is blob URL.
    // Only disable fallback when the MJPEG /video stream is confirmed loaded.
    const currentSrc = imageSrcRef.current || "";
    if (currentSrc.includes('/video')) {
      setUseSnapshotFallback(false);
      mjpegRetryCountRef.current = 0;
    }
    setStreamConnected(true);
  }, []);

  useEffect(() => {
    if (streamConnected && !useSnapshotFallback && !imageSrc) {
      const streamUrl = `${BACKEND_URL}/video?t=${Date.now()}`;
      updateImageSrc(streamUrl);
    }
  }, [streamConnected, useSnapshotFallback, imageSrc, BACKEND_URL]);

  const handleImageError = useCallback(() => {
    if (streamTimeoutRef.current) {
      clearTimeout(streamTimeoutRef.current);
      streamTimeoutRef.current = null;
    }
    setIsStreamLoading(false);
    
    if (imageSrcRef.current && imageSrcRef.current.includes('/video')) {
      if (mjpegRetryCountRef.current < maxMjpegRetries) {
        mjpegRetryCountRef.current += 1;
        setIsStreamLoading(true);
        const streamUrl = `${BACKEND_URL}/video?t=${Date.now()}`;
        updateImageSrc(streamUrl);
        return;
      }
      setUseSnapshotFallback(true);
      setStreamConnected(false);
    }
  }, [BACKEND_URL, maxMjpegRetries]);

  useEffect(() => {
    let isMounted = true;

    const initializePanel = async () => {
      const modeConfigured = await configureMode();
      if (!isMounted || !modeConfigured) {
        return;
      }

      setIsStreamLoading(false);
      await fetchStatusData();
      // Reduced polling interval from 1000ms to 2000ms for better performance
      statusIntervalRef.current = setInterval(fetchStatusData, 2000);
    };

    initializePanel();

    return () => {
      isMounted = false;
      console.log("Cleaning up StatusPanel - stopping camera");
      if (snapshotIntervalRef.current) clearInterval(snapshotIntervalRef.current);
      if (statusIntervalRef.current) clearInterval(statusIntervalRef.current);
      if (streamTimeoutRef.current) clearTimeout(streamTimeoutRef.current);
      // Do not stop backend recognition on unmount/remount.
      // This avoids false camera stops during React dev remount cycles.
      stopStream(false);
    };
  }, [configureMode, fetchStatusData, stopStream]);

  useEffect(() => {
    if (useSnapshotFallback) {
      console.log("Using snapshot fallback - MJPEG stream unavailable");

      const pollSnapshot = async () => {
        try {
          const endpoint = snapshotEndpointRef.current || "/snapshot";
          const res = await fetch(`${BACKEND_URL}${endpoint}?ts=${Date.now()}`);
          // Backend returns 404 while camera/frame is still warming up.
          // Treat it as transient instead of logging hard errors repeatedly.
          if (res.status === 404) {
            snapshot404CountRef.current += 1;
            // Some backend versions expose /frame instead of /snapshot.
            if (endpoint === "/snapshot" && snapshot404CountRef.current >= 3) {
              snapshotEndpointRef.current = "/frame";
              snapshot404CountRef.current = 0;
            }
            return;
          }
          snapshot404CountRef.current = 0;
          if (!res.ok) throw new Error(`Snapshot failed (${res.status})`);
          const blob = await res.blob();
          const url = URL.createObjectURL(blob);

          if (imageSrcRef.current && imageSrcRef.current.startsWith('blob:')) {
            URL.revokeObjectURL(imageSrcRef.current);
          }

          updateImageSrc(url);
          setStreamConnected(true);
          setIsStreamLoading(false);
        } catch (e) {
          console.error("Snapshot polling error:", e);
        }
      };

      pollSnapshot();
      snapshotIntervalRef.current = setInterval(pollSnapshot, 500);

      return () => {
        if (snapshotIntervalRef.current) {
          clearInterval(snapshotIntervalRef.current);
          snapshotIntervalRef.current = null;
        }
      };
    }
  }, [useSnapshotFallback, BACKEND_URL]);

  const isProcessingRecognition = recentlyRecognized?.status === "processing";
  
  // For class mode: success means present or late
  const isClassAttendanceSuccess = recentlyRecognized?.status === "present" || recentlyRecognized?.status === "late";
  
  // For events mode: success means attendance was recorded (present/late)
  const isEventAttendanceSuccess = recentlyRecognized?.status === "present" || recentlyRecognized?.status === "late";
  
  // Combined success check - works for both modes
  const isAttendanceSuccess = mode === 'class' ? isClassAttendanceSuccess : isEventAttendanceSuccess;

  return (
    <main className="status-panel">
      <section className="video-section">
        <h2 className="section-title">Live Camera</h2>
        <aside className={`video-container ${streamConnected ? 'active' : ''}`}>
          <span className={`status-badge connection ${streamConnected ? 'connected' : 'disconnected'}`}>
            {streamConnected ? "Connected" : "Disconnected"}
          </span>

          <span className={`status-badge recognition ${isRecognitionActive || status.recognition_running ? 'active' : 'inactive'}`}>
            {isRecognitionActive || status.recognition_running ? "Recognition Active" : "Recognition Inactive"}
          </span>

          {isStreamLoading && (
            <aside className="loading-indicator" role="status" aria-live="polite">
              <div className="spinner"></div>
              <p>Loading camera feed...</p>
            </aside>
          )}

          {imageSrc ? (
            <img
              key={imageSrc}
              src={imageSrc}
              alt="Live Camera Feed"
              className={`video-feed ${isStreamLoading ? 'loading' : ''}`}
              onLoad={handleImageLoad}
              onError={handleImageError}
            />
          ) : (
            <figure className="video-placeholder">
              <figcaption>{isStreamLoading ? 'Preparing camera...' : 'No camera feed available'}</figcaption>
            </figure>
          )}

          {!streamConnected && !isStreamLoading && (
            <aside className="connection-error" role="alert">
              <h3>Camera Disconnected</h3>
              <p>{status.recognition_running ? "Waiting for camera frame..." : "Start recognition to view live feed"}</p>
            </aside>
          )}
        </aside>
      </section>

      <section className="status-section">
        <header className="panel-header">
          <h1>Face Attendance System</h1>
          <div className="mode-info">
            {mode === 'class' && subject ? (
              <p>
                <strong>Mode:</strong> Class - <strong>Subject:</strong> {subject.name}
                {subject?.monitoring?.enabled ? " - Monitoring Mode" : ""}
              </p>
            ) : mode === 'events' ? (
              <p>
                <strong>Mode:</strong> Events
                {subject?.name ? " - " : ""}
                {subject?.name ? <><strong>Event:</strong> {subject.name}</> : null}
              </p>
            ) : mode === 'hallway' ? (
              <p><strong>Mode:</strong> Hallway Mode</p>
            ) : (
              <p><strong>Mode:</strong> Default</p>
            )}
          </div>
        </header>

        {mode === 'hallway' && (
          <section className="card attendance-mode-card">
            <div className="attendance-mode-indicator">
              <span className={`mode-badge ${attendanceMode === 'IN' ? 'mode-in' : 'mode-out'}`}>
                Current Mode: {attendanceMode}
              </span>
              <button
                type="button"
                className={`mode-switch-btn ${attendanceMode === 'IN' ? 'switch-to-out' : 'switch-to-in'}`}
                onClick={switchAttendanceMode}
                disabled={isSwitchingMode}
              >
                {isSwitchingMode ? 'Switching...' : attendanceMode === 'IN' ? 'Switch to OUT Mode' : 'Switch to IN Mode'}
              </button>
            </div>
          </section>
        )}

        {message && (
          <section className={`card message-alert ${messageType}`}>
            <p>{message}</p>
          </section>
        )}

        <section className="card system-status" aria-labelledby="system-status-heading">
          <h2 id="system-status-heading">System Status</h2>

          <div className="status-grid">
            <div className={`status-item ${status.recognition_running ? 'active' : 'inactive'}`}>
              <span className="label">Recognition</span>
              <span className="value">
                {status.recognition_running ? "Active" : "Inactive"}
              </span>
            </div>
            <div className={`status-item ${status.camera_active ? 'active' : 'error'}`}>
              <span className="label">Camera</span>
              <span className="value">
                {status.camera_active ? "Connected" : "Disconnected"}
              </span>
            </div>
            <div className={`status-item ${status.status === 'running' ? 'active' : status.status === 'error' || status.status === 'stopped' ? 'error' : 'inactive'}`}>
              <span className="label">System</span>
              <span className="value">{status.status || "Unknown"}</span>
            </div>
          </div>

          <footer className="control-footer">
            {isRecognitionActive || status.recognition_running ? (
              <button
                onClick={handleStopRecognition}
                className="control-btn stop"
              >
                Stop Recognition
              </button>
            ) : (
              <button
                onClick={handleStartRecognition}
                disabled={isStreamLoading}
                className="control-btn start"
              >
                {isStreamLoading ? 'Starting...' : 'Start Recognition'}
              </button>
            )}
          </footer>
        </section>

        {recentlyRecognized && (
          <section className="card student-details" aria-labelledby="student-details-heading">
            <h2 id="student-details-heading">Student Recognized</h2>
            <div className="student-info">
              <div className="student-main">
                {/* Show attendance status GIF for both Class and Events modes when not processing */}
                {!isProcessingRecognition && (
                  <div className="attendance-status-gif">
                    {isAttendanceSuccess ? (
                      <img 
                        src="/check 2.gif" 
                        alt="Attendance Verified" 
                        className="status-gif success-gif"
                      />
                    ) : (
                      <img 
                        src="/cross x.gif" 
                        alt="Attendance Failed" 
                        className="status-gif error-gif"
                      />
                    )}
                  </div>
                )}
                <h3 className="student-name">{recentlyRecognized.name}</h3>
                <p className="student-id">ID: {recentlyRecognized.student_id}</p>
                
                {/* Show status badge for both Class and Events modes when not processing */}
                {!isProcessingRecognition && (
                  <span className={`attendance-status-badge ${isAttendanceSuccess ? 'success' : 'failed'}`}>
                    {mode === 'class' ? (
                      // For class mode, show Present/Late/Absent based on status
                      recentlyRecognized.status === 'present' || recentlyRecognized.status === 'late' || recentlyRecognized.status === 'PRESENT' || recentlyRecognized.status === 'LATE'
                        ? 'Present' 
                        : recentlyRecognized.status === 'already_timed_in'
                          ? 'Already Timed In'
                          : recentlyRecognized.status === 'no_time_in'
                            ? 'No Time In'
                            : recentlyRecognized.status === 'already_timed_out'
                              ? 'Already Timed Out'
                              : 'Absent'
                    ) : (
                      // For events mode
                      isAttendanceSuccess
                        ? 'Attendance Verified' 
                        : 'Attendance Failed'
                    )}
                  </span>
                )}
              </div>
              <div className="student-meta">
                <p><strong>Course:</strong> {recentlyRecognized.course}</p>
                <p><strong>Year:</strong> {recentlyRecognized.year}</p>
                <p><strong>Time:</strong> {recentlyRecognized.time}</p>
                <p><strong>Date:</strong> {recentlyRecognized.date}</p>
                {recentlyRecognized.attendance_mode && (
                  <p><strong>Attendance Mode:</strong> {recentlyRecognized.attendance_mode}</p>
                )}
                {recentlyRecognized.attendance_type && (
                  <p><strong>Type:</strong> {recentlyRecognized.attendance_type}</p>
                )}
                
                {/* Show message for both modes - with specific payment/receipt messages for events */}
                {recentlyRecognized.message && (
                  <p className="attendance-message">
                    <strong>Message:</strong> {
                      mode === 'events' ? (
                        // Events mode: show specific messages based on receipt status
                        recentlyRecognized.message.includes("No verified receipt") || recognizedPaymentStatus?.receipt_status === "not_found"
                          ? "Receipt not found - Please submit receipt"
                          : recognizedPaymentStatus?.receipt_status === "pending"
                            ? "Receipt pending verification"
                            : recognizedPaymentStatus?.receipt_status === "rejected"
                              ? "Receipt rejected - Please resubmit"
                              : recentlyRecognized.message
                      ) : (
                        // Class mode: show the message directly
                        recentlyRecognized.message
                      )
                    }
                  </p>
                )}
              </div>
            </div>
          </section>
        )}
      </section>
      

      
      {/* Full Screen Confirmation Overlay for Events Mode */}
      {showConfirmation && mode === 'events' && recentlyRecognized && !isProcessingRecognition && (
        <div className="confirmation-overlay">
          <div className="confirmation-content">
            <div className={`confirmation-icon ${isAttendanceSuccess ? 'success' : 'failed'}`}>
              {isAttendanceSuccess ? (
                <svg viewBox="0 0 100 100" className="check-mark">
                  <circle cx="50" cy="50" r="45" fill="none" stroke="currentColor" strokeWidth="3" className="circle"/>
                  <path d="M30 50 L45 65 L70 35" fill="none" stroke="currentColor" strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" className="check"/>
                </svg>
              ) : (
                <svg viewBox="0 0 100 100" className="cross-mark">
                  <circle cx="50" cy="50" r="45" fill="none" stroke="currentColor" strokeWidth="3" className="circle"/>
                  <path d="M35 35 L65 65 M65 35 L35 65" fill="none" stroke="currentColor" strokeWidth="5" strokeLinecap="round" className="cross"/>
                </svg>
              )}
            </div>
            <h2 className="confirmation-title">
              {isAttendanceSuccess
                ? 'Attendance Verified' 
                : 'Attendance Failed'}
            </h2>
            <p className="confirmation-student-name">{recentlyRecognized.name}</p>
            <p className="confirmation-student-id">ID: {recentlyRecognized.student_id}</p>
            <p className="confirmation-timestamp">{recentlyRecognized.time} • {recentlyRecognized.date}</p>
          </div>
        </div>
      )}
    </main>
  );
}

export default StatusPanel;
