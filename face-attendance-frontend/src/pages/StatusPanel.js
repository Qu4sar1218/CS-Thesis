// Import React hooks and axios for HTTP requests
import React, { useEffect, useState, useRef, useCallback } from "react";
import axios from "axios";

// StatusPanel component displays live video, detected student info, and attendance
function StatusPanel({ mode, subject }) {
  // State to hold status info from backend
  const [status, setStatus] = useState({});
  // State to hold attendance info from backend
  const [attendance, setAttendance] = useState([]);
  // Add connection state
  const [streamConnected, setStreamConnected] = useState(false);
  // Add stream loading state
  const [isStreamLoading, setIsStreamLoading] = useState(true);
  // Recognition active state (backend)
  const [isRecognitionActive, setIsRecognitionActive] = useState(false);

  // Backend base URL
  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:8000";
  // imageSrc: initially null (do not render <img> with empty src to avoid React warning)
  const [imageSrc, setImageSrc] = useState(null);
  const [useSnapshotFallback, setUseSnapshotFallback] = useState(false);
  const snapshotIntervalRef = useRef(null);
  const statusIntervalRef = useRef(null);
  const streamTimeoutRef = useRef(null);
  const cameraStatusIntervalRef = useRef(null);
  // ref to track current imageSrc for cleanup without causing effect deps
  const imageSrcRef = useRef("");
  const updateImageSrc = (url) => {
    // Clean up previous blob URL when replacing it
    try {
      if (imageSrcRef.current && imageSrcRef.current.startsWith && imageSrcRef.current.startsWith('blob:') && imageSrcRef.current !== url) {
        URL.revokeObjectURL(imageSrcRef.current);
      }
    } catch (e) {
      // ignore
    }
    imageSrcRef.current = url;
    // Use null for no src to avoid React warning about empty string
    setImageSrc(url || null);
  };

  // Fetch status and attendance data
  const fetchStatusData = useCallback(async () => {
    try {
      // Fetch current status (detected student, liveness, etc.)
      const statusRes = await axios.get(`${BACKEND_URL}/status`);
      setStatus(statusRes.data);
      // Keep recognition state in sync
      setIsRecognitionActive(Boolean(statusRes.data.recognition_running));

      // Fetch attendance list - FIXED: Use the correct endpoint structure
      const attendanceRes = await axios.get(`${BACKEND_URL}/attendance`);
      setAttendance(attendanceRes.data.attendance || []);

          } catch (err) {
      // Mark as disconnected on error
            console.error("Error fetching status:", err);
    }
  }, [BACKEND_URL]);

  const startStream = useCallback(() => {
    console.log("Starting camera stream");
    setIsStreamLoading(true);
    setUseSnapshotFallback(false);

    const startPollingForFrame = () => {
      let waited = 0;
      const baseInterval = 200;
      if (cameraStatusIntervalRef.current) clearInterval(cameraStatusIntervalRef.current);
      cameraStatusIntervalRef.current = setInterval(async () => {
        try {
          const res = await axios.get(`${BACKEND_URL}/camera_status`);
          if (res.data && res.data.has_frame) {
            const streamUrl = `${BACKEND_URL}/video?t=${Date.now()}`;
            updateImageSrc(streamUrl);
            setStreamConnected(true);
            setIsStreamLoading(false);
            clearInterval(cameraStatusIntervalRef.current);
            cameraStatusIntervalRef.current = null;
          } else {
            waited += baseInterval;
            if (waited >= 12000) {
              console.warn('Timed out waiting for camera frame — enabling snapshot fallback');
              setUseSnapshotFallback(true);
              setIsStreamLoading(false);
              clearInterval(cameraStatusIntervalRef.current);
              cameraStatusIntervalRef.current = null;
            }
          }
        } catch (err) {
          console.warn('camera_status error; trying direct /video probe', err);
          const streamUrl = `${BACKEND_URL}/video?t=${Date.now()}`;
          updateImageSrc(streamUrl);
        }
      }, baseInterval);
    };

    const tryStart = async () => {
      try {
        await axios.post(`${BACKEND_URL}/start_recognition`);
        setIsRecognitionActive(true);
        startPollingForFrame();
        return;
      } catch (e1) {
        console.warn('start_recognition failed, trying /start', e1);
      }
      try {
        await axios.post(`${BACKEND_URL}/start`);
        setIsRecognitionActive(true);
        startPollingForFrame();
        return;
      } catch (e2) {
        console.warn('start failed, trying stream-only', e2);
      }
      try {
        await axios.post(`${BACKEND_URL}/start_stream_only`);
        setIsRecognitionActive(false);
        startPollingForFrame();
        return;
      } catch (e3) {
        console.error('All start attempts failed', e3);
        setUseSnapshotFallback(true);
        setIsStreamLoading(false);
        setStreamConnected(false);
      }
    };

    tryStart();

    if (streamTimeoutRef.current) clearTimeout(streamTimeoutRef.current);
    streamTimeoutRef.current = setTimeout(() => {
      console.warn("Stream timeout — enabling snapshot fallback");
      setIsStreamLoading(false);
      setUseSnapshotFallback(true);
    }, 12000);
  }, [BACKEND_URL]);

  // Start recognition (explicit control) - called by UI button
  const handleStartRecognition = useCallback(async () => {
    // Re-run full start flow to open camera
    startStream();
  }, [startStream]);

  // Stop recognition (explicit control) - called by UI button
  const handleStopRecognition = useCallback(async () => {
    try {
      setIsStreamLoading(true);
      await axios.post(`${BACKEND_URL}/stop_recognition`);
      setIsRecognitionActive(false);
    } catch (err) {
      console.error('Error stopping recognition:', err);
      // Also try generic stop
      try { await axios.post(`${BACKEND_URL}/stop`); } catch(e) {}
    } finally {
      setIsStreamLoading(false);
      // keep connection indicator as appropriate
      setStreamConnected(false);
    }
  }, [BACKEND_URL]);

  // Stop the video stream
  const stopStream = useCallback(() => {
    
    // Clear any snapshot polling
    if (snapshotIntervalRef.current) {
      clearInterval(snapshotIntervalRef.current);
      snapshotIntervalRef.current = null;
    }
    if (cameraStatusIntervalRef.current) {
      clearInterval(cameraStatusIntervalRef.current);
      cameraStatusIntervalRef.current = null;
    }
    
    // Clear the image source (use null so <img> is not rendered)
    updateImageSrc(null);
    
    // Stop the backend recognition
    axios.post(`${BACKEND_URL}/stop`).catch(err => {
      console.error("Error stopping backend:", err);
    });
  }, [BACKEND_URL]);

  // Handle image load success
  const handleImageLoad = useCallback(() => {
    // Clear the connection timeout
    if (streamTimeoutRef.current) {
      clearTimeout(streamTimeoutRef.current);
      streamTimeoutRef.current = null;
    }
      if (cameraStatusIntervalRef.current) {
        clearInterval(cameraStatusIntervalRef.current);
        cameraStatusIntervalRef.current = null;
      }
    setIsStreamLoading(false);
    // If we successfully loaded the MJPEG stream, disable snapshot fallback and mark connected
    setUseSnapshotFallback(false);
    setStreamConnected(true);
  }, []);

  // Handle image load error
  const handleImageError = useCallback((e) => {
    
    // Clear the connection timeout
    if (streamTimeoutRef.current) {
      clearTimeout(streamTimeoutRef.current);
      streamTimeoutRef.current = null;
    }
    setIsStreamLoading(false);
    
    // Only set fallback if we were trying to load the video stream
    if (imageSrcRef.current && imageSrcRef.current.includes('/video')) {
      setUseSnapshotFallback(true);
      setStreamConnected(false);
    }
  }, []);

  // Initialize stream when component mounts
  useEffect(() => {
    startStream();
    
    // Set up status polling
    statusIntervalRef.current = setInterval(fetchStatusData, 1000);
    
    // Cleanup on unmount - this will be called when Close Panel is clicked
    return () => {
      console.log("Cleaning up StatusPanel - stopping camera");
      if (snapshotIntervalRef.current) {
        clearInterval(snapshotIntervalRef.current);
      }
      if (statusIntervalRef.current) {
        clearInterval(statusIntervalRef.current);
      }
      if (streamTimeoutRef.current) {
        clearTimeout(streamTimeoutRef.current);
      }
      
      // Stop the stream last
      stopStream();
    };
    }, [startStream, fetchStatusData, stopStream]);

  // Start/stop snapshot polling when fallback is enabled
  useEffect(() => {
    if (useSnapshotFallback) {
      console.log("Using snapshot fallback - MJPEG stream unavailable");
      
      // Poll snapshot every 800ms to avoid camera contention
      const pollSnapshot = async () => {
        try {
          const res = await fetch(`${BACKEND_URL}/snapshot?ts=${Date.now()}`);
          if (!res.ok) throw new Error("Snapshot failed");
          const blob = await res.blob();
          const url = URL.createObjectURL(blob);
          
          // Clean up previous URL
          if (imageSrcRef.current && imageSrcRef.current.startsWith('blob:')) {
            URL.revokeObjectURL(imageSrcRef.current);
          }
          
          updateImageSrc(url);
        } catch (e) {
          console.error("Snapshot polling error:", e);
        }
      };

      pollSnapshot();
      snapshotIntervalRef.current = setInterval(pollSnapshot, 800);
      
      return () => {
        if (snapshotIntervalRef.current) {
          clearInterval(snapshotIntervalRef.current);
          snapshotIntervalRef.current = null;
        }
      };
    }
  }, [useSnapshotFallback, BACKEND_URL]);

  return (
    <div style={{ display: "flex", height: "100vh", fontFamily: "Inter, sans-serif" }}>

      {/* Left Panel: Live Video Feed from backend camera */}
      <div style={{
        flex: 1,
        backgroundColor: "#1f2937",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "20px",
        position: "relative"
      }}>
        {/* Show connection status */}
        <div style={{
          position: "absolute",
          top: "10px",
          left: "10px",
          padding: "5px 10px",
          backgroundColor: streamConnected ? "#10b981" : "#ef4444",
          color: "white",
          borderRadius: "5px",
          fontSize: "12px",
          zIndex: 1000
        }}>
          {streamConnected ? "Connected" : "Disconnected"}
        </div>

        {/* System status */}
        <div style={{
          position: "absolute",
          top: "10px",
          right: "10px",
          padding: "5px 10px",
          backgroundColor: status.recognition_running ? "#10b981" : "#f59e0b",
          color: "white",
          borderRadius: "5px",
          fontSize: "12px",
          zIndex: 1000
        }}>
          {isRecognitionActive || status.recognition_running ? "Recognition Active" : "Recognition Inactive"}
        </div>

        {/* Loading indicator */}
        {isStreamLoading && (
          <div style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            color: "white",
            fontSize: "18px",
            zIndex: 100
          }}>
            Loading camera feed...
          </div>
        )}

        {/* Video stream from backend */}
        {imageSrc ? (
          <img
            src={imageSrc}
          alt="Live Camera"
          style={{
            width: "100%",
            height: "100%",
            objectFit: "contain",
            borderRadius: "20px",
            boxShadow: "0 8px 24px rgba(0,0,0,0.3)",
            border: "3px solid #4f46e5",
            // always render the image element — use opacity while loading so browser starts fetching stream
            opacity: isStreamLoading ? 0.02 : 1,
            transition: "opacity 300ms ease-in-out",
            transform: "scaleX(1)"  // Normal orientation (not mirrored)
          }}
          onLoad={handleImageLoad}
          onError={handleImageError}
          />
        ) : (
          // Render a placeholder when no image source is available
          <div style={{
            width: "100%",
            height: "100%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "white",
            fontSize: "18px",
            pointerEvents: "none",
          }}> 
            {isStreamLoading ? 'Preparing camera...' : 'No camera feed'}
          </div>
        )}

        {/* Error overlay for disconnected state */}
        {!streamConnected && (
          <div style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            backgroundColor: "rgba(239, 68, 68, 0.9)",
            color: "white",
            padding: "20px",
            borderRadius: "10px",
            textAlign: "center",
            maxWidth: "80%",
            zIndex: 100
          }}>
            <h3>Connection Lost</h3>
            <p>Attempting to reconnect...</p>
            <p style={{ fontSize: "12px", marginTop: "10px" }}>
              Make sure the backend server is running on port 8000
            </p>
          </div>
        )}
      </div>

      {/* Right Panel: Info and Attendance */}
      <div style={{
        flex: 1,
        padding: "40px",
        backgroundColor: "#f3f4f6",
        overflowY: "auto"
      }}>
        {/* Title of the system */}
        <h1 style={{ color: "#111827", marginBottom: "8px" }}>Face Attendance System</h1>
        <div style={{ marginBottom: "20px", color: "#374151" }}>
          {mode === 'class' && subject ? (
            <div><strong>Mode:</strong> Class • <strong>Subject:</strong> {subject.name}</div>
          ) : mode === 'events' ? (
            <div><strong>Mode:</strong> Events</div>
          ) : (
            <div><strong>Mode:</strong> Default</div>
          )}
        </div>

        {/* Current Student Info Panel */}
        <div style={{
          marginBottom: "30px",
          padding: "25px",
          borderRadius: "20px",
          background: "linear-gradient(135deg, #e0f2fe, #bae6fd)",
          boxShadow: "0 6px 20px rgba(0,0,0,0.1)"
        }}>
          <h2 style={{ marginBottom: "15px", color: "#1e3a8a" }}>System Status</h2>
          <p><strong>Recognition:</strong> {status.recognition_running ? "Active" : "Inactive"}</p>
          <p><strong>Camera:</strong> {status.camera_active ? "Connected" : "Disconnected"}</p>
          <p><strong>Status:</strong> {status.status || "Unknown"}</p>
          
          {/* You can add face detection results here when available */}
          <div style={{ marginTop: "15px", padding: "10px", backgroundColor: "rgba(255,255,255,0.5)", borderRadius: "10px" }}>
            <p style={{ fontStyle: "italic", color: "#666" }}>
              Face detection results will appear here when students are recognized.
            </p>
          </div>
          {/* Recognition control button */}
          <div style={{ marginTop: 12 }}>
            {isRecognitionActive || status.recognition_running ? (
              <button onClick={handleStopRecognition} style={{ padding: '8px 14px', background: '#ef4444', color: 'white', border: 'none', borderRadius: 8, cursor: 'pointer' }}>
                Stop Recognition
              </button>
            ) : (
              <button onClick={handleStartRecognition} disabled={isStreamLoading} style={{ padding: '8px 14px', background: '#10b981', color: 'white', border: 'none', borderRadius: 8, cursor: isStreamLoading ? 'not-allowed' : 'pointer', opacity: isStreamLoading ? 0.7 : 1 }}>
                {isStreamLoading ? 'Starting…' : 'Start Recognition'}
              </button>
            )}
          </div>
        </div>

        {/* Attendance Panel - FIXED: Use correct attendance data structure */}
        <div style={{
          padding: "25px",
          borderRadius: "20px",
          background: "linear-gradient(135deg, #fef3c7, #fde68a)",
          boxShadow: "0 6px 20px rgba(0,0,0,0.1)"
        }}>
          <h2 style={{ marginBottom: "15px", color: "#78350f" }}>Attendance</h2>
          <p><strong>Total Present:</strong> {attendance.length || 0}</p>
          <div style={{ maxHeight: "300px", overflowY: "auto", paddingLeft: "0" }}>
            {attendance.length > 0 ? (
              <ul style={{ paddingLeft: "20px" }}>
                {attendance.map((record, index) => (
                  <li key={index} style={{ marginBottom: "8px" }}>
                    <strong>{record.name}</strong> - {record.timestamp}
                    <br />
                    <small>{record.course} • {record.year}</small>
                  </li>
                ))}
              </ul>
            ) : (
              <p style={{ fontStyle: "italic", color: "#666" }}>No attendance records yet.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// Export the StatusPanel component as default
export default StatusPanel;