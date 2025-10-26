import React, { useEffect, useState, useRef, useCallback } from "react";
import axios from "axios";
import "../styles/StatusPanel.css";

function StatusPanel({ mode, subject }) {
  const [status, setStatus] = useState({});
  const [attendance, setAttendance] = useState([]);
  const [streamConnected, setStreamConnected] = useState(false);
  const [isStreamLoading, setIsStreamLoading] = useState(true);
  const [isRecognitionActive, setIsRecognitionActive] = useState(false);
  const [imageSrc, setImageSrc] = useState(null);
  const [useSnapshotFallback, setUseSnapshotFallback] = useState(false);

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:8000";
  
  const snapshotIntervalRef = useRef(null);
  const statusIntervalRef = useRef(null);
  const streamTimeoutRef = useRef(null);
  const cameraStatusIntervalRef = useRef(null);
  const imageSrcRef = useRef("");

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

  const fetchStatusData = useCallback(async () => {
    try {
      const statusRes = await axios.get(`${BACKEND_URL}/status`);
      setStatus(statusRes.data);
      setIsRecognitionActive(Boolean(statusRes.data.recognition_running));

      const attendanceRes = await axios.get(`${BACKEND_URL}/attendance`);
      setAttendance(attendanceRes.data.attendance || []);
    } catch (err) {
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
              console.warn('Timed out waiting for camera frame – enabling snapshot fallback');
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
        // ✅ use correct backend route
        await axios.post(`${BACKEND_URL}/start`);
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

    tryStart();

    if (streamTimeoutRef.current) clearTimeout(streamTimeoutRef.current);
    streamTimeoutRef.current = setTimeout(() => {
      console.warn("Stream timeout – enabling snapshot fallback");
      setIsStreamLoading(false);
      setUseSnapshotFallback(true);
    }, 12000);
  }, [BACKEND_URL]);

  const handleStartRecognition = useCallback(async () => {
    startStream();
  }, [startStream]);

  const handleStopRecognition = useCallback(async () => {
    try {
      setIsStreamLoading(true);
      // ✅ use correct backend route
      await axios.post(`${BACKEND_URL}/stop`);
      setIsRecognitionActive(false);
    } catch (err) {
      console.error("Error stopping recognition:", err);
    } finally {
      setIsStreamLoading(false);
      setStreamConnected(false);
    }
  }, [BACKEND_URL]);

  const stopStream = useCallback(() => {
    if (snapshotIntervalRef.current) {
      clearInterval(snapshotIntervalRef.current);
      snapshotIntervalRef.current = null;
    }
    if (cameraStatusIntervalRef.current) {
      clearInterval(cameraStatusIntervalRef.current);
      cameraStatusIntervalRef.current = null;
    }
    
    updateImageSrc(null);
    
    axios.post(`${BACKEND_URL}/stop`).catch(err => {
      console.error("Error stopping backend:", err);
    });
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
    setUseSnapshotFallback(false);
    setStreamConnected(true);
  }, []);

  const handleImageError = useCallback(() => {
    if (streamTimeoutRef.current) {
      clearTimeout(streamTimeoutRef.current);
      streamTimeoutRef.current = null;
    }
    setIsStreamLoading(false);
    
    if (imageSrcRef.current && imageSrcRef.current.includes('/video')) {
      setUseSnapshotFallback(true);
      setStreamConnected(false);
    }
  }, []);

  useEffect(() => {
    startStream();
    statusIntervalRef.current = setInterval(fetchStatusData, 1000);
    
    return () => {
      console.log("Cleaning up StatusPanel - stopping camera");
      if (snapshotIntervalRef.current) clearInterval(snapshotIntervalRef.current);
      if (statusIntervalRef.current) clearInterval(statusIntervalRef.current);
      if (streamTimeoutRef.current) clearTimeout(streamTimeoutRef.current);
      stopStream();
    };
  }, [startStream, fetchStatusData, stopStream]);

  useEffect(() => {
    if (useSnapshotFallback) {
      console.log("Using snapshot fallback - MJPEG stream unavailable");
      
      const pollSnapshot = async () => {
        try {
          const res = await fetch(`${BACKEND_URL}/snapshot?ts=${Date.now()}`);
          if (!res.ok) throw new Error("Snapshot failed");
          const blob = await res.blob();
          const url = URL.createObjectURL(blob);
          
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
    <main className="status-panel">
      <aside className="video-container">
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
            src={imageSrc}
            alt="Live Camera"
            className={`video-feed ${isStreamLoading ? 'loading' : ''}`}
            onLoad={handleImageLoad}
            onError={handleImageError}
          />
        ) : (
          <figure className="video-placeholder">
            <figcaption>{isStreamLoading ? 'Preparing camera...' : 'No camera feed'}</figcaption>
          </figure>
        )}

        {!streamConnected && !isStreamLoading && (
          <aside className="connection-error" role="alert">
            <h3>Connection Lost</h3>
            <p>Attempting to reconnect...</p>
            <p className="error-hint">Make sure the backend server is running on port 8000</p>
          </aside>
        )}
      </aside>

      <section className="info-panel">
        <header className="panel-header">
          <h1>Face Attendance System</h1>
          <div className="mode-info">
            {mode === 'class' && subject ? (
              <p><strong>Mode:</strong> Class • <strong>Subject:</strong> {subject.name}</p>
            ) : mode === 'events' ? (
              <p><strong>Mode:</strong> Events</p>
            ) : (
              <p><strong>Mode:</strong> Default</p>
            )}
          </div>
        </header>

        <section className="card system-status" aria-labelledby="system-status-heading">
          <h2 id="system-status-heading">System Status</h2>
          <div className="status-grid">
            <div className="status-item">
              <span className="label">Recognition</span>
              <span className={`value ${status.recognition_running ? 'active' : 'inactive'}`}>
                {status.recognition_running ? "Active" : "Inactive"}
              </span>
            </div>
            <div className="status-item">
              <span className="label">Camera</span>
              <span className={`value ${status.camera_active ? 'active' : 'inactive'}`}>
                {status.camera_active ? "Connected" : "Disconnected"}
              </span>
            </div>
            <div className="status-item">
              <span className="label">Status</span>
              <span className="value">{status.status || "Unknown"}</span>
            </div>
          </div>
          
          <div className="detection-results">
            <p>Face detection results will appear here when students are recognized.</p>
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
                {isStreamLoading ? 'Starting…' : 'Start Recognition'}
              </button>
            )}
          </footer>
        </section>

        <section className="card attendance-list" aria-labelledby="attendance-heading">
          <h2 id="attendance-heading">Attendance</h2>
          <p className="attendance-count">
            <strong>Total Present:</strong> {attendance.length || 0}
          </p>
          <div className="attendance-scroll">
            {attendance.length > 0 ? (
              <ul>
                {attendance.map((record, index) => (
                  <li key={index} className="attendance-item">
                    <div className="attendance-main">
                      <strong className="student-name">{record.name}</strong>
                      <span className="timestamp">{record.timestamp}</span>
                    </div>
                    <div className="attendance-meta">
                      {record.course} • {record.year}
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="empty-state">No attendance records yet.</p>
            )}
          </div>
        </section>
      </section>
    </main>
  );
}

export default StatusPanel;
