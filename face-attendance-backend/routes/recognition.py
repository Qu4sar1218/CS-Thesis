"""
Face recognition routes.
"""
import cv2
import threading
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Iterator
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse, Response, JSONResponse

try:
    import face_recognition
    HAVE_FACE_RECOG = True
except Exception:
    HAVE_FACE_RECOG = False

from config import settings
from database.connection import get_database
from face_recognition import load_faces_from_db, recognize_face, known_face_names
from utils import format_student_name

logger = logging.getLogger(__name__)

router = APIRouter()

# Global variables for recognition
latest_frame: Optional[bytes] = None
latest_frame_lock = threading.Lock()
latest_frame_cond = threading.Condition(latest_frame_lock)

recognition_running = False
stop_streaming = False
active_camera: Optional[cv2.VideoCapture] = None

# Attendance tracking
attendance_records: list = []
attendance_lock = threading.Lock()

# Recently recognized student tracking
recently_recognized: Optional[Dict[str, Any]] = None
recently_recognized_lock = threading.Lock()

# Mode tracking
current_mode = "class"  # Default to class mode
current_event_id: Optional[str] = None
current_class_id: Optional[str] = None


def release_camera() -> None:
    """Release the active camera."""
    global active_camera
    if active_camera:
        active_camera.release()
        active_camera = None


def open_camera() -> Optional[cv2.VideoCapture]:
    """Open camera for video capture."""
    global active_camera
    for i in range(3):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            active_camera = cap
            logger.info(f"🎥 Camera opened at index {i}")
            return cap
    logger.error("❌ No camera detected")
    return None


def recognition_loop() -> None:
    """Main recognition loop running in a separate thread."""
    global latest_frame, recognition_running, stop_streaming

    logger.info("🎬 Starting recognition loop...")
    cap = open_camera()
    if not cap:
        logger.error("❌ Failed to open camera, stopping recognition")
        recognition_running = False
        return

    logger.info("✅ Camera opened successfully, starting recognition")

    frame_count = 0

    while recognition_running and not stop_streaming:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=settings.frame_scale, fy=settings.frame_scale)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces every frame for bounding boxes
        if HAVE_FACE_RECOG:
            faces_small = face_recognition.face_locations(rgb_small, model="hog")

            # Scale back to original size
            faces = [(int(top / settings.frame_scale), int(right / settings.frame_scale),
                     int(bottom / settings.frame_scale), int(left / settings.frame_scale))
                    for (top, right, bottom, left) in faces_small]

            # Draw bounding boxes for all detected faces
            for (top, right, bottom, left) in faces:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Do recognition every N frames
            if frame_count % settings.process_every_n_frames == 0:
                encs = face_recognition.face_encodings(rgb_small, faces_small)

                for (top, right, bottom, left), enc in zip(faces, encs):
                    name = "Unknown"  # Initialize name
                    recognition_result = recognize_face(enc)

                    if recognition_result:
                        name = recognition_result['name']
                        student_id = recognition_result['student_id']
                        course = recognition_result['course']
                        year = recognition_result['year']

                        # Record attendance in memory and database
                        current_time = datetime.now().strftime('%H:%M:%S')

                        # Check payment status for events mode
                        payment_verified = False
                        if current_mode == "events" and current_event_id:
                            try:
                                # Check payment status via API call
                                import aiohttp

                                async def check_payment() -> bool:
                                    async with aiohttp.ClientSession() as session:
                                        url = f"http://127.0.0.1:{settings.port}/students/{student_id}/payment-status/{current_event_id}"
                                        async with session.get(url) as response:
                                            if response.status == 200:
                                                data = await response.json()
                                                return data.get("paid", False)
                                            return False

                                payment_verified = asyncio.run(check_payment())
                            except Exception as e:
                                logger.error(f"❌ Error checking payment status for {student_id}: {e}")
                                payment_verified = False

                        with attendance_lock:
                            # Check if student already recorded today
                            today_records = [r for r in attendance_records
                                           if r['student_id'] == student_id and r['date'] == datetime.now().strftime('%Y-%m-%d')]
                            if not today_records:
                                record = {
                                    'name': format_student_name(
                                        "", "", name  # Simplified - would need full student data
                                    ),
                                    'student_id': student_id,
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'date': datetime.now().strftime('%Y-%m-%d'),
                                    'time': current_time,
                                    'course': course,
                                    'year': year,
                                    'status': 'present',
                                    'payment_verified': payment_verified if current_mode == "events" else None
                                }
                                attendance_records.append(record)

                                # Save to database asynchronously
                                from attendance import save_attendance_to_db, update_attendance_status

                                asyncio.run_coroutine_threadsafe(
                                    save_attendance_to_db(record),
                                    asyncio.get_event_loop()
                                )

                                # Also update database record if it exists as absent
                                if current_mode == "class" and current_class_id:
                                    asyncio.run_coroutine_threadsafe(
                                        update_attendance_status(student_id, current_class_id, 'present'),
                                        asyncio.get_event_loop()
                                    )

                        # Update recently recognized student
                        with recently_recognized_lock:
                            global recently_recognized
                            recently_recognized = {
                                'name': name,
                                'student_id': student_id,
                                'time': current_time,
                                'date': datetime.now().strftime('%Y-%m-%d'),
                                'course': course,
                                'year': year
                            }

                    # Draw name immediately
                    cv2.putText(frame, name,
                              (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        frame_count += 1

        # Always update the frame for smooth streaming
        _, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), settings.jpeg_quality])
        with latest_frame_lock:
            latest_frame = jpeg.tobytes()
            latest_frame_cond.notify_all()

    release_camera()
    logger.info("🛑 Recognition stopped")


def frame_stream() -> Iterator[bytes]:
    """Generator for video frame streaming."""
    last_frame = None
    while recognition_running:
        with latest_frame_cond:
            latest_frame_cond.wait(timeout=0.1)  # Shorter timeout for more responsive streaming
            if latest_frame:
                last_frame = latest_frame
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + latest_frame + b"\r\n"
            elif last_frame:
                # Send the last known frame to keep the stream alive
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + last_frame + b"\r\n"


@router.get("/")
def root() -> Dict[str, Any]:
    """Root endpoint."""
    from face_recognition import known_face_names
    return {"status": "ok", "faces_loaded": len(known_face_names)}


@router.get("/health")
def health() -> Dict[str, Any]:
    """Health check endpoint."""
    from face_recognition import known_face_names
    return {"running": recognition_running, "faces": len(known_face_names)}


@router.get("/video")
def video() -> StreamingResponse:
    """Video streaming endpoint."""
    return StreamingResponse(frame_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/frame")
def get_frame() -> Response:
    """Get the latest frame as a JPEG response."""
    if latest_frame is None:
        return JSONResponse({"error": "No frame available"}, status_code=404)

    return Response(content=latest_frame, media_type="image/jpeg")


@router.post("/start")
def start() -> Dict[str, str]:
    """Start face recognition."""
    global recognition_running, stop_streaming
    if recognition_running:
        return {"status": "already_running"}

    logger.info("🚀 Starting face recognition...")
    recognition_running = True
    stop_streaming = False

    try:
        thread = threading.Thread(target=recognition_loop, daemon=True)
        thread.start()
        logger.info("✅ Recognition thread started")
    except Exception as e:
        logger.error(f"❌ Failed to start recognition thread: {e}")
        recognition_running = False
        return {"status": "failed", "error": str(e)}

    return {"status": "started"}


@router.post("/stop")
def stop() -> Dict[str, str]:
    """Stop face recognition."""
    global recognition_running, stop_streaming
    recognition_running = False
    stop_streaming = True
    release_camera()
    return {"status": "stopped"}


@router.post("/reload_faces")
async def reload_faces() -> Dict[str, str]:
    """Reload faces from database."""
    await load_faces_from_db()
    return {"status": "reloaded from database"}


@router.post("/set-mode")
async def set_mode(mode_data: Dict[str, Any]) -> Dict[str, Any]:
    """Set the current recognition mode."""
    global current_mode, current_event_id, current_class_id
    mode = mode_data.get("mode")
    event_id = mode_data.get("event_id")

    if mode not in ["class", "events"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid mode")

    if mode == "class":
        # For class mode, event_id is actually class_id
        class_id = event_id
        if not class_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="class_id is required for class mode")

        # Check if class exists
        db = get_database()
        class_doc = await db.classes.find_one({"_id": class_id})
        if not class_doc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Class not found")

        # TODO: Check if class is scheduled today
        # from utils import is_class_scheduled_today
        # if not is_class_scheduled_today(class_doc.get("schedule", "")):
        #     raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This class is not scheduled for today")

        current_class_id = class_id
        current_event_id = None
    else:
        # For events mode
        current_class_id = None
        current_event_id = event_id

    current_mode = mode
    return {"message": f"Mode set to {mode}", "event_id": event_id}


@router.get("/status")
def get_status() -> Dict[str, Any]:
    """Get current system status."""
    return {
        "status": "running" if recognition_running else "stopped",
        "recognition_running": recognition_running,
        "camera_active": active_camera is not None,
        "faces_loaded": len(known_face_names),
        "current_mode": current_mode,
        "current_event_id": current_event_id
    }


@router.get("/camera_status")
def camera_status() -> Dict[str, Any]:
    """Get camera status and frame availability."""
    return {
        "camera_active": active_camera is not None,
        "has_frame": latest_frame is not None,
        "recognition_running": recognition_running
    }


@router.get("/attendance")
def get_attendance() -> Dict[str, list]:
    """Get current attendance records."""
    with attendance_lock:
        return {"attendance": attendance_records.copy()}


@router.get("/recently-recognized")
def get_recently_recognized() -> Dict[str, Any]:
    """Get the most recently recognized student."""
    with recently_recognized_lock:
        return {"recently_recognized": recently_recognized}


@router.get("/snapshot")
def snapshot() -> Response:
    """Get a snapshot from the camera."""
    if latest_frame is None:
        return JSONResponse({"error": "No frame available"}, status_code=404)

    return Response(content=latest_frame, media_type="image/jpeg")


@router.post("/clear_attendance")
def clear_attendance() -> Dict[str, str]:
    """Clear attendance records."""
    global attendance_records
    with attendance_lock:
        attendance_records.clear()
    return {"status": "cleared"}
