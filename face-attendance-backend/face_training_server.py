from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
import platform
# Face recognition import
try:
    import face_recognition
    HAVE_FACE_RECOG = True
except Exception as e:
    print(f"[WARN] face_recognition missing: {e}")
    HAVE_FACE_RECOG = False
import pickle
from datetime import datetime
from werkzeug.utils import secure_filename
import tempfile
import shutil
import time

# MongoDB imports
# Removed unused MongoClient import



app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, "StudentFaceData")
ENCODINGS_DIR = os.path.join(BACKEND_ROOT, "data", "encodings")
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Ensure directories exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(ENCODINGS_DIR, exist_ok=True)


# ============================================================================
# STORAGE FUNCTIONS - Replace these to use Firestore/Firebase/Cloud Storage
# ============================================================================

def save_image_to_storage(student_id, image_data):
    """
    Save image locally for now.

    TO MIGRATE TO FIRESTORE/FIREBASE:
    1. Replace this function to upload to Firebase Storage:
       - Use firebase_admin.storage.bucket().blob(path).upload_from_string(image_data)
    2. Return the cloud storage URL instead of local path
    3. Update train_face_model() to download images from cloud storage

    Args:
        student_id: Unique student identifier
        image_data: Raw image bytes

    Returns:
        str: Path where image was saved
    """
    student_folder = os.path.join(DATASET_DIR, str(student_id))
    os.makedirs(student_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_path = os.path.join(student_folder, f"{timestamp}.jpg")

    with open(image_path, "wb") as f:
        f.write(image_data)

    return image_path


def save_encodings_to_storage(student_id, encodings):
    """
    Save face encodings ATOMICALLY - tmp file → move.
    
    TO MIGRATE TO FIRESTORE: [unchanged comments]
    """
    encoding_path = os.path.join(ENCODINGS_DIR, f"{student_id}.pkl")
    tmp_path = encoding_path + '.tmp'
    
    # Atomic write: dump to tmp first
    with open(tmp_path, "wb") as f:
        pickle.dump(encodings, f)
    
    # Atomic move - safe even on crash
    shutil.move(tmp_path, encoding_path)
    
    return encoding_path


def load_encodings_from_storage(student_id):
    """
    Load face encodings from local storage.

    TO MIGRATE TO FIRESTORE:
    Replace with:
        doc = db.collection('face_encodings').document(student_id).get()
        return np.array(doc.to_dict()['encodings']) if doc.exists else None

    Args:
        student_id: Unique student identifier

    Returns:
        list: List of face encodings or None if not found
    """
    encoding_path = os.path.join(ENCODINGS_DIR, f"{student_id}.pkl")

    if not os.path.exists(encoding_path):
        return None

    with open(encoding_path, "rb") as f:
        return pickle.load(f)


def get_student_images(student_id):
    """
    Get all image paths for a student from local storage.

    TO MIGRATE TO FIREBASE STORAGE:
    1. List all blobs in the student's folder:
       blobs = bucket.list_blobs(prefix=f"dataset/{student_id}/")
    2. Download each blob temporarily or process directly from URLs
    3. Return list of temporary paths or cloud URLs

    Args:
        student_id: Unique student identifier

    Returns:
        list: List of image file paths
    """
    student_folder = os.path.join(DATASET_DIR, str(student_id))

    if not os.path.exists(student_folder):
        return []

    image_files = []
    for filename in os.listdir(student_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(os.path.join(student_folder, filename))

    return image_files

# ============================================================================
# END OF STORAGE FUNCTIONS
# ============================================================================


def allowed_file(filename):
    """Check if uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================================
# WEBCAM CAPTURE FUNCTIONS FOR POSITION-BASED FACE TRAINING
# ============================================================================

# Valid positions for face capture
VALID_POSITIONS = ['front', 'left', 'right', 'up', 'down']

# Position-specific error messages for guidance
POSITION_GUIDANCE = {
    'front': "Face capture failed. Please look at the CENTER and capture again.",
    'left': "Face capture failed. Please look LEFT and capture again.",
    'right': "Face capture failed. Please look RIGHT and capture again.",
    'up': "Face capture failed. Please look UP and capture again.",
    'down': "Face capture failed. Please look DOWN and capture again."
}


def get_camera_backends_for_platform():
    """Choose camera backends by platform."""
    if platform.system() == "Windows":
        backends = [cv2.CAP_MSMF]
        backends.append(cv2.CAP_ANY)
        backends.append(cv2.CAP_DSHOW)
        return backends
    return [cv2.CAP_ANY]


def capture_frame_from_webcam():
    """
    Capture a single frame from the webcam.
    
    Returns:
        tuple: (success: bool, image_data: bytes or error_message: str)
    """
    if not HAVE_FACE_RECOG:
        return False, "Face recognition library not available"
    
    # Try different backends to find a working camera
    backends = get_camera_backends_for_platform()
    camera_index = 0  # Default camera
    
    cap = None
    for backend in backends:
        try:
            cap = cv2.VideoCapture(camera_index, backend)
            if cap and cap.isOpened():
                # Apply camera settings for stability
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                # DISCARD 5 FRAMES for auto-exposure settling
                for _ in range(5):
                    ret, _ = cap.read()
                
                break
            elif cap:
                cap.release()
                cap = None
        except Exception as e:
            continue
    
    if not cap or not cap.isOpened():
        return False, "Could not access webcam. Please ensure camera is connected and not in use by another application."
    
    try:
        # Capture a single frame
        ret, frame = cap.read()
        
        if not ret or frame is None:
            return False, "Failed to capture frame from webcam"
        
        # Flip horizontally for more natural mirror-like view
        frame = cv2.flip(frame, 1)
        
        # Encode as JPEG
        _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        image_data = jpeg.tobytes()
        
        return True, image_data
        
    except Exception as e:
        return False, f"Error capturing frame: {str(e)}"
    finally:
        # Always release the camera
        if cap:
            cap.release()


def get_next_image_number(student_id, position):
    """
    Get the next available image number for a student ID and position.
    
    Args:
        student_id: Unique student identifier
        position: Position string (front, left, right, up, down)
    
    Returns:
        int: The next available image number
    """
    student_folder = os.path.join(DATASET_DIR, str(student_id))
    
    if not os.path.exists(student_folder):
        return 1
    
    # Find existing images with this position prefix
    prefix = f"{student_id}_{position}_"
    max_number = 0
    
    for filename in os.listdir(student_folder):
        if filename.startswith(prefix) and filename.endswith('.jpg'):
            # Extract number from filename format: studentID_position_number.jpg
            try:
                # Remove prefix and .jpg suffix
                num_str = filename[len(prefix):-4]
                number = int(num_str)
                if number > max_number:
                    max_number = number
            except ValueError:
                continue
    
    return max_number + 1


def detect_and_validate_face(image_data):
    """
    Detect and validate a face in image data (bytes) - IN MEMORY.
    
    Args:
        image_data: Raw image bytes
    
    Returns:
        tuple: (success: bool, result: str or None)
        - On success: result is None (validated)
        - On failure: result is error message
    """
    try:
        # Decode in memory - NO TEMP FILE
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return False, "Failed to decode image data"
        
        # Convert BGR to RGB for face_recognition
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Validate face
        success, result = detect_and_encode_face(image_rgb)
        
        return success, None if success else result
            
    except Exception as e:
        return False, f"Error processing image: {str(e)}"


# ============================================================================
# END OF WEBCAM CAPTURE FUNCTIONS
# ============================================================================


def detect_and_encode_face(image_or_path):
    """
    Detect face in image and generate encoding.

    Args:
        image_or_path: Either np.ndarray image or path to image file

    Returns:
        tuple: (success: bool, encoding: np.array or error_message: str)
    """
    try:
        # Load image if path provided, or use directly if numpy array
        if isinstance(image_or_path, str):
            image = face_recognition.load_image_file(image_or_path)
        else:
            image = image_or_path

        # Detect face locations
        face_locations = face_recognition.face_locations(image, model="hog")

        if len(face_locations) == 0:
            return False, "No face detected in the image"

        if len(face_locations) > 1:
            return False, f"Multiple faces detected ({len(face_locations)}). Please ensure only one face is visible"

        # Generate face encoding with num_jitters=2
        face_encodings = face_recognition.face_encodings(image, face_locations, num_jitters=2)

        if len(face_encodings) == 0:
            return False, "Could not generate face encoding"

        return True, face_encodings[0]

    except Exception as e:
        return False, f"Error processing image: {str(e)}"


def train_face_model(student_id):
    """
    INCREMENTAL face training: only encode NEW images.
    
    Loads existing encodings, appends only new images' encodings.
    
    Args:
        student_id: Unique student identifier

    Returns:
        tuple: (success: bool, message: str, new_encodings_count: int)
    """
    try:
        # Load EXISTING encodings if any
        existing_encodings = load_encodings_from_storage(student_id) or []
        existing_count = len(existing_encodings)
        
        # Get ALL images, but only process NEW ones
        image_paths = get_student_images(student_id)
        
        if not image_paths:
            return False, "No images found for this student", 0
        
        if existing_count >= len(image_paths):
            return True, f"Already trained {existing_count} encodings (no new images)", 0
        
        # Only process NEW images (from existing_count onwards)
        new_image_paths = image_paths[existing_count:]
        
        new_encodings = []
        failed_images = []
        
        print(f"🔄 Incremental training: {len(new_image_paths)} new images (total images: {len(image_paths)})")
        
        # Process only NEW images
        for image_path in new_image_paths:
            success, result = detect_and_encode_face(image_path)
            
            if success:
                new_encodings.append(result)
            else:
                failed_images.append({
                    "path": os.path.basename(image_path),
                    "error": result
                })
        
        if new_encodings:
            # APPEND to existing
            all_encodings = existing_encodings + new_encodings
            save_encodings_to_storage(student_id, all_encodings)
            
            success_msg = f"Added {len(new_encodings)} new encodings (total: {len(all_encodings)})"
            if failed_images:
                success_msg += f" ({len(failed_images)} failed)"
            
            return True, success_msg, len(new_encodings)
        elif existing_encodings:
            return True, f"No new valid encodings (total: {len(existing_encodings)})", 0
        else:
            error_details = "; ".join([f"{f['path']}: {f['error']}" for f in failed_images])
            return False, f"No valid encodings. Errors: {error_details}", 0
            
    except Exception as e:
        return False, f"Training failed: {str(e)}", 0


@app.route("/train-face", methods=["POST"])
def train_face():
    """
    Endpoint to receive student face images and train face recognition model.

    Expected form data:
        - student_id: Unique identifier for the student
        - image: Image file (JPEG/PNG)

    Returns:
        JSON response with training status
    """
    try:
        # Validate student_id
        if "student_id" not in request.form:
            return jsonify({
                "error": "Missing student_id",
                "encodings_saved": False
            }), 400

        student_id = request.form["student_id"].strip()

        if not student_id:
            return jsonify({
                "error": "student_id cannot be empty",
                "encodings_saved": False
            }), 400

        # Validate image file
        if "image" not in request.files:
            return jsonify({
                "error": "Missing image file",
                "encodings_saved": False
            }), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "error": "No file selected",
                "encodings_saved": False
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                "error": f"File type not allowed. Accepted formats: {', '.join(ALLOWED_EXTENSIONS)}",
                "encodings_saved": False
            }), 400

        # Read image data
        image_data = file.read()

        if len(image_data) == 0:
            return jsonify({
                "error": "Empty image file",
                "encodings_saved": False
            }), 400

        # Save image to storage
        image_path = save_image_to_storage(student_id, image_data)
        print(f"✅ Image saved for student {student_id} at {image_path}")

        # Detect and validate face in the uploaded image
        success, result = detect_and_encode_face(image_path)

        if not success:
            # Remove the invalid image
            try:
                os.remove(image_path)
            except:
                pass

            return jsonify({
                "error": result,
                "encodings_saved": False
            }), 400

        # Train the face model with all images for this student
        train_success, train_message, encodings_count = train_face_model(student_id)

        if not train_success:
            return jsonify({
                "error": train_message,
                "encodings_saved": False
            }), 500

        return jsonify({
            "message": f"✅ Face data saved and trained successfully for student {student_id}",
            "encodings_saved": True,
            "encodings_count": encodings_count,
            "details": train_message
        }), 200

    except Exception as e:
        print(f"❌ Error in /train-face: {str(e)}")
        return jsonify({
            "error": f"Server error: {str(e)}",
            "encodings_saved": False
        }), 500


@app.route("/capture-face", methods=["POST"])
def capture_face():
    """
    Endpoint to capture a face image from webcam with position-based training.
    
    Expected JSON body:
        {
            "student_id": "114300",
            "position": "front"
        }
    
    Positions: front, left, right, up, down
    
    Returns:
        JSON response with capture status
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "failed",
                "message": "Invalid request: JSON body required"
            }), 400
        
        # Validate student_id
        student_id = data.get("student_id", "").strip()
        if not student_id:
            return jsonify({
                "status": "failed",
                "message": "Missing student_id"
            }), 400
        
        # Validate position
        position = data.get("position", "").strip().lower()
        if not position:
            return jsonify({
                "status": "failed",
                "message": "Missing position"
            }), 400
        
        if position not in VALID_POSITIONS:
            return jsonify({
                "status": "failed",
                "message": f"Invalid position. Must be one of: {', '.join(VALID_POSITIONS)}"
            }), 400
        
        # Capture ONE frame from webcam
        print(f"📷 Capturing face for student {student_id} at position: {position}")
        capture_success, capture_result = capture_frame_from_webcam()
        
        if not capture_success:
            return jsonify({
                "status": "failed",
                "position": position,
                "message": capture_result
            }), 400
        
        image_data = capture_result
        
        # Validate face in the captured image
        validation_success, validation_result = detect_and_validate_face(image_data)
        
        if not validation_success:
            # Get position-specific error message
            error_message = POSITION_GUIDANCE.get(position, 
                "Face capture failed. Please capture again.")
            
            return jsonify({
                "status": "failed",
                "position": position,
                "message": error_message
            }), 400
        
        # Image is valid - save FIRST sample with position-based filename
        # Get next image number for this position
        image_number = get_next_image_number(student_id, position)
        
        # Create student folder if it doesn't exist
        student_folder = os.path.join(DATASET_DIR, str(student_id))
        os.makedirs(student_folder, exist_ok=True)
        
        # Generate filename: studentID_position_number.jpg
        filename = f"{student_id}_{position}_{image_number}.jpg"
        image_path = os.path.join(student_folder, filename)
        
        # Save the FIRST image
        with open(image_path, "wb") as f:
            f.write(image_data)
        
        print(f"✅ Face captured and saved for student {student_id}: {filename}")
        
        # SILENTLY grab 2 MORE samples (200ms apart) - NO RESPONSE CHANGE
        for extra_idx in range(2):
            time.sleep(0.2)
            extra_success, extra_data = capture_frame_from_webcam()
            if extra_success:
                # Get next number for extra sample
                extra_number = get_next_image_number(student_id, position)
                extra_filename = f"{student_id}_{position}_{extra_number}.jpg"
                extra_path = os.path.join(student_folder, extra_filename)
                
                # Validate & save extra sample
                extra_valid, _ = detect_and_validate_face(extra_data)
                if extra_valid:
                    with open(extra_path, "wb") as f:
                        f.write(extra_data)
                    print(f"📸 Extra sample {extra_idx+1} saved: {extra_filename}")
                else:
                    print(f"⚠️ Extra sample {extra_idx+1} invalid face")
        
        return jsonify({
            "status": "success",
            "position": position,
            "file": filename  # Original first file only
        }), 200
        
    except Exception as e:
        print(f"❌ Error in /capture-face: {str(e)}")
        return jsonify({
            "status": "failed",
            "message": f"Server error: {str(e)}"
        }), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify server is running."""
    return jsonify({
        "status": "healthy",
        "message": "Face recognition training server is running"
    }), 200


@app.route("/student-info/<student_id>", methods=["GET"])
def get_student_info(student_id):
    """
    Get information about a student's face training data.

    Args:
        student_id: Unique student identifier

    Returns:
        JSON with student training status
    """
    try:
        image_paths = get_student_images(student_id)
        encodings = load_encodings_from_storage(student_id)

        return jsonify({
            "student_id": student_id,
            "images_count": len(image_paths),
            "encodings_count": len(encodings) if encodings else 0,
            "is_trained": encodings is not None and len(encodings) > 0
        }), 200

    except Exception as e:
        return jsonify({
            "error": f"Error retrieving student info: {str(e)}"
        }), 500


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Face Recognition Training Server Starting...")
    print("=" * 60)
    print(f"📁 Dataset directory: {os.path.abspath(DATASET_DIR)}")
    print(f"🔐 Encodings directory: {os.path.abspath(ENCODINGS_DIR)}")
    print("=" * 60)
    print("📡 Server running on http://0.0.0.0:5000")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5000, debug=False)
