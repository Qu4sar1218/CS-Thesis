from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
import face_recognition
import pickle
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, "Images")
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
    Save face encodings locally for now.
    
    TO MIGRATE TO FIRESTORE:
    1. Store encodings as a document in Firestore:
       db.collection('face_encodings').document(student_id).set({
           'encodings': encodings.tolist(),  # Convert numpy to list
           'updated_at': datetime.now()
       })
    2. Remove the pickle file approach
    3. Update load_encodings_from_storage() accordingly
    
    Args:
        student_id: Unique student identifier
        encodings: List of face encoding arrays
        
    Returns:
        str: Path where encodings were saved
    """
    encoding_path = os.path.join(ENCODINGS_DIR, f"{student_id}.pkl")
    
    with open(encoding_path, "wb") as f:
        pickle.dump(encodings, f)
    
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


def detect_and_encode_face(image_path):
    """
    Detect face in image and generate encoding.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        tuple: (success: bool, encoding: np.array or error_message: str)
    """
    try:
        # Load the image
        image = face_recognition.load_image_file(image_path)
        
        # Detect face locations
        face_locations = face_recognition.face_locations(image, model="hog")
        
        if len(face_locations) == 0:
            return False, "No face detected in the image"
        
        if len(face_locations) > 1:
            return False, f"Multiple faces detected ({len(face_locations)}). Please ensure only one face is visible"
        
        # Generate face encoding
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        if len(face_encodings) == 0:
            return False, "Could not generate face encoding"
        
        return True, face_encodings[0]
    
    except Exception as e:
        return False, f"Error processing image: {str(e)}"


def train_face_model(student_id):
    """
    Process all images for a student and generate face encodings.
    
    This function:
    1. Retrieves all images for the student
    2. Detects and encodes faces in each image
    3. Saves the encodings for future recognition
    
    Args:
        student_id: Unique student identifier
        
    Returns:
        tuple: (success: bool, message: str, encodings_count: int)
    """
    try:
        # Get all images for this student
        image_paths = get_student_images(student_id)
        
        if not image_paths:
            return False, "No images found for this student", 0
        
        encodings = []
        failed_images = []
        
        # Process each image
        for image_path in image_paths:
            success, result = detect_and_encode_face(image_path)
            
            if success:
                encodings.append(result)
            else:
                failed_images.append({
                    "path": os.path.basename(image_path),
                    "error": result
                })
        
        if not encodings:
            error_details = "; ".join([f"{f['path']}: {f['error']}" for f in failed_images])
            return False, f"No valid face encodings generated. Errors: {error_details}", 0
        
        # Save encodings to storage
        save_encodings_to_storage(student_id, encodings)
        
        success_msg = f"Successfully trained {len(encodings)} face encoding(s) for student {student_id}"
        if failed_images:
            success_msg += f" ({len(failed_images)} image(s) failed)"
        
        return True, success_msg, len(encodings)
    
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
    
    app.run(host="0.0.0.0", port=5000, debug=True)