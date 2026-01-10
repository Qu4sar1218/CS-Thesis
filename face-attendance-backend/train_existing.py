import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from face_training_server import train_face_model

# Get the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STUDENT_FACE_DATA_DIR = os.path.join(PROJECT_ROOT, "StudentFaceData")

def train_existing_students():
    """Train face models for all existing students in StudentFaceData folder."""
    if not os.path.exists(STUDENT_FACE_DATA_DIR):
        print(f"❌ StudentFaceData directory not found: {STUDENT_FACE_DATA_DIR}")
        return

    # Get all student folders
    student_folders = [f for f in os.listdir(STUDENT_FACE_DATA_DIR)
                      if os.path.isdir(os.path.join(STUDENT_FACE_DATA_DIR, f)) and f != "__pycache__"]

    if not student_folders:
        print("❌ No student folders found in StudentFaceData")
        return

    print(f"📁 Found {len(student_folders)} student folder(s): {', '.join(student_folders)}")

    for student_id in student_folders:
        print(f"\n🔄 Training face model for student {student_id}...")
        success, message, count = train_face_model(student_id)
        if success:
            print(f"✅ {message}")
        else:
            print(f"❌ Failed to train for {student_id}: {message}")

if __name__ == "__main__":
    train_existing_students()
