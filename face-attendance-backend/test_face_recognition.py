#!/usr/bin/env python3
"""Test script for face recognition functionality."""

try:
    import face_recognition
    print("✅ face_recognition imported successfully")
    print(f"Version: {getattr(face_recognition, '__version__', 'unknown')}")

    # Test basic functionality
    print("Testing face_recognition functionality...")

    # Create a simple test
    test_image = face_recognition.load_image_file("test.jpg")  # This will fail if no image, but we can check if the function exists
    print("✅ face_recognition.load_image_file function available")

except ImportError as e:
    print(f"❌ ImportError: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")
