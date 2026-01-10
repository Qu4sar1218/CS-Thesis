# Student Face Training Data

This folder contains face images for student recognition. Only student faces should be stored here.

## Folder Structure

```
StudentFaceData/
├── README.md
├── student_encodings.pkl  (auto-generated)
├── 117878/               (student ID folders)
│   ├── face1.jpg
│   ├── face2.jpg
│   └── face3.jpg
├── 111234/               (another student)
│   ├── photo1.jpg
│   └── photo2.jpg
└── ...
```

## How to Add Student Faces

1. **Create a folder** named with the student's ID (e.g., `117878`)
2. **Add face images** to that folder (JPG, JPEG, or PNG format)
3. **Use clear, well-lit photos** with only the student's face visible
4. **Multiple angles** are recommended for better recognition

## Usage with Python Script

Run the face recognition script:

```bash
python face_recognition_script.py
```

The script will:
- Automatically load/generate face encodings from images
- Start webcam recognition
- Display recognized student names and IDs
- Save encodings for faster future loading

## Important Notes

- Only student faces should be in this folder
- Each student should have their own subfolder
- Images should be clear and well-lit
- Multiple images per student improve recognition accuracy
- The system automatically updates when new images are added

## Troubleshooting

- **No faces detected**: Ensure images are clear and well-lit
- **Wrong person recognized**: Add more images or check image quality
- **Script won't start**: Make sure you have the required Python packages installed

## Required Python Packages

```bash
pip install opencv-python face-recognition numpy
