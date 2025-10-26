# TODO: Fix Face Detection and Recognition Issues

## Step 1: Update face_training_server.py ✅ DONE
- Change DATASET_DIR to align with main.py's FACE_DATA_DIR (use os.path.join(PROJECT_ROOT, "Images")) ✅
- Change ENCODINGS_DIR to os.path.join(BACKEND_ROOT, "data", "encodings") ✅
- Ensure multiple encodings per student are saved correctly (already does list) ✅

## Step 2: Modify main.py ✅ DONE
- Ensure loading of encodings handles multiple (already checks for list and takes first) ✅
- Adjust recognition parameters: Increase resize factor from 0.25 to 0.5 for better accuracy, lower FACE_MATCH_THRESHOLD to 0.5 ✅
- Verify FACE_DATA_DIR discovery ✅

## Step 3: Retrain Face Data
- After code changes, user needs to retrain faces using the updated server

## Step 4: Test Recognition
- Run the system and verify face detection and recognition works
