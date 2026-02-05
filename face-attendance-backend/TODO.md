# TODO: Move Database Codes from main.py to database.py

## Steps to Complete

- [x] Move all Pydantic models (UserBase, StudentBase, etc.) from main.py to database.py
- [x] Move authentication utilities (verify_password, get_password_hash, create_access_token) from main.py to database.py
- [x] Move attendance functions (save_attendance_to_db, update_attendance_status) from main.py to database.py
- [x] Create an APIRouter in database.py and move all database-related API routes (students, teachers, classes, attendance, events, receipts, analytics, db views) to it
- [x] Update main.py to import the router from database.py and include it in the FastAPI app
- [x] Remove the moved code from main.py and add necessary imports
- [x] Ensure face recognition, camera, streaming, and non-database routes remain in main.py
- [x] Test the application to ensure database operations work correctly
- [x] Verify that all routes are accessible and functioning
