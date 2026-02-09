# TODO: Complete Refactoring - Move Database Codes from main.py to database.py and Switch to main_new.py

## Steps to Complete

- [ ] Update database.py to import Pydantic models from models.py instead of defining them
- [ ] Move all remaining database-related API routes from main.py to database.py's router (students, teachers, classes, attendance, events, receipts, analytics, db views)
- [ ] Update main.py to import the router from database.py and include it in the FastAPI app
- [ ] Remove the moved routes and models from main.py, keeping only face recognition, camera, streaming, and non-database routes
- [ ] Ensure main.py imports necessary functions from database.py and models.py
- [ ] Rename main_new.py to main.py to make it the main entry point
- [ ] Test the application to ensure database operations work correctly
- [ ] Verify that all routes are accessible and functioning
