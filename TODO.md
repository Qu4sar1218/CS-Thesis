# Add Event Management to Admin Dashboard

## Problem
Currently, events are hardcoded in the database initialization. Admins need the ability to add/manage events through the UI instead of relying on hardcoded data.

## Solution
1. Add "Add Event" functionality to admin dashboard
2. Create event management page/component
3. Update backend API to support event creation
4. Remove hardcoded events from database initialization

## Tasks
- [x] Add "Add Event" button to AdminDashboard navigation
- [x] Create EventManagement.jsx component for adding events
- [x] Add event management route in App.jsx
- [x] Update backend API to support POST /events endpoint
- [x] Remove hardcoded events from init_db.py (make them sample data only)
- [x] Remove price field from EventManagement component
- [ ] Test event creation and verify it appears in StudentReceiptSubmission
- [x] Fix sidebar scrolling in all dashboards
