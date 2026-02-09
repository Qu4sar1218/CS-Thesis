"""
Attendance management utilities and routes.
"""
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, status
from bson import ObjectId

from database.connection import get_database
from models import Attendance, AttendanceCreate
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


async def save_attendance_to_db(record: Dict[str, Any]) -> None:
    """Save attendance record to database."""
    try:
        db = get_database()
        attendance_collection = db.attendance
        result = await attendance_collection.insert_one(record)
        logger.info(f"✅ Attendance saved to DB: {record['name']}")
    except Exception as e:
        logger.error(f"❌ Failed to save attendance to DB: {e}")


async def update_attendance_status(student_id: str, class_id: str, attendance_status: str) -> None:
    """Update attendance status for a student in a class."""
    try:
        db = get_database()
        current_date = datetime.now().strftime('%Y-%m-%d')
        attendance_collection = db.attendance

        # Update existing absent record to present
        result = await attendance_collection.update_one(
            {
                "student_id": student_id,
                "class_id": class_id,
                "date": current_date,
                "status": "absent"
            },
            {"$set": {"status": attendance_status}}
        )

        if result.modified_count > 0:
            logger.info(f"✅ Updated attendance status for {student_id} in class {class_id} to {attendance_status}")
        else:
            logger.warning(f"⚠️ No absent record found to update for {student_id} in class {class_id}")

    except Exception as e:
        logger.error(f"❌ Failed to update attendance status for {student_id}: {e}")


@router.post("/check-in")
async def check_in(attendance_data: Dict[str, Any]) -> Dict[str, Any]:
    """Manual check-in for attendance."""
    db = get_database()

    student_id = attendance_data["student_id"]
    class_id = attendance_data["class_id"]

    # Check if student exists
    student = await db.students.find_one({"student_id": student_id})
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")

    # Check if class exists
    class_doc = await db.classes.find_one({"_id": ObjectId(class_id)})
    if not class_doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Class not found")

    current_time = datetime.now()
    attendance_record = {
        "student_id": student_id,
        "class_id": class_id,
        "date": current_time.strftime('%Y-%m-%d'),
        "check_in_time": current_time.strftime('%H:%M:%S'),
        "status": "present",
        "timestamp": current_time.strftime('%Y-%m-%d %H:%M:%S')
    }

    result = await db.attendance.insert_one(attendance_record)
    return {"message": "Check-in recorded", "attendance_id": str(result.inserted_id)}


@router.post("/check-out")
async def check_out(attendance_data: Dict[str, Any]) -> Dict[str, Any]:
    """Manual check-out for attendance."""
    db = get_database()

    student_id = attendance_data["student_id"]
    class_id = attendance_data["class_id"]

    current_time = datetime.now()
    result = await db.attendance.update_one(
        {
            "student_id": student_id,
            "class_id": class_id,
            "date": current_time.strftime('%Y-%m-%d'),
            "check_out_time": {"$exists": False}
        },
        {"$set": {"check_out_time": current_time.strftime('%H:%M:%S')}}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No active check-in found")

    return {"message": "Check-out recorded"}


@router.post("/initialize-class/{class_id}")
async def initialize_class_attendance(class_id: str) -> Dict[str, Any]:
    """Initialize attendance records for all enrolled students in a class as absent for today."""
    db = get_database()

    # Check if class exists
    class_doc = await db.classes.find_one({"_id": ObjectId(class_id)})
    if not class_doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Class not found")

    enrolled_students = class_doc.get("enrolled_students", [])
    if not enrolled_students:
        return {"message": "No enrolled students found for this class"}

    current_date = datetime.now().strftime('%Y-%m-%d')
    current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    attendance_collection = db.attendance
    initialized_count = 0

    for student_id in enrolled_students:
        # Check if student already has a record for today
        existing_record = await attendance_collection.find_one({
            "student_id": student_id,
            "class_id": class_id,
            "date": current_date
        })

        if not existing_record:
            # Create absent record
            attendance_record = {
                "student_id": student_id,
                "class_id": class_id,
                "date": current_date,
                "status": "absent",
                "timestamp": current_timestamp,
                "subject": class_doc.get("class_name", "Unknown Subject")
            }
            await attendance_collection.insert_one(attendance_record)
            initialized_count += 1

    return {"message": f"Initialized attendance for {initialized_count} students"}


@router.get("/db")
async def get_attendance_from_db() -> Dict[str, List[Dict[str, Any]]]:
    """Get attendance records from database."""
    try:
        db = get_database()
        attendance_collection = db.attendance
        records = []
        async for record in attendance_collection.find().sort("timestamp", -1).limit(100):
            record['_id'] = str(record['_id'])  # Convert ObjectId to string
            records.append(record)
        return {"attendance": records}
    except Exception as e:
        logger.error(f"❌ Failed to fetch attendance from DB: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                          detail="Failed to fetch attendance")


@router.get("/analytics/summary")
async def get_attendance_summary(date_from: Optional[str] = None, date_to: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Get attendance summary statistics."""
    db = get_database()

    query = {}
    if date_from and date_to:
        query["date"] = {"$gte": date_from, "$lte": date_to}

    pipeline = [
        {"$match": query},
        {"$group": {
            "_id": {
                "date": "$date",
                "class_id": "$class_id"
            },
            "present_count": {"$sum": 1},
            "total_students": {"$addToSet": "$student_id"}
        }},
        {"$project": {
            "date": "$_id.date",
            "class_id": "$_id.class_id",
            "present_count": 1,
            "enrolled_count": {"$size": "$total_students"}
        }}
    ]

    results = []
    async for doc in db.attendance.aggregate(pipeline):
        results.append(doc)

    return {"summary": results}


@router.get("/analytics/student/{student_id}")
async def get_student_attendance(student_id: str, date_from: Optional[str] = None, date_to: Optional[str] = None) -> Dict[str, Any]:
    """Get attendance records for a specific student."""
    db = get_database()

    query = {"student_id": student_id}
    if date_from and date_to:
        query["date"] = {"$gte": date_from, "$lte": date_to}

    records = []
    async for record in db.attendance.find(query).sort("date", -1):
        record["_id"] = str(record["_id"])
        records.append(record)

    return {"student_id": student_id, "attendance": records}


@router.get("/analytics/student/{student_id}/insights")
async def get_student_attendance_insights(student_id: str) -> Dict[str, Any]:
    """Get comprehensive attendance insights for a student."""
    db = get_database()

    # Get total attendance summary
    pipeline = [
        {"$match": {"student_id": student_id}},
        {"$group": {
            "_id": None,
            "total_sessions": {"$sum": 1},
            "present_count": {"$sum": {"$cond": [{"$eq": ["$status", "present"]}, 1, 0]}},
            "absent_count": {"$sum": {"$cond": [{"$eq": ["$status", "absent"]}, 1, 0]}}
        }}
    ]

    summary_result = await db.attendance.aggregate(pipeline).to_list(length=1)
    summary = summary_result[0] if summary_result else {"total_sessions": 0, "present_count": 0, "absent_count": 0}

    total_sessions = summary["total_sessions"]
    attendance_percentage = (summary["present_count"] / total_sessions * 100) if total_sessions > 0 else 0

    # Determine attendance status
    if attendance_percentage >= 90:
        status = "Good Standing"
    elif 75 <= attendance_percentage < 90:
        status = "Warning"
    else:
        status = "At Risk"

    # Get subject-based breakdown
    subject_pipeline = [
        {"$match": {"student_id": student_id}},
        {"$lookup": {
            "from": "classes",
            "localField": "class_id",
            "foreignField": "_id",
            "as": "class_info"
        }},
        {"$unwind": {"path": "$class_info", "preserveNullAndEmptyArrays": True}},
        {"$group": {
            "_id": {
                "subject": {"$ifNull": ["$class_info.class_name", "$subject"]},
                "class_id": "$class_id"
            },
            "total_sessions": {"$sum": 1},
            "present_count": {"$sum": {"$cond": [{"$eq": ["$status", "present"]}, 1, 0]}},
            "absent_count": {"$sum": {"$cond": [{"$eq": ["$status", "absent"]}, 1, 0]}}
        }},
        {"$project": {
            "subject": "$_id.subject",
            "total_sessions": 1,
            "present_count": 1,
            "absent_count": 1,
            "attendance_percentage": {
                "$multiply": [{"$divide": ["$present_count", "$total_sessions"]}, 100]
            }
        }},
        {"$sort": {"subject": 1}}
    ]

    subject_breakdown = []
    async for doc in db.attendance.aggregate(subject_pipeline):
        subject_breakdown.append({
            "subject": doc["subject"] or "Unknown Subject",
            "attendance_percentage": round(doc["attendance_percentage"], 1),
            "present_count": doc["present_count"],
            "absent_count": doc["absent_count"]
        })

    # Get face recognition activity log (assuming face_logs collection exists)
    face_logs = []
    try:
        face_logs_collection = db.face_logs
        async for log in face_logs_collection.find({"student_id": student_id}).sort("timestamp", -1).limit(10):
            face_logs.append({
                "date": log.get("date", ""),
                "time": log.get("time", ""),
                "subject": log.get("subject", "N/A"),
                "result": log.get("result", "Unknown")
            })
    except:
        # If face_logs collection doesn't exist, use attendance records with recognition data
        recognition_pipeline = [
            {"$match": {"student_id": student_id, "status": "present"}},
            {"$lookup": {
                "from": "classes",
                "localField": "class_id",
                "foreignField": "_id",
                "as": "class_info"
            }},
            {"$unwind": {"path": "$class_info", "preserveNullAndEmptyArrays": True}},
            {"$project": {
                "date": 1,
                "time": {"$ifNull": ["$check_in_time", "N/A"]},
                "subject": {"$ifNull": ["$class_info.class_name", "$subject"]},
                "result": "Verified"  # Assuming present means verified
            }},
            {"$sort": {"timestamp": -1}},
            {"$limit": 10}
        ]

        async for doc in db.attendance.aggregate(recognition_pipeline):
            face_logs.append({
                "date": doc["date"],
                "time": doc["time"],
                "subject": doc["subject"] or "Unknown Subject",
                "result": doc["result"]
            })

    # Generate smart feedback
    feedback = generate_attendance_feedback(attendance_percentage, status)

    return {
        "attendance_summary": {
            "total_sessions": total_sessions,
            "present_count": summary["present_count"],
            "absent_count": summary["absent_count"],
            "attendance_percentage": round(attendance_percentage, 1),
            "status": status
        },
        "subject_breakdown": subject_breakdown,
        "face_recognition_logs": face_logs,
        "smart_feedback": feedback
    }


def generate_attendance_feedback(attendance_percentage: float, status: str) -> str:
    """Generate rule-based attendance feedback message."""
    if status == "Good Standing":
        return "Excellent attendance. Keep it up!"
    elif status == "Warning":
        return "Your attendance is in the warning zone. Please maintain regular attendance."
    else:  # At Risk
        return "Your attendance is at risk. Please attend classes regularly to avoid academic penalties."


@router.get("/analytics/class/{class_id}")
async def get_class_attendance(class_id: str, date_from: Optional[str] = None, date_to: Optional[str] = None) -> Dict[str, Any]:
    """Get attendance records for a specific class."""
    db = get_database()

    query = {"class_id": class_id}
    if date_from and date_to:
        query["date"] = {"$gte": date_from, "$lte": date_to}

    records = []
    async for record in db.attendance.find(query).sort("date", -1):
        record["_id"] = str(record["_id"])
        records.append(record)

    return {"class_id": class_id, "attendance": records}
