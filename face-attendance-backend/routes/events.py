"""
Event management routes.
"""
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status
from bson import ObjectId
from datetime import datetime

from database.connection import get_database
from models import Event, EventCreate, Receipt, ReceiptCreate
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/")
async def create_event(event: EventCreate) -> Dict[str, Any]:
    """Create a new event."""
    db = get_database()

    result = await db.events.insert_one(event.dict())
    return {"message": "Event created successfully", "event_id": str(result.inserted_id)}


@router.get("/")
async def get_events() -> Dict[str, List[Dict[str, Any]]]:
    """Get all events."""
    db = get_database()

    events = []
    async for event in db.events.find():
        event["_id"] = str(event["_id"])
        events.append(event)
    return {"events": events}


@router.get("/{event_id}")
async def get_event(event_id: str) -> Dict[str, Any]:
    """Get event by ID."""
    db = get_database()

    event = await db.events.find_one({"_id": ObjectId(event_id)})
    if not event:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")
    event["_id"] = str(event["_id"])
    return event


@router.put("/{event_id}")
async def update_event(event_id: str, event_update: Dict[str, Any]) -> Dict[str, str]:
    """Update event information."""
    db = get_database()

    result = await db.events.update_one(
        {"_id": ObjectId(event_id)},
        {"$set": event_update}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")
    return {"message": "Event updated successfully"}


@router.delete("/{event_id}")
async def delete_event(event_id: str) -> Dict[str, str]:
    """Delete an event."""
    db = get_database()

    result = await db.events.delete_one({"_id": ObjectId(event_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")
    return {"message": "Event deleted successfully"}


@router.post("/receipts")
async def submit_receipt(receipt: ReceiptCreate) -> Dict[str, Any]:
    """Submit a receipt for verification."""
    db = get_database()

    # Check if student exists
    student = await db.students.find_one({"student_id": receipt.student_id})
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")

    # Check if event exists
    event = await db.events.find_one({"_id": ObjectId(receipt.event_id)})
    if not event:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")

    # Check if receipt already exists for this student and event
    existing_receipt = await db.receipts.find_one({
        "student_id": receipt.student_id,
        "event_id": receipt.event_id
    })
    if existing_receipt:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                          detail="Receipt already submitted for this event")

    receipt_dict = receipt.dict()
    receipt_dict["status"] = "pending"
    receipt_dict["submitted_at"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    result = await db.receipts.insert_one(receipt_dict)
    return {"message": "Receipt submitted successfully", "receipt_id": str(result.inserted_id)}


@router.get("/receipts")
async def get_receipts(status_filter: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Get all receipts, optionally filtered by status."""
    db = get_database()

    query = {}
    if status_filter:
        query["status"] = status_filter

    receipts = []
    async for receipt in db.receipts.find(query).sort("submitted_at", -1):
        receipt["_id"] = str(receipt["_id"])
        receipts.append(receipt)
    return {"receipts": receipts}


@router.get("/receipts/student/{student_id}")
async def get_student_receipts(student_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """Get receipts for a specific student."""
    db = get_database()

    receipts = []
    async for receipt in db.receipts.find({"student_id": student_id}).sort("submitted_at", -1):
        receipt["_id"] = str(receipt["_id"])
        receipts.append(receipt)
    return {"receipts": receipts}


@router.put("/receipts/{receipt_id}/verify")
async def verify_receipt(receipt_id: str, verification_data: Dict[str, Any]) -> Dict[str, str]:
    """Verify or reject a receipt (admin only)."""
    db = get_database()

    status_value = verification_data.get("status")  # "verified" or "rejected"
    verified_by = verification_data.get("verified_by")

    if status_value not in ["verified", "rejected"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid status")

    result = await db.receipts.update_one(
        {"_id": ObjectId(receipt_id)},
        {"$set": {
            "status": status_value,
            "verified_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "verified_by": verified_by
        }}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Receipt not found")

    return {"message": f"Receipt {status_value} successfully"}


@router.delete("/receipts/{receipt_id}")
async def delete_receipt(receipt_id: str) -> Dict[str, str]:
    """Delete a receipt (admin only)."""
    db = get_database()

    result = await db.receipts.delete_one({"_id": ObjectId(receipt_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Receipt not found")
    return {"message": "Receipt deleted successfully"}
