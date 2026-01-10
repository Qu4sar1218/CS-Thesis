#!/usr/bin/env python3
"""
Receipt Population Script for Face Attendance System
Run this script to populate the receipts collection in MongoDB with data from a JSON file
"""

import asyncio
import json
import os
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from bson import ObjectId

async def populate_receipts_from_file(json_file_path: str):
    """Populate receipts collection from a JSON file."""

    # Check if file exists
    if not os.path.exists(json_file_path):
        print(f"❌ Error: File {json_file_path} not found")
        return

    # Connect to MongoDB
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["InterACTS"]

    print("🚀 Populating receipts collection...")

    try:
        # Read JSON file
        with open(json_file_path, 'r') as f:
            receipts_data = json.load(f)

        receipts_collection = db.receipts

        # Validate and prepare receipts data
        valid_receipts = []
        for receipt in receipts_data:
            # Validate required fields
            required_fields = ['student_id', 'event_id', 'transaction_id', 'receipt_image']
            if not all(field in receipt for field in required_fields):
                print(f"⚠️  Skipping receipt missing required fields: {receipt}")
                continue

            # Set defaults
            receipt.setdefault('status', 'pending')
            receipt.setdefault('submitted_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

            # Convert event_id to ObjectId if it's a string
            if isinstance(receipt['event_id'], str) and len(receipt['event_id']) == 24:
                try:
                    receipt['event_id'] = ObjectId(receipt['event_id'])
                except:
                    pass  # Keep as string if not valid ObjectId

            valid_receipts.append(receipt)

        if not valid_receipts:
            print("❌ No valid receipts found in the file")
            return

        # Insert receipts
        result = await receipts_collection.insert_many(valid_receipts)
        print(f"✅ Successfully inserted {len(result.inserted_ids)} receipts")

        # Show inserted receipt IDs
        for i, receipt_id in enumerate(result.inserted_ids):
            print(f"   Receipt {i+1}: {receipt_id}")

    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON file: {e}")
    except Exception as e:
        print(f"❌ Error populating receipts: {e}")
    finally:
        client.close()

async def populate_sample_receipts():
    """Populate with sample receipts for testing."""

    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["InterACTS"]

    print("🚀 Populating sample receipts...")

    try:
        # Get some events to reference
        events_collection = db.events
        events = await events_collection.find().limit(3).to_list(length=3)

        if not events:
            print("❌ No events found. Please run init_db.py first to create events.")
            return

        receipts_collection = db.receipts

        sample_receipts = [
            {
                "student_id": "114001",
                "event_id": str(events[0]['_id']),
                "transaction_id": "001234",
                "receipt_image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAAIAAoDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAhEAACAQMDBQAAAAAAAAAAAAABAgMABAUGIWGRkqGx0f/EABUBAQEAAAAAAAAAAAAAAAAAAAMF/8QAGhEAAgIDAAAAAAAAAAAAAAAAAAECEgMRkf/aAAwDAQACEQMRAD8AltJagyeH0AthI5xdrLcNM91BF5pX2HaH9bcfaSXWGaRmknyJckliyjqTzSlT54b6bk+h0R+IRjWjBqO6O2mhP//Z",
                "status": "pending",
                "submitted_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                "student_id": "114002",
                "event_id": str(events[1]['_id']) if len(events) > 1 else str(events[0]['_id']),
                "transaction_id": "005678",
                "receipt_image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAAIAAoDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAhEAACAQMDBQAAAAAAAAAAAAABAgMABAUGIWGRkqGx0f/EABUBAQEAAAAAAAAAAAAAAAAAAAMF/8QAGhEAAgIDAAAAAAAAAAAAAAAAAAECEgMRkf/aAAwDAQACEQMRAD8AltJagyeH0AthI5xdrLcNM91BF5pX2HaH9bcfaSXWGaRmknyJckliyjqTzSlT54b6bk+h0R+IRjWjBqO6O2mhP//Z",
                "status": "verified",
                "submitted_at": (datetime.now().replace(hour=16, minute=45)).strftime('%Y-%m-%d %H:%M:%S'),
                "verified_at": (datetime.now().replace(hour=9, minute=15)).strftime('%Y-%m-%d %H:%M:%S'),
                "verified_by": "admin"
            }
        ]

        result = await receipts_collection.insert_many(sample_receipts)
        print(f"✅ Successfully inserted {len(result.inserted_ids)} sample receipts")

    except Exception as e:
        print(f"❌ Error populating sample receipts: {e}")
    finally:
        client.close()

def print_usage():
    """Print usage instructions."""
    print("Usage:")
    print("  python populate_receipts.py sample    # Populate with sample receipts")
    print("  python populate_receipts.py file <path_to_json_file>  # Populate from JSON file")
    print("")
    print("JSON file format example:")
    print("""
[
    {
        "student_id": "114001",
        "event_id": "507f1f77bcf86cd799439011",
        "transaction_id": "001234",
        "receipt_image": "data:image/jpeg;base64,...",
        "status": "pending",
        "submitted_at": "2024-12-20 14:30:00"
    }
]
""")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1]

    if command == "sample":
        asyncio.run(populate_sample_receipts())
    elif command == "file":
        if len(sys.argv) < 3:
            print("❌ Error: Please provide the path to the JSON file")
            print_usage()
            sys.exit(1)
        json_file = sys.argv[2]
        asyncio.run(populate_receipts_from_file(json_file))
    else:
        print(f"❌ Unknown command: {command}")
        print_usage()
        sys.exit(1)
