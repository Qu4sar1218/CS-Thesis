"""
Database view routes for administrative database inspection.
"""
from typing import Dict, List, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from database.connection import get_database

router = APIRouter()


@router.get("/db/collections")
async def get_db_collections() -> Dict[str, List[str]]:
    """Get list of all collections in the database."""
    db = get_database()

    try:
        collections = await db.list_collection_names()
        return {"collections": collections}
    except Exception as e:
        return JSONResponse({"error": "Failed to list collections"}, status_code=500)


@router.get("/db/{collection}")
async def get_collection_data(collection: str, limit: int = 10) -> Dict[str, Any]:
    """Get data from a specific collection."""
    db = get_database()

    try:
        collection_obj = db[collection]
        documents = []
        async for doc in collection_obj.find().limit(limit):
            doc["_id"] = str(doc["_id"])
            documents.append(doc)
        return {"collection": collection, "documents": documents, "limit": limit}
    except Exception as e:
        return JSONResponse({"error": f"Failed to fetch data from {collection}"}, status_code=500)
