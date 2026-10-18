"""
db.py - Secure MongoDB Atlas Connection and Audit Logging Module
----------------------------------------------------------------
This module manages the connection to MongoDB Atlas for Restock-Predictor,
logging CSV upload summaries, dataset characteristics, and execution metadata.

Security & Architectural Principles:
- 12-Factor App: Credentials ingested exclusively via MONGO_URI environment variable.
- Non-blocking: Strict socket & server selection timeouts (3500ms) prevent UI freezing.
- Zero-Failure Demo Resilience: If MONGO_URI is omitted or Atlas IP whitelist blocks
  access during live demos, logs seamlessly route to an in-memory session store.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

# Attempt to load dotenv if available in local environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure module-level logger
logger = logging.getLogger("restock_predictor.db")

# In-memory session fallback store for when MongoDB Atlas is unreachable or unconfigured
_SESSION_FALLBACK_LOGS: List[Dict[str, Any]] = []

def get_mongo_uri() -> Optional[str]:
    """Retrieve the MongoDB connection URI from environment variables."""
    return os.getenv("MONGO_URI", "").strip() or None

def get_db_name() -> str:
    """Retrieve the target database name, defaulting to 'restock_predictor'."""
    return os.getenv("MONGO_DB_NAME", "restock_predictor").strip()

def get_collection_name() -> str:
    """Retrieve the target collection name, defaulting to 'upload_logs'."""
    return os.getenv("MONGO_COLLECTION_NAME", "upload_logs").strip()

def get_mongo_client():
    """
    Instantiate and return a MongoClient instance with tight timeouts.
    Returns None if pymongo is not installed or MONGO_URI is missing.
    """
    mongo_uri = get_mongo_uri()
    if not mongo_uri:
        return None

    try:
        from pymongo import MongoClient
        from pymongo.server_api import ServerApi

        # Set aggressive timeouts (3.5s) to avoid UI lockups if Atlas IP whitelist is pending
        client = MongoClient(
            mongo_uri,
            server_api=ServerApi('1'),
            serverSelectionTimeoutMS=3500,
            connectTimeoutMS=3500,
            socketTimeoutMS=3500,
            appname="RestockPredictor-DEVHacktoberfest"
        )
        return client
    except ImportError:
        logger.warning("pymongo is not installed. Running in local fallback logging mode.")
        return None
    except Exception as exc:
        logger.error(f"Failed to initialize MongoClient: {exc}")
        return None

def test_connection() -> Dict[str, Any]:
    """
    Test MongoDB Atlas connectivity and ping the cluster.
    Returns a status dictionary detailing connectivity state.
    """
    mongo_uri = get_mongo_uri()
    if not mongo_uri:
        return {
            "connected": False,
            "status": "unconfigured",
            "message": "MONGO_URI not configured. Running in resilient local session audit mode."
        }

    try:
        client = get_mongo_client()
        if client is None:
            return {
                "connected": False,
                "status": "driver_missing",
                "message": "MongoDB driver (pymongo) unavailable. Running in local fallback mode."
            }

        # Issue ping command to verify deployment credentials and IP access list
        client.admin.command('ping')
        db_name = get_db_name()
        coll_name = get_collection_name()
        
        # Count existing docs
        doc_count = client[db_name][coll_name].count_documents({})
        return {
            "connected": True,
            "status": "connected",
            "database": db_name,
            "collection": coll_name,
            "doc_count": doc_count,
            "message": f"Successfully connected to MongoDB Atlas cluster ({db_name}.{coll_name})."
        }
    except Exception as exc:
        logger.warning(f"MongoDB connection test failed: {exc}")
        return {
            "connected": False,
            "status": "connection_error",
            "message": f"MongoDB Atlas unreachable: {str(exc)[:120]} (Check Network/IP Access List)"
        }

def log_upload_summary(
    filename: str,
    row_count: int,
    item_count: Optional[int] = None,
    items: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Log an audit record of the uploaded sales CSV to MongoDB Atlas.
    
    Document Schema:
    - timestamp: ISO 8601 UTC timestamp
    - filename: Name of uploaded file
    - row_count: Number of valid data rows parsed
    - item_count: Unique number of Item_IDs detected
    - items: List of unique item identifier strings
    - metadata: Additional context (date ranges, user agent, engine used)
    - status: 'logged_to_atlas' | 'fallback_session_store'
    """
    timestamp = datetime.now(timezone.utc)
    log_doc = {
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "filename": filename,
        "row_count": int(row_count),
        "item_count": int(item_count) if item_count is not None else 0,
        "items": list(items or []),
        "metadata": metadata or {},
        "created_at": timestamp
    }

    client = get_mongo_client()
    if client is not None:
        try:
            db_name = get_db_name()
            coll_name = get_collection_name()
            db = client[db_name]
            collection = db[coll_name]

            insert_result = collection.insert_one(log_doc)
            log_doc["_id"] = str(insert_result.inserted_id)
            log_doc["status"] = "logged_to_atlas"
            logger.info(f"Logged upload '{filename}' ({row_count} rows) to Atlas. ID: {log_doc['_id']}")
            return {
                "success": True,
                "storage": "mongodb_atlas",
                "id": str(insert_result.inserted_id),
                "record": log_doc
            }
        except Exception as exc:
            logger.warning(f"Failed to insert record into MongoDB Atlas: {exc}. Using fallback log store.")
            log_doc["error_detail"] = str(exc)

    # Resilient fallback: preserve log in session memory so small business owners never crash
    log_doc["_id"] = f"local-{len(_SESSION_FALLBACK_LOGS) + 1}"
    log_doc["status"] = "fallback_session_store"
    _SESSION_FALLBACK_LOGS.insert(0, log_doc)
    
    return {
        "success": True,
        "storage": "fallback_session_store",
        "id": log_doc["_id"],
        "record": log_doc,
        "note": "Logged to session audit buffer (MONGO_URI not configured or cluster unreachable)."
    }

def get_recent_logs(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Retrieve the most recent upload logs for the audit table in the Streamlit UI.
    """
    client = get_mongo_client()
    if client is not None:
        try:
            db_name = get_db_name()
            coll_name = get_collection_name()
            collection = client[db_name][coll_name]
            cursor = collection.find().sort("created_at", -1).limit(limit)
            results = []
            for doc in cursor:
                doc["_id"] = str(doc.get("_id", ""))
                if "created_at" in doc and hasattr(doc["created_at"], "isoformat"):
                    doc["created_at"] = doc["created_at"].isoformat()
                results.append(doc)
            if results:
                return results
        except Exception as exc:
            logger.warning(f"Could not retrieve logs from MongoDB Atlas: {exc}")

    # Return in-memory fallback logs if Atlas was unconfigured/unreachable
    return _SESSION_FALLBACK_LOGS[:limit]
