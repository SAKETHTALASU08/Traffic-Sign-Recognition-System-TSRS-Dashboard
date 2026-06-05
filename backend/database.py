"""
database.py — SQLite Database Module for Traffic Sign Recognition System
=========================================================================

This module handles all database operations for storing and retrieving
prediction history. We use SQLite because:
  1. Zero configuration — no separate DB server needed
  2. Perfect for single-server deployments with moderate write loads
  3. File-based, so backups are trivial (just copy the .db file)

Thread Safety:
  SQLite connections are NOT thread-safe by default. Since FastAPI uses
  a thread pool for sync endpoints, we set check_same_thread=False and
  rely on SQLite's internal serialization (WAL mode) for concurrent access.
  Each function opens its own connection via a context manager to avoid
  sharing connections across threads.
"""

import sqlite3
import os
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Store the database file inside the backend/ directory.
# os.path.dirname(__file__) resolves to the directory containing this script,
# so the DB file lives at backend/predictions.db regardless of where the
# server is launched from.
DB_PATH = os.path.join(os.path.dirname(__file__), "predictions.db")


# ---------------------------------------------------------------------------
# Connection Helper
# ---------------------------------------------------------------------------

@contextmanager
def get_db_connection():
    """
    Context manager that yields a SQLite connection and guarantees cleanup.

    Key settings:
      - check_same_thread=False: allows the connection to be used from
        FastAPI's thread-pool workers (different OS threads).
      - row_factory = sqlite3.Row: lets us access columns by name, making
        the code more readable and less error-prone than positional indexing.

    Usage:
        with get_db_connection() as conn:
            cursor = conn.execute("SELECT ...")
            rows = cursor.fetchall()
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    # sqlite3.Row makes rows behave like dicts — access by column name
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        # Always close the connection when done, even if an exception occurs.
        # This releases the file lock and frees resources.
        conn.close()


# ---------------------------------------------------------------------------
# Schema Initialization
# ---------------------------------------------------------------------------

def init_db() -> None:
    """
    Create the predictions table if it does not already exist.

    Called once during application startup (in the FastAPI lifespan handler).
    Using IF NOT EXISTS makes this idempotent — safe to call multiple times
    without risking data loss or duplicate-table errors.

    Table schema:
      - id:           Auto-incrementing primary key for unique identification.
      - timestamp:    ISO-8601 string of when the prediction was made (UTC).
      - class_id:     Integer ID of the predicted traffic sign class (0-42).
      - class_name:   Human-readable name of the predicted class.
      - confidence:   Model's confidence score for the prediction (0.0–1.0).
      - latency_ms:   Inference time in milliseconds (for performance monitoring).
      - thumbnail:    Base64-encoded 64×64 JPEG thumbnail of the input image.
                      Stored as TEXT to keep the schema simple; in a production
                      system at scale, you might store images in object storage
                      and keep only a URL here.
      - created_at:   Server-side timestamp (SQLite's CURRENT_TIMESTAMP).
                      Acts as a fallback / audit field separate from the
                      application-level 'timestamp'.
    """
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                class_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                latency_ms REAL NOT NULL,
                thumbnail TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Commit the DDL statement so the table persists on disk.
        conn.commit()
    print(f"[database] Initialized SQLite database at {DB_PATH}")


# ---------------------------------------------------------------------------
# Write Operations
# ---------------------------------------------------------------------------

def log_prediction(
    class_id: int,
    class_name: str,
    confidence: float,
    latency_ms: float,
    thumbnail_base64: str | None = None,
) -> int:
    """
    Insert a new prediction record into the database.

    Parameters
    ----------
    class_id : int
        Numeric class index (0–42) from the GTSRB dataset.
    class_name : str
        Human-readable label, e.g. "Stop", "Yield".
    confidence : float
        Softmax probability of the predicted class (0.0–1.0).
    latency_ms : float
        Wall-clock inference time in milliseconds.
    thumbnail_base64 : str or None
        Optional base64-encoded thumbnail for display in the history UI.

    Returns
    -------
    int
        The auto-generated row ID of the newly inserted record.
    """
    # Generate a UTC ISO-8601 timestamp for consistency across time zones.
    timestamp = datetime.now(timezone.utc).isoformat()

    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions (timestamp, class_id, class_name, confidence, latency_ms, thumbnail)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (timestamp, class_id, class_name, confidence, latency_ms, thumbnail_base64),
        )
        conn.commit()

        # lastrowid gives us the AUTOINCREMENT id of the row we just inserted.
        row_id = cursor.lastrowid
        print(f"[database] Logged prediction #{row_id}: {class_name} ({confidence:.2%})")
        return row_id


# ---------------------------------------------------------------------------
# Read Operations — History
# ---------------------------------------------------------------------------

# Whitelist of columns that the client is allowed to sort by.
# This prevents SQL injection via the sort_by parameter (you can't
# parameterise ORDER BY column names with ? placeholders).
_ALLOWED_SORT_COLUMNS = {"timestamp", "confidence", "latency_ms", "class_name", "id"}
_ALLOWED_ORDERS = {"asc", "desc"}


def get_history(
    limit: int = 50,
    offset: int = 0,
    sort_by: str = "timestamp",
    order: str = "desc",
) -> dict[str, Any]:
    """
    Retrieve paginated prediction history.

    Parameters
    ----------
    limit : int
        Maximum number of records to return (page size).
    offset : int
        Number of records to skip (for pagination).
    sort_by : str
        Column to sort by. Must be one of the whitelisted columns.
    order : str
        Sort direction — 'asc' or 'desc'.

    Returns
    -------
    dict
        {
            "predictions": [ { ... }, ... ],   # list of prediction dicts
            "total_count": int                 # total rows (for pagination UI)
        }
    """
    # --- Input validation / sanitization ---
    # Guard against SQL injection by only allowing known-good column names.
    if sort_by not in _ALLOWED_SORT_COLUMNS:
        sort_by = "timestamp"
    if order.lower() not in _ALLOWED_ORDERS:
        order = "desc"

    with get_db_connection() as conn:
        # First, get the total number of records for pagination metadata.
        total_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

        # Fetch the requested page of results.
        # Note: sort_by and order are f-string interpolated (NOT parameterised)
        # because they're validated above against a whitelist.
        # limit and offset ARE parameterised to prevent injection.
        cursor = conn.execute(
            f"SELECT * FROM predictions ORDER BY {sort_by} {order} LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = cursor.fetchall()

        # Convert sqlite3.Row objects to plain dicts for JSON serialization.
        predictions = [dict(row) for row in rows]

    return {
        "predictions": predictions,
        "total_count": total_count,
    }


# ---------------------------------------------------------------------------
# Read Operations — Analytics
# ---------------------------------------------------------------------------

def get_analytics() -> dict[str, Any]:
    """
    Compute aggregated statistics across all predictions.

    Returns a dictionary with:
      - total_predictions: total number of inference calls logged.
      - avg_confidence: mean confidence score across all predictions.
      - class_distribution: dict mapping class_name → count (how often
        each traffic sign was predicted).
      - recent_trend: list of the 10 most recent predictions (for
        sparkline / trend visualisation on the dashboard).

    This is computed on-the-fly from the database. For a high-traffic
    production system, you'd want to materialise these stats periodically
    or use a time-series database.
    """
    with get_db_connection() as conn:
        # --- Total predictions ---
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

        # --- Average confidence ---
        # COALESCE handles the edge case where the table is empty
        # (AVG of zero rows returns NULL; we default to 0.0).
        avg_conf_row = conn.execute(
            "SELECT COALESCE(AVG(confidence), 0.0) FROM predictions"
        ).fetchone()
        avg_confidence = round(avg_conf_row[0], 4)

        # --- Class distribution ---
        # GROUP BY + COUNT gives us how many times each sign was predicted.
        dist_rows = conn.execute(
            "SELECT class_name, COUNT(*) as count FROM predictions GROUP BY class_name ORDER BY count DESC"
        ).fetchall()
        class_distribution = {row["class_name"]: row["count"] for row in dist_rows}

        # --- Recent trend ---
        # Last 10 predictions for a quick activity feed / trend chart.
        trend_rows = conn.execute(
            "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 10"
        ).fetchall()
        recent_trend = [dict(row) for row in trend_rows]

    return {
        "total_predictions": total,
        "avg_confidence": avg_confidence,
        "class_distribution": class_distribution,
        "recent_trend": recent_trend,
    }
