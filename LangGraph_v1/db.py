"""
db.py -- read-only access to the CV pipeline's state.db.

The database is opened with SQLite's mode=ro, so nothing in this app can
change it, even by mistake. Two helper SQL functions are registered on the
connection so queries can filter on values that are stored as text:

  years(experience)        "3 years 4 months" -> 3   (months ignored, blank -> NULL)
  degree_level(education)  "MBA - IBA"        -> 4   (PhD 5, Masters 4, Bachelors 3,
                                                       Intermediate 2, Matric 1, unknown 0)

Nothing is written back -- these are computed while a query runs.
"""

import re
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "state.db"

DEGREE_NAMES = {5: "PhD", 4: "Masters", 3: "Bachelors", 2: "Intermediate", 1: "Matric", 0: "Unknown"}
DEGREE_LEVELS = {"phd": 5, "masters": 4, "bachelors": 3, "intermediate": 2, "matric": 1}

# Same ranking the pipeline uses to pick the highest degree.
_DEGREE_RANKS = [
    (5, re.compile(r"\b(ph\.?\s?d|doctorate)\b", re.I)),
    (4, re.compile(r"\b(masters?|ms|m\.s|msc|m\.sc|mphil|m\.phil|mba|ma|mcom|m\.com|mca|llm)\b", re.I)),
    (3, re.compile(r"\b(bachelors?|bs|b\.s|bsc|b\.sc|b\.e|ba|b\.a|bba|bcom|b\.com|bca|bds|mbbs|llb|b\.?tech|adp|associate degree)\b", re.I)),
    (2, re.compile(r"\b(intermediate|inter|f\.?sc|ics|i\.?com|f\.a|a[- ]levels?|hssc|dae|diploma)\b", re.I)),
    (1, re.compile(r"\b(matric\w*|ssc|o[- ]levels?|secondary school)\b", re.I)),
]


def years(experience):
    """Whole years from the pipeline's "X years Y months" text; months are
    ignored. Blank/unknown experience -> None (never treated as 0)."""
    if not experience:
        return None
    m = re.match(r"\s*(\d+)\s+years?", str(experience))
    return int(m.group(1)) if m else None


def degree_level(education):
    return next((rank for rank, rx in _DEGREE_RANKS if rx.search(education or "")), 0)


def connect():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"state.db not found at {DB_PATH}")
    conn = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.create_function("years", 1, years, deterministic=True)
    conn.create_function("degree_level", 1, degree_level, deterministic=True)
    return conn


def candidate_email_column(conn):
    """The pipeline renamed candidates.email -> candidate_email; support both
    so older and newer state.db snapshots both work."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(candidates)")}
    return "candidate_email" if "candidate_email" in cols else "email"
