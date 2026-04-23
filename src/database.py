"""
database.py - SQLite database for query history and RAGAS evaluations
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "phishingguard.db"


def init_db():
    """Initialise the SQLite database and create tables."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            question    TEXT    NOT NULL,
            intent      TEXT,
            answer      TEXT,
            contexts    TEXT,
            ontology_verified   BOOLEAN DEFAULT 0,
            confidence_score    REAL    DEFAULT 0.0,
            ontology_reasoning  TEXT,
            timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id            INTEGER REFERENCES queries(id),
            faithfulness        REAL,
            answer_relevance    REAL,
            context_relevance   REAL,
            context_recall      REAL,
            overall_score       REAL,
            timestamp           DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT NOT NULL,
            description TEXT,
            severity    TEXT DEFAULT 'medium',
            timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Seed some demo alerts
    cursor.execute("SELECT COUNT(*) FROM alerts")
    if cursor.fetchone()[0] == 0:
        alerts = [
            ("High phishing activity detected",
             "32 malicious emails quarantined", "high"),
            ("New phishing domain detected",
             "verify-account-update.com flagged", "medium"),
            ("BEC attack pattern identified",
             "Finance Department targeted", "medium"),
        ]
        cursor.executemany(
            "INSERT INTO alerts (title, description, severity) VALUES (?,?,?)",
            alerts
        )

    conn.commit()
    conn.close()


def save_query(question: str, intent: str, answer: str,
               contexts: list, ontology_verified: bool,
               confidence_score: float, ontology_reasoning: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO queries
            (question, intent, answer, contexts,
             ontology_verified, confidence_score, ontology_reasoning)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (question, intent, answer, json.dumps(contexts),
          ontology_verified, confidence_score, ontology_reasoning))
    query_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return query_id


def save_evaluation(query_id: int, faithfulness: float,
                    answer_relevance: float, context_relevance: float,
                    context_recall: float):
    overall = round(
        (faithfulness + answer_relevance + context_relevance + context_recall)
        / 4 * 100, 1
    )
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO evaluations
            (query_id, faithfulness, answer_relevance,
             context_relevance, context_recall, overall_score)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (query_id, faithfulness, answer_relevance,
          context_relevance, context_recall, overall))
    conn.commit()
    conn.close()


def get_recent_queries(limit: int = 10) -> list:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, question, timestamp
        FROM queries
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_query_by_id(query_id: int) -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT q.id, q.question, q.intent, q.answer, q.contexts,
               q.ontology_verified, q.confidence_score, q.ontology_reasoning,
               q.timestamp, e.faithfulness, e.answer_relevance,
               e.context_relevance, e.context_recall, e.overall_score
        FROM queries q
        LEFT JOIN evaluations e ON e.query_id = q.id
        WHERE q.id = ?
        ORDER BY e.timestamp DESC
        LIMIT 1
    """, (query_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    contexts = []
    if row[4]:
        try:
            contexts = json.loads(row[4])
        except json.JSONDecodeError:
            contexts = []
    return {
        "id": row[0],
        "question": row[1],
        "intent": row[2],
        "answer": row[3],
        "contexts": contexts,
        "ontology_verified": bool(row[5]),
        "confidence_score": row[6] or 0.0,
        "ontology_reasoning": row[7],
        "timestamp": row[8],
        "evaluation": {
            "faithfulness": round((row[9] or 0) * 100),
            "answer_relevance": round((row[10] or 0) * 100),
            "context_relevance": round((row[11] or 0) * 100),
            "context_recall": round((row[12] or 0) * 100),
            "overall_score": row[13] or 0,
        },
    }


def get_latest_evaluation() -> dict:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT faithfulness, answer_relevance,
               context_relevance, context_recall, overall_score
        FROM evaluations
        ORDER BY timestamp DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "faithfulness":       round(row[0] * 100),
            "answer_relevance":   round(row[1] * 100),
            "context_relevance":  round(row[2] * 100),
            "context_recall":     round(row[3] * 100),
            "overall_score":      row[4],
        }
    return {
        "faithfulness": 0, "answer_relevance": 0,
        "context_relevance": 0, "context_recall": 0,
        "overall_score": 0,
    }


def get_kb_stats() -> dict:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM queries")
    total_queries = cursor.fetchone()[0]
    conn.close()
    return {
        "documents":   12,
        "chunks":      247,
        "embeddings":  247,
        "total_queries": total_queries,
    }


def get_alerts(limit: int = 5) -> list:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT title, description, severity, timestamp
        FROM alerts
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows
