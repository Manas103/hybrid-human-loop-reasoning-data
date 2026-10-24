"""DuckDB storage for generated traces and their defect/confidence metadata.

The router's selection of which traces go to review is a plain SQL query
(order by confidence, take the bottom slice), not a separate scoring
service; DuckDB is the thing that makes that a one-line, inspectable
operation instead of a hand-rolled sort.
"""
from __future__ import annotations

import duckdb

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY,
    prompt TEXT,
    noun TEXT,
    step1_true INTEGER,
    final_true INTEGER,
    completion TEXT,
    confidence DOUBLE,
    wrong_answer BOOLEAN,
    unfaithful_step BOOLEAN,
    is_defective BOOLEAN
);
"""


def connect(path: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(path)
    con.execute(SCHEMA)
    return con


def insert_traces(con: duckdb.DuckDBPyConnection, records: list[dict]) -> None:
    con.executemany(
        "INSERT OR REPLACE INTO traces VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                r["id"], r["prompt"], r["noun"], r["step1_true"], r["final_true"],
                r["completion"], r["confidence"], r["wrong_answer"], r["unfaithful_step"],
                r["is_defective"],
            )
            for r in records
        ],
    )


def count_traces(con: duckdb.DuckDBPyConnection) -> int:
    return con.execute("SELECT COUNT(*) FROM traces").fetchone()[0]


def select_bottom_confidence_ids(con: duckdb.DuckDBPyConnection, k: int) -> list[int]:
    rows = con.execute("SELECT id FROM traces ORDER BY confidence ASC LIMIT ?", [k]).fetchall()
    return [r[0] for r in rows]


def fetch_all(con: duckdb.DuckDBPyConnection) -> list[dict]:
    cols = [d[0] for d in con.execute("SELECT * FROM traces LIMIT 0").description]
    rows = con.execute("SELECT * FROM traces ORDER BY id").fetchall()
    return [dict(zip(cols, row)) for row in rows]


def defect_rate(con: duckdb.DuckDBPyConnection) -> float:
    total, defective = con.execute(
        "SELECT COUNT(*), SUM(CASE WHEN is_defective THEN 1 ELSE 0 END) FROM traces"
    ).fetchone()
    return 100.0 * defective / total if total else 0.0
