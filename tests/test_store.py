import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import store


def _sample_records():
    return [
        {"id": i, "prompt": f"p{i}", "noun": "pears", "step1_true": 10, "final_true": 5,
         "completion": "x", "confidence": -0.1 * i, "wrong_answer": i % 2 == 0,
         "unfaithful_step": False, "is_defective": i % 2 == 0}
        for i in range(10)
    ]


def test_insert_and_count():
    with tempfile.TemporaryDirectory() as d:
        con = store.connect(str(Path(d) / "t.duckdb"))
        store.insert_traces(con, _sample_records())
        assert store.count_traces(con) == 10


def test_select_bottom_confidence_ids_orders_by_confidence_ascending():
    with tempfile.TemporaryDirectory() as d:
        con = store.connect(str(Path(d) / "t.duckdb"))
        store.insert_traces(con, _sample_records())
        # confidence = -0.1*i, so lowest (most negative) confidence is the highest id
        bottom3 = store.select_bottom_confidence_ids(con, 3)
        assert bottom3 == [9, 8, 7]


def test_defect_rate():
    with tempfile.TemporaryDirectory() as d:
        con = store.connect(str(Path(d) / "t.duckdb"))
        store.insert_traces(con, _sample_records())
        # 5 of 10 have i % 2 == 0 -> defective
        assert store.defect_rate(con) == 50.0


def test_fetch_all_roundtrips_fields():
    with tempfile.TemporaryDirectory() as d:
        con = store.connect(str(Path(d) / "t.duckdb"))
        store.insert_traces(con, _sample_records())
        rows = store.fetch_all(con)
        assert len(rows) == 10
        assert rows[0]["noun"] == "pears"
