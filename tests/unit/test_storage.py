"""Tests for MetadataDB and cache — no model, no Qdrant needed."""
import time
import pytest
from pathlib import Path


@pytest.fixture
def db(tmp_path):
    from deepsearch.storage.metadata_db import MetadataDB
    return MetadataDB(tmp_path / "test.db")


@pytest.fixture
def sample_file(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("Sample content for testing." * 10)
    return p


class TestMetadataDB:
    def test_upsert_and_get(self, db, sample_file):
        db.upsert_file("fid-1", sample_file, chunk_count=5)
        result = db.get_file(sample_file)
        assert result is not None
        assert result["file_id"] == "fid-1"
        assert result["chunk_count"] == 5

    def test_is_indexed_after_upsert(self, db, sample_file):
        db.upsert_file("fid-2", sample_file, chunk_count=3)
        assert db.is_indexed(sample_file) is True

    def test_is_indexed_missing_file(self, db, tmp_path):
        missing = tmp_path / "ghost.txt"
        missing.write_text("x")
        assert db.is_indexed(missing) is False

    def test_file_count(self, db, tmp_path):
        for i in range(3):
            p = tmp_path / f"file{i}.txt"
            p.write_text(f"content {i}")
            db.upsert_file(f"fid-{i}", p, chunk_count=1)
        assert db.file_count == 3

    def test_delete_file(self, db, sample_file):
        db.upsert_file("fid-del", sample_file, chunk_count=2)
        db.delete_file("fid-del")
        assert db.get_file(sample_file) is None

    def test_list_files(self, db, tmp_path):
        for i in range(2):
            p = tmp_path / f"f{i}.txt"
            p.write_text("x")
            db.upsert_file(f"id{i}", p, chunk_count=1)
        files = db.list_files()
        assert len(files) == 2

    def test_upsert_updates_existing(self, db, sample_file):
        db.upsert_file("fid-u", sample_file, chunk_count=1)
        db.upsert_file("fid-u", sample_file, chunk_count=99)
        result = db.get_file(sample_file)
        assert result["chunk_count"] == 99

    def test_feedback_add_and_get(self, db):
        db.add_feedback("What is Python?", "Python is a language.", rating=1)
        db.add_feedback("What is Java?", "Java is OOP.", rating=-1, comment="wrong")
        rows = db.get_feedback()
        assert len(rows) == 2
        assert any(r["rating"] == 1 for r in rows)
        assert any(r["comment"] == "wrong" for r in rows)

    def test_get_file_by_id(self, db, sample_file):
        db.upsert_file("fid-byid", sample_file, chunk_count=5)
        result = db.get_file_by_id("fid-byid")
        assert result is not None
        assert result["chunk_count"] == 5
