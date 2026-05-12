"""Tests for document parsers — no model required, uses tmp files."""
import json
import pytest
from pathlib import Path


@pytest.fixture
def txt_file(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text("Hello, world.\nThis is a test document.\nLine three.")
    return p


@pytest.fixture
def md_file(tmp_path):
    p = tmp_path / "sample.md"
    p.write_text("# Title\n\nSome paragraph.\n\n## Section\n\nMore text here.")
    return p


@pytest.fixture
def json_file(tmp_path):
    p = tmp_path / "data.json"
    p.write_text(json.dumps({"key": "value", "items": [1, 2, 3]}))
    return p


@pytest.fixture
def csv_file(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA\n")
    return p


@pytest.fixture
def html_file(tmp_path):
    p = tmp_path / "page.html"
    p.write_text("<html><body><h1>Title</h1><p>Hello from HTML.</p></body></html>")
    return p


class TestTextParser:
    def test_parses_txt(self, txt_file):
        from deepsearch.ingestion.parsers.text_parser import TextParser
        parser = TextParser()
        doc = parser.parse(txt_file, file_id="fid-1")
        assert doc.file_id == "fid-1"
        assert "Hello, world." in doc.text
        assert doc.metadata["extension"] == ".txt"

    def test_parses_md(self, md_file):
        from deepsearch.ingestion.parsers.text_parser import TextParser
        parser = TextParser()
        doc = parser.parse(md_file, file_id="fid-2")
        assert "Title" in doc.text
        assert doc.metadata["file_name"] == "sample.md"

    def test_can_parse_extensions(self):
        from deepsearch.ingestion.parsers.text_parser import TextParser
        parser = TextParser()
        assert parser.can_parse(Path("x.txt"))
        assert parser.can_parse(Path("x.py"))
        assert not parser.can_parse(Path("x.pdf"))


class TestJSONParser:
    def test_parses_json(self, json_file):
        from deepsearch.ingestion.parsers.text_parser import JSONParser
        parser = JSONParser()
        doc = parser.parse(json_file, file_id="j1")
        assert '"key"' in doc.text
        assert "value" in doc.text

    def test_pretty_printed(self, json_file):
        from deepsearch.ingestion.parsers.text_parser import JSONParser
        parser = JSONParser()
        doc = parser.parse(json_file, file_id="j2")
        # Pretty-printing adds newlines
        assert "\n" in doc.text

    def test_invalid_json_falls_back(self, tmp_path):
        from deepsearch.ingestion.parsers.text_parser import JSONParser
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        parser = JSONParser()
        doc = parser.parse(bad, file_id="bad")
        assert "{not valid json" in doc.text


class TestCSVParser:
    def test_parses_csv(self, csv_file):
        from deepsearch.ingestion.parsers.text_parser import CSVParser
        parser = CSVParser()
        doc = parser.parse(csv_file, file_id="csv1")
        assert "Alice" in doc.text
        assert "Bob" in doc.text
        assert doc.metadata["row_count"] == 2

    def test_columns_in_metadata(self, csv_file):
        from deepsearch.ingestion.parsers.text_parser import CSVParser
        parser = CSVParser()
        doc = parser.parse(csv_file, file_id="csv2")
        assert "name" in doc.metadata["columns"]
        assert "age" in doc.metadata["columns"]


class TestHTMLParser:
    def test_strips_tags(self, html_file):
        from deepsearch.ingestion.parsers.text_parser import HTMLParser
        parser = HTMLParser()
        doc = parser.parse(html_file, file_id="html1")
        assert "<html>" not in doc.text
        assert "Hello from HTML." in doc.text
        assert "Title" in doc.text


class TestParserRegistry:
    def test_dispatcher_finds_correct_parser(self, txt_file, tmp_path):
        """IngestionDispatcher selects the right parser for each extension."""
        from unittest.mock import MagicMock
        from deepsearch.ingestion.dispatcher import IngestionDispatcher

        vs = MagicMock()
        vs.upsert_batch = MagicMock()
        db = MagicMock()
        db.is_indexed = MagicMock(return_value=False)
        db.upsert_file = MagicMock()
        db.get_file = MagicMock(return_value=None)

        embedder = MagicMock()
        embedder.embed_chunks = MagicMock(return_value=[
            {"chunk_id": "c1", "dense_vector": [0.1]*384,
             "sparse_indices": [], "sparse_values": [], "payload": {}}
        ])

        dispatcher = IngestionDispatcher(
            vector_store=vs,
            metadata_db=db,
            embedder=embedder,
        )
        fid = dispatcher.ingest(txt_file)
        assert isinstance(fid, str)
        assert len(fid) > 0
        vs.upsert_batch.assert_called_once()
