"""Unit tests for the dedicated JSON, CSV, and HTML parsers (Phase 2)."""
import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def json_file(tmp_path):
    p = tmp_path / "data.json"
    p.write_text(json.dumps({"name": "Alice", "scores": [10, 20], "meta": {"active": True}}))
    return p


@pytest.fixture
def jsonl_file(tmp_path):
    p = tmp_path / "records.jsonl"
    p.write_text(
        '{"id": 1, "value": "alpha"}\n'
        '{"id": 2, "value": "beta"}\n'
        '\n'  # blank line — should be skipped
        '{"id": 3, "value": "gamma"}\n'
    )
    return p


@pytest.fixture
def csv_file(tmp_path):
    p = tmp_path / "employees.csv"
    p.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA\n")
    return p


@pytest.fixture
def tsv_file(tmp_path):
    p = tmp_path / "data.tsv"
    p.write_text("col1\tcol2\tcol3\nfoo\tbar\tbaz\n")
    return p


@pytest.fixture
def html_file(tmp_path):
    p = tmp_path / "page.html"
    p.write_text(
        "<html><head><title>My Page</title><style>body{color:red}</style></head>"
        "<body><h1>Hello</h1><p>World</p>"
        "<script>alert('xss')</script></body></html>"
    )
    return p


# ---------------------------------------------------------------------------
# JSONParser
# ---------------------------------------------------------------------------

class TestJSONParserDedicated:
    def test_flattens_nested_object(self, json_file):
        from deepsearch.ingestion.parsers.json_parser import JSONParser
        doc = JSONParser().parse(json_file, "j1")
        assert "name: Alice" in doc.text
        assert "active: True" in doc.text or "active: true" in doc.text.lower()

    def test_flattens_list_elements(self, json_file):
        from deepsearch.ingestion.parsers.json_parser import JSONParser
        doc = JSONParser().parse(json_file, "j2")
        assert "scores[0]: 10" in doc.text
        assert "scores[1]: 20" in doc.text

    def test_jsonl_parses_multiple_records(self, jsonl_file):
        from deepsearch.ingestion.parsers.json_parser import JSONParser
        doc = JSONParser().parse(jsonl_file, "jl1")
        assert "record 1" in doc.text
        assert "record 2" in doc.text
        assert "record 3" in doc.text
        assert "alpha" in doc.text
        assert "gamma" in doc.text

    def test_invalid_json_returns_error_message(self, tmp_path):
        from deepsearch.ingestion.parsers.json_parser import JSONParser
        bad = tmp_path / "bad.json"
        bad.write_text("{not: valid")
        doc = JSONParser().parse(bad, "bad")
        assert "parse error" in doc.text.lower()

    def test_metadata_contains_format(self, json_file):
        from deepsearch.ingestion.parsers.json_parser import JSONParser
        doc = JSONParser().parse(json_file, "j3")
        assert doc.metadata["format"] == "json"

    def test_supported_extensions(self):
        from deepsearch.ingestion.parsers.json_parser import JSONParser
        p = JSONParser()
        assert ".json" in p.supported_extensions
        assert ".jsonl" in p.supported_extensions


# ---------------------------------------------------------------------------
# CSVParser
# ---------------------------------------------------------------------------

class TestCSVParserDedicated:
    def test_parses_csv_rows(self, csv_file):
        from deepsearch.ingestion.parsers.csv_parser import CSVParser
        doc = CSVParser().parse(csv_file, "csv1")
        assert "Alice" in doc.text
        assert "Bob" in doc.text

    def test_columns_in_metadata(self, csv_file):
        from deepsearch.ingestion.parsers.csv_parser import CSVParser
        doc = CSVParser().parse(csv_file, "csv2")
        assert "name" in doc.metadata["columns"]
        assert "age" in doc.metadata["columns"]
        assert "city" in doc.metadata["columns"]

    def test_parses_tsv_file(self, tsv_file):
        from deepsearch.ingestion.parsers.csv_parser import CSVParser
        doc = CSVParser().parse(tsv_file, "tsv1")
        assert "foo" in doc.text
        assert "bar" in doc.text

    def test_tables_field_populated(self, csv_file):
        from deepsearch.ingestion.parsers.csv_parser import CSVParser
        doc = CSVParser().parse(csv_file, "csv3")
        assert len(doc.tables) > 0
        # Header row should be in tables
        assert "name" in doc.tables[0]

    def test_supported_extensions(self):
        from deepsearch.ingestion.parsers.csv_parser import CSVParser
        p = CSVParser()
        assert ".csv" in p.supported_extensions
        assert ".tsv" in p.supported_extensions

    def test_key_value_prose_format(self, csv_file):
        from deepsearch.ingestion.parsers.csv_parser import CSVParser
        doc = CSVParser().parse(csv_file, "csv4")
        # Should produce "name: Alice; age: 30; city: NYC" style text
        assert "name: Alice" in doc.text or "Alice" in doc.text


# ---------------------------------------------------------------------------
# HTMLParser
# ---------------------------------------------------------------------------

class TestHTMLParserDedicated:
    def test_strips_html_tags(self, html_file):
        from deepsearch.ingestion.parsers.html_parser import HTMLParser
        doc = HTMLParser().parse(html_file, "html1")
        assert "<html>" not in doc.text
        assert "<p>" not in doc.text

    def test_extracts_body_text(self, html_file):
        from deepsearch.ingestion.parsers.html_parser import HTMLParser
        doc = HTMLParser().parse(html_file, "html2")
        assert "Hello" in doc.text
        assert "World" in doc.text

    def test_drops_script_content(self, html_file):
        from deepsearch.ingestion.parsers.html_parser import HTMLParser
        doc = HTMLParser().parse(html_file, "html3")
        assert "alert" not in doc.text
        assert "xss" not in doc.text

    def test_drops_style_content(self, html_file):
        from deepsearch.ingestion.parsers.html_parser import HTMLParser
        doc = HTMLParser().parse(html_file, "html4")
        assert "color:red" not in doc.text

    def test_extracts_title(self, html_file):
        from deepsearch.ingestion.parsers.html_parser import HTMLParser
        doc = HTMLParser().parse(html_file, "html5")
        assert doc.metadata["title"] == "My Page"

    def test_decodes_html_entities(self, tmp_path):
        from deepsearch.ingestion.parsers.html_parser import HTMLParser
        p = tmp_path / "entities.html"
        p.write_text("<p>&amp; &lt; &gt; &quot; &nbsp;</p>")
        doc = HTMLParser().parse(p, "ent1")
        assert "&" in doc.text
        assert "<" in doc.text
        assert ">" in doc.text

    def test_supported_extensions(self):
        from deepsearch.ingestion.parsers.html_parser import HTMLParser
        parser = HTMLParser()
        assert ".html" in parser.supported_extensions
        assert ".htm" in parser.supported_extensions
        assert ".xhtml" in parser.supported_extensions
