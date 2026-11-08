"""Tests for the Metadata model."""

import json
from typing import Any

from pydantic import ValidationError
import pytest

from focus_feed.metadata import Metadata, StepMap, write_metadata_json


class TestSerialization:
    """Tests for serialization methods."""

    @pytest.mark.parametrize(
        "status,include_error",
        [
            ("complete", False),  # Complete MP4 with all fields
            ("in_progress", False),  # In-progress MP4
            ("error", True),  # Error metadata (no step maps)
        ],
    )
    def test_to_json_serialization(self, status, include_error):
        """to_json should correctly serialize metadata with various states."""
        kwargs: dict[str, Any] = {
            "title": "Test",
            "description": "Test desc",
            "filename": "test.mp4",
            "filepath": "test.mp4",
            "filetype": "mp4",
            "date": 1728931200,
            "size": 100,
        }
        if status == "complete":
            kwargs["transcribe"] = StepMap(
                status="complete", input_size=12345678, output_size=456789
            )
            kwargs["summarize"] = StepMap(status="complete", input_size=456789, output_size=1997)
            kwargs["content"] = "Transcript here"
        elif status == "in_progress":
            kwargs["transcribe"] = StepMap(
                status="in_progress",
                start_time=1728931200,
                input_size=12345678,
                output_size=0,
                model="large-v3-turbo",
            )
        elif status == "error":
            kwargs["error"] = "Transcription failed"

        metadata = Metadata(**kwargs)
        json_str = metadata.to_json()
        data = json.loads(json_str)

        if status == "complete":
            assert data["transcribe"]["status"] == "complete"
            assert data["summarize"]["status"] == "complete"
            assert data["content"] == "Transcript here"
        elif status == "in_progress":
            assert data["transcribe"]["status"] == "in_progress"
            assert data["transcribe"]["model"] == "large-v3-turbo"
            assert "error" not in data
        elif status == "error":
            assert data["error"] == "Transcription failed"
            assert "transcribe" not in data
            assert "summarize" not in data

    def test_to_json_excludes_none_fields(self):
        """to_json should exclude None fields when not set."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title="Test",
            description="Test",
            filename="test.pdf",
            filetype="pdf",
            date=1728931200,
            size=5000,
        )
        json_str = metadata.to_json()
        data = json.loads(json_str)
        assert "transcribe" not in data
        assert "summarize" not in data
        assert "error" not in data
        assert "content" not in data
        assert "duration" not in data
        assert "creation_time" not in data

    def test_to_dict_complete(self):
        """to_dict should produce a dict matching the schema."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title="Test",
            description="Test",
            filename="test.pdf",
            filetype="pdf",
            date=1728931200,
            size=5000,
        )
        data = metadata.to_dict()
        assert data["extract"]["status"] == "complete"
        assert data["title"] == "Test"
        assert data["filetype"] == "pdf"
        assert "error" not in data
        assert "transcribe" not in data
        assert "summarize" not in data

    @pytest.mark.parametrize(
        "filetype,expected_content",
        [
            ("pdf", "This is the extracted plaintext."),
            ("html", "Plaintext from HTML."),
        ],
    )
    def test_to_json_with_content(self, filetype, expected_content):
        """to_json should include content field for PDF/HTML with content."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000 if filetype == "pdf" else 3000,
                output_size=1234,
            ),
            title="Test",
            description="Test",
            filename=f"test.{filetype}",
            filetype=filetype,
            date=1728931200,
            size=5000,
            content=expected_content,
        )
        json_str = metadata.to_json()
        data = json.loads(json_str)
        assert data["content"] == expected_content

    def test_to_json_mp4_excludes_transcribe_and_summarize_for_pdf(self):
        """to_json should not include transcribe/summarize for PDF."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title="Test",
            description="Test",
            filename="test.pdf",
            filetype="pdf",
            date=1728931200,
            size=100,
        )
        json_str = metadata.to_json()
        data = json.loads(json_str)
        assert "transcribe" not in data
        assert "summarize" not in data

    def test_to_dict_with_generate_title(self):
        """to_dict should include generate_title when set."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            generate_title=StepMap(
                status="complete",
                start_time=1792096520,
                end_time=1792096525,
                input_size=456789,
                output_size=24,
                words=4,
                model="test-model",
                input_tokens=456,
                output_tokens=8,
            ),
            title="Test",
            description="Test",
            filename="test.pdf",
            filetype="pdf",
            date=1728931200,
            size=5000,
        )
        data = metadata.to_dict()
        assert data["generate_title"]["status"] == "complete"
        assert data["generate_title"]["model"] == "test-model"
        assert data["generate_title"]["input_tokens"] == 456
        assert data["generate_title"]["output_tokens"] == 8

    def test_to_dict_excludes_generate_title_when_none(self):
        """to_dict should exclude generate_title when it is None."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title="Test",
            description="Test",
            filename="test.pdf",
            filetype="pdf",
            date=1728931200,
            size=5000,
        )
        data = metadata.to_dict()
        assert "generate_title" not in data


class TestDeserialization:
    """Tests for deserialization methods."""

    @pytest.mark.parametrize(
        "filetype,expected_content",
        [
            ("pdf", "This is the extracted plaintext."),
            ("html", "Plaintext from HTML."),
        ],
    )
    def test_from_json_with_content(self, filetype, expected_content):
        """from_json should deserialize a PDF/HTML with content field."""
        json_str = json.dumps(
            {
                "extract": {
                    "status": "complete",
                    "input_size": 5000,
                    "output_size": 1234,
                },
                "summarize": {
                    "status": "complete",
                    "input_size": 1234,
                    "output_size": 500,
                },
                "title": "Test Doc",
                "description": "A document",
                "filename": f"doc.{filetype}",
                "filepath": f"docs/doc.{filetype}",
                "filetype": filetype,
                "date": 1728931200,
                "size": 5000,
                "content": expected_content,
            }
        )
        metadata = Metadata.from_json(json_str)
        assert metadata.content == expected_content
        assert metadata.feed_eligible is True

    def test_from_json_complete_mp4(self):
        """from_json should deserialize a valid JSON string."""
        json_str = json.dumps(
            {
                "transcribe": {
                    "status": "complete",
                    "input_size": 12345678,
                    "output_size": 456789,
                },
                "summarize": {
                    "status": "complete",
                    "input_size": 456789,
                    "output_size": 1997,
                },
                "title": "Lecture 1",
                "description": "Introduction",
                "filename": "lecture.mp4",
                "filepath": "videos/lecture.mp4",
                "filetype": "mp4",
                "date": 1728931200,
                "size": 12345678,
                "content": "Transcript here",
            }
        )
        metadata = Metadata.from_json(json_str)
        assert metadata.transcribe is not None
        assert metadata.transcribe.status == "complete"
        assert metadata.summarize is not None
        assert metadata.summarize.status == "complete"
        assert metadata.title == "Lecture 1"
        assert metadata.filetype == "mp4"
        assert metadata.content == "Transcript here"
        assert metadata.error is None
        assert metadata.feed_eligible is True

    def test_from_json_error(self):
        """from_json should deserialize error metadata."""
        json_str = json.dumps(
            {
                "title": "Failed",
                "description": "Error",
                "filename": "bad.mp4",
                "filepath": "videos/bad.mp4",
                "filetype": "mp4",
                "date": 1728931200,
                "size": 0,
                "error": "Transcription failed",
            }
        )
        metadata = Metadata.from_json(json_str)
        assert metadata.error == "Transcription failed"
        assert metadata.feed_eligible is False

    def test_from_json_in_progress_with_step_maps(self):
        """from_json should deserialize in_progress metadata with step maps."""
        json_str = json.dumps(
            {
                "transcribe": {
                    "status": "in_progress",
                    "start_time": 1728931200,
                    "input_size": 12345678,
                    "output_size": 0,
                },
                "title": "Processing",
                "description": "In progress",
                "filename": "proc.mp4",
                "filepath": "videos/proc.mp4",
                "filetype": "mp4",
                "date": 1728931200,
                "size": 100,
            }
        )
        metadata = Metadata.from_json(json_str)
        assert metadata.transcribe is not None
        assert metadata.transcribe.status == "in_progress"
        assert metadata.feed_eligible is False

    def test_from_dict_complete(self):
        """from_dict should deserialize a valid dictionary."""
        data = {
            "extract": {
                "status": "complete",
                "input_size": 5000,
                "output_size": 1234,
            },
            "summarize": {
                "status": "complete",
                "input_size": 1234,
                "output_size": 500,
            },
            "title": "Test",
            "description": "Test",
            "filename": "test.pdf",
            "filepath": "docs/test.pdf",
            "filetype": "pdf",
            "date": 1728931200,
            "size": 5000,
        }
        metadata = Metadata.from_dict(data)
        assert metadata.extract is not None
        assert metadata.extract.status == "complete"
        assert metadata.summarize is not None
        assert metadata.summarize.status == "complete"
        assert metadata.filetype == "pdf"
        assert metadata.feed_eligible is True

    def test_from_json_missing_required_field(self):
        """from_json should reject missing required fields."""
        json_str = json.dumps(
            {
                "title": "Test",
                # missing description, filename, filepath, filetype, size
            }
        )
        with pytest.raises(ValidationError):
            Metadata.from_json(json_str)


class TestExtraFields:
    """Tests for extra field rejection."""

    def test_extra_field_ignored(self):
        """Extra fields should be ignored (extra='ignore')."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title="Test",
            description="Test",
            filename="test.mp4",
            filetype="mp4",
            date=1728931200,
            size=100,
            extra_field="should be ignored",  # type: ignore[call-arg]
        )
        # Extra fields are ignored, not stored
        assert not hasattr(metadata, "extra_field")
        assert "extra_field" not in metadata.to_dict()


class TestJsonFileIO:
    """Tests for JSON file write/read methods."""

    def test_to_json_file_creates_parent_dirs(self, tmp_path):
        """to_json_file should create parent directories if needed."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title="Test",
            description="Test",
            filename="doc.pdf",
            filetype="pdf",
            date=1728931200,
            size=5000,
        )
        output_path = tmp_path / "subdir" / "doc.pdf.json"
        metadata.to_json_file(output_path)

        assert output_path.exists()

    @pytest.mark.parametrize("filetype", ["mp4", "pdf"])
    def test_roundtrip_complete_metadata(self, tmp_path, filetype):
        """Writing and reading back should produce equivalent metadata."""
        if filetype == "mp4":
            original = Metadata(
                transcribe=StepMap(
                    status="complete",
                    input_size=12345678,
                    output_size=456789,
                ),
                summarize=StepMap(
                    status="complete",
                    input_size=456789,
                    output_size=1997,
                ),
                title="Round Trip",
                description="Test round trip",
                filename="roundtrip.mp4",
                filetype="mp4",
                date=1728931200,
                mtime=1728931201,
                size=999999,
                content="Original transcript",
            )
            json_path = tmp_path / "roundtrip.mp4.json"
        else:
            original = Metadata(
                extract=StepMap(
                    status="complete",
                    input_size=5000,
                    output_size=1234,
                ),
                summarize=StepMap(
                    status="complete",
                    input_size=1234,
                    output_size=500,
                ),
                title="PDF Round Trip",
                description="Test round trip",
                filename="doc.pdf",
                filetype="pdf",
                date=1728931200,
                mtime=1728931201,
                size=5000,
                content="This is the extracted plaintext content.",
            )
            json_path = tmp_path / "doc.pdf.json"
        original.to_json_file(json_path)

        loaded = Metadata.from_json_file(json_path)
        assert loaded.title == original.title
        assert loaded.description == original.description
        assert loaded.filename == original.filename
        assert loaded.filetype == original.filetype
        assert loaded.size == original.size
        assert loaded.content == original.content
        assert loaded.error == original.error
        assert loaded.feed_eligible == original.feed_eligible

    def test_from_json_file_error_metadata(self, tmp_path):
        """from_json_file should read error metadata correctly."""
        json_content = json.dumps(
            {
                "title": "Failed",
                "description": "Error",
                "filename": "bad.mp4",
                "filepath": "bad.mp4",
                "filetype": "mp4",
                "date": 1728931200,
                "size": 0,
                "error": "Transcription failed",
            }
        )
        json_path = tmp_path / "bad.mp4.json"
        json_path.write_text(json_content, encoding="utf-8")

        metadata = Metadata.from_json_file(json_path)
        assert metadata.error == "Transcription failed"
        assert metadata.feed_eligible is False

    def test_write_metadata_json_normalizes_and_writes_atomically(self, tmp_path):
        """Canonical writer should output loadable metadata beside its source."""
        source_path = tmp_path / "subdir" / "my_lecture.MP4"
        source_path.parent.mkdir()
        source_path.write_bytes(b"video")
        data: dict[str, Any] = {
            "filepath": str(source_path),
            "transcribe": {"status": "in_progress"},
        }

        write_metadata_json(data, source_path)

        json_path = source_path.with_name(source_path.name + ".json")
        written_data = json.loads(json_path.read_text(encoding="utf-8"))
        metadata = Metadata.from_json_file(json_path)
        assert "filepath" not in written_data
        assert written_data["filename"] == source_path.name
        assert written_data["filetype"] == "mp4"
        assert written_data["title"] == {
            "filename": "my lecture",
            "extracted": None,
            "generated": None,
        }
        assert written_data["size"] == 5
        assert written_data["mtime"] == source_path.stat().st_mtime
        assert metadata.filename == source_path.name
        assert metadata.transcribe is not None
        assert metadata.transcribe.input_size == 0
        assert metadata.transcribe.output_size == 0
        assert not json_path.with_name(json_path.name + ".tmp").exists()

    def test_write_metadata_json_preserves_metadata_when_source_missing(self, tmp_path):
        """Missing source should retain existing size/mtime or default size."""
        source_path = tmp_path / "missing_file.pdf"
        data: dict[str, Any] = {"size": 12, "mtime": 34.5}

        write_metadata_json(data, source_path)

        json_path = source_path.with_name(source_path.name + ".json")
        written_data = json.loads(json_path.read_text(encoding="utf-8"))
        assert written_data["size"] == 12
        assert written_data["mtime"] == 34.5

        fallback_data: dict[str, Any] = {}
        write_metadata_json(fallback_data, tmp_path / "another_missing.pdf")
        fallback = json.loads((tmp_path / "another_missing.pdf.json").read_text(encoding="utf-8"))
        assert fallback["size"] == 0


class TestSchemaCompliance:
    """Tests to verify the model matches design doc section 3.3 schema."""

    def test_schema_has_all_required_fields(self):
        """Model should have all fields from design doc section 3.3."""
        json_schema = Metadata.model_json_schema()
        properties = json_schema["properties"]
        expected_fields = {
            "transcribe",
            "summarize",
            "extract",
            "title",
            "description",
            "filename",
            "filetype",
            "date",
            "mtime",
            "size",
            "content",
            "duration",
            "creation_time",
            "hash",
            "error",
        }
        assert expected_fields.issubset(set(properties.keys()))
        assert "filepath" not in properties

    def test_schema_does_not_have_old_fields(self):
        """Model should not have old status dict or removed fields."""
        json_schema = Metadata.model_json_schema()
        properties = json_schema["properties"]
        assert "status" not in properties
        assert "transcript" not in properties
        assert "processing_date" not in properties
        assert "transcription_remote_model" not in properties
        assert "started_at" not in properties
        assert "completed_at" not in properties

    def test_feed_eligible_is_computed(self):
        """feed_eligible should be a computed property that appears in serialization."""
        metadata = Metadata(
            transcribe=StepMap(
                status="complete",
                input_size=12345678,
                output_size=456789,
            ),
            summarize=StepMap(
                status="complete",
                input_size=456789,
                output_size=1997,
            ),
            title="Test",
            description="Test",
            filename="test.mp4",
            filetype="mp4",
            date=1728931200,
            size=100,
        )
        # The computed field should appear in model_dump
        data = metadata.model_dump()
        assert "feed_eligible" in data
        assert data["feed_eligible"] is True

    def test_date_field_is_number(self):
        """date should be a number type (UNIX epoch float), nullable."""
        json_schema = Metadata.model_json_schema()
        date_schema = json_schema["properties"]["date"]
        # date is float | None, so it uses anyOf with number and null
        assert "anyOf" in date_schema
        types = {t.get("type") for t in date_schema["anyOf"]}
        assert "number" in types
        assert "null" in types

    def test_size_is_integer(self):
        """size should be an integer type."""
        json_schema = Metadata.model_json_schema()
        assert json_schema["properties"]["size"]["type"] == "integer"

    @pytest.mark.parametrize("field_name", ["transcribe", "summarize", "extract"])
    def test_step_fields_are_optional(self, field_name):
        """Step map fields should be optional."""
        json_schema = Metadata.model_json_schema()
        assert field_name in json_schema["properties"]
        assert field_name not in json_schema.get("required", [])

    @pytest.mark.parametrize("field_name", ["description", "summary", "error"])
    def test_optional_string_fields(self, field_name):
        """Optional string fields should be nullable strings."""
        json_schema = Metadata.model_json_schema()
        assert field_name in json_schema["properties"]
        field_schema = json_schema["properties"][field_name]
        assert "anyOf" in field_schema
        types = {t.get("type") for t in field_schema["anyOf"]}
        assert "string" in types
        assert "null" in types
        assert field_name not in json_schema.get("required", [])

    def test_step_map_schema_has_required_fields(self):
        """StepMap size fields should default to zero."""
        assert StepMap.model_fields["input_size"].default == 0
        assert StepMap.model_fields["output_size"].default == 0
        step = StepMap()
        assert step.input_size == 0
        assert step.output_size == 0

    def test_publication_date_field_exists(self):
        """publication_date field should exist in the Metadata model."""
        json_schema = Metadata.model_json_schema()
        assert "publication_date" in json_schema["properties"]

    def test_publication_date_defaults_to_none(self):
        """publication_date should default to None when not provided."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title={"filename": "test", "extracted": None, "generated": None},
            description="Test",
            filename="test.pdf",
            filetype="pdf",
            date=1728931200,
            size=5000,
        )
        assert metadata.publication_date is None

    def test_publication_date_stored_when_provided(self):
        """publication_date should be stored when provided."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title={"filename": "test", "extracted": None, "generated": None},
            description="Test",
            filename="test.pdf",
            filetype="pdf",
            date=1728931200,
            publication_date=1728931200.0,
            size=5000,
        )
        assert metadata.publication_date == 1728931200.0

    def test_publication_date_serialized(self):
        """publication_date should be included in JSON serialization when set."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title={"filename": "test", "extracted": None, "generated": None},
            description="Test",
            filename="test.pdf",
            filetype="pdf",
            date=1728931200,
            publication_date=1728931200.0,
            size=5000,
        )
        data = json.loads(metadata.to_json())
        assert "publication_date" in data
        assert data["publication_date"] == 1728931200.0

    def test_publication_date_excluded_when_none(self):
        """publication_date should be excluded from JSON when None."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title={"filename": "test", "extracted": None, "generated": None},
            description="Test",
            filename="test.pdf",
            filetype="pdf",
            date=1728931200,
            size=5000,
        )
        data = json.loads(metadata.to_json())
        assert "publication_date" not in data

    def test_hash_field_exists(self):
        """hash field should exist in the Metadata model."""
        json_schema = Metadata.model_json_schema()
        assert "hash" in json_schema["properties"]

    def test_hash_defaults_to_none(self):
        """hash should default to None when not provided."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title={"filename": "test", "extracted": None, "generated": None},
            description="Test",
            filename="test.pdf",
            filetype="pdf",
            date=1728931200,
            size=5000,
        )
        assert metadata.hash is None

    def test_hash_stored_when_provided(self):
        """hash should be stored when provided."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title={"filename": "test", "extracted": None, "generated": None},
            description="Test",
            filename="test.pdf",
            filetype="pdf",
            date=1728931200,
            size=5000,
            hash="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
        )
        assert metadata.hash == "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"

    def test_hash_serialized_when_set(self):
        """hash should be included in JSON serialization when set."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title={"filename": "test", "extracted": None, "generated": None},
            description="Test",
            filename="test.pdf",
            filetype="pdf",
            date=1728931200,
            size=5000,
            hash="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
        )
        data = json.loads(metadata.to_json())
        assert "hash" in data
        assert data["hash"] == "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"

    def test_hash_excluded_when_none(self):
        """hash should be excluded from JSON serialization when None."""
        metadata = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title={"filename": "test", "extracted": None, "generated": None},
            description="Test",
            filename="test.pdf",
            filetype="pdf",
            date=1728931200,
            size=5000,
        )
        data = json.loads(metadata.to_json())
        assert "hash" not in data

    def test_hash_roundtrip(self, tmp_path):
        """hash should survive a write/read roundtrip."""
        original = Metadata(
            extract=StepMap(
                status="complete",
                input_size=5000,
                output_size=1234,
            ),
            title={"filename": "test", "extracted": None, "generated": None},
            description="Test",
            filename="test.pdf",
            filetype="pdf",
            date=1728931200,
            size=5000,
            hash="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
        )
        json_path = tmp_path / "test.pdf.json"
        original.to_json_file(json_path)

        loaded = Metadata.from_json_file(json_path)
        assert loaded.hash == original.hash

    def test_mtime_roundtrip(self, tmp_path):
        """mtime should survive a write/read roundtrip."""
        original = Metadata(
            extract=StepMap(status="complete", input_size=5000, output_size=1234),
            summarize=StepMap(status="complete", input_size=1234, output_size=500),
            title={"filename": "test", "extracted": None, "generated": None},
            description="Test",
            filename="test.pdf",
            filetype="pdf",
            date=1728931200,
            mtime=1728931234.5,
            size=5000,
        )
        json_path = tmp_path / "test.pdf.json"
        original.to_json_file(json_path)

        loaded = Metadata.from_json_file(json_path)
        assert loaded.mtime == original.mtime
