"""Tests for the Metadata model."""

import json

from focus_feed.metadata import Metadata, StepMap


class TestTitleDict:
    """Tests for the title dict structure and final_title precedence."""

    def _make_base_metadata(self, title: dict[str, str | None] | str) -> Metadata:
        """Create a Metadata instance with the given title."""
        metadata = Metadata(
            extract=StepMap(status="complete", input_size=100, output_size=50),
            summarize=StepMap(status="complete", input_size=50, output_size=20),
            title=title,
            description="Test",
            filename="test.pdf",
            filetype="pdf",
            date=1728931200,
            size=100,
        )
        metadata._source_filepath = "test.pdf"
        return metadata

    def test_title_dict_has_all_three_keys(self):
        """Title dict should have filename, extracted, and generated keys."""
        metadata = self._make_base_metadata(
            {"filename": "my file", "extracted": "Extracted Title", "generated": "Generated Title"}
        )
        title = metadata.title
        assert isinstance(title, dict)
        assert "filename" in title
        assert "extracted" in title
        assert "generated" in title

    def test_title_dict_missing_keys_are_added(self):
        """Title dict missing keys should have them added with null values."""
        metadata = self._make_base_metadata({"filename": "my file"})
        title = metadata.title
        assert isinstance(title, dict)
        assert title["filename"] == "my file"
        assert title["extracted"] is None
        assert title["generated"] is None

    def test_title_dict_none_values_preserved(self):
        """Title dict with None values should preserve them."""
        metadata = self._make_base_metadata(
            {"filename": "my file", "extracted": None, "generated": None}
        )
        title = metadata.title
        assert isinstance(title, dict)
        assert title["extracted"] is None
        assert title["generated"] is None

    def test_final_title_precedence_generated_over_extracted(self):
        """final_title should prefer generated over extracted."""
        metadata = self._make_base_metadata(
            {"filename": "my file", "extracted": "Extracted Title", "generated": "Generated Title"}
        )
        assert metadata.final_title == "Generated Title"

    def test_final_title_precedence_extracted_over_filename(self):
        """final_title should prefer extracted over filename when generated is empty."""
        metadata = self._make_base_metadata(
            {"filename": "my file", "extracted": "Extracted Title", "generated": ""}
        )
        assert metadata.final_title == "Extracted Title"

    def test_final_title_precedence_extracted_over_filename_when_generated_none(self):
        """final_title should prefer extracted over filename when generated is None."""
        metadata = self._make_base_metadata(
            {"filename": "my file", "extracted": "Extracted Title", "generated": None}
        )
        assert metadata.final_title == "Extracted Title"

    def test_final_title_falls_back_to_filename(self):
        """final_title should fall back to filename when generated and extracted are empty."""
        metadata = self._make_base_metadata(
            {"filename": "my file", "extracted": "", "generated": ""}
        )
        assert metadata.final_title == "my file"

    def test_final_title_falls_back_to_filename_when_extracted_none(self):
        """final_title should fall back to filename when extracted is None."""
        metadata = self._make_base_metadata(
            {"filename": "my file", "extracted": None, "generated": None}
        )
        assert metadata.final_title == "my file"

    def test_final_title_empty_when_all_empty(self):
        """final_title should return empty string when all values are empty."""
        metadata = self._make_base_metadata({"filename": "", "extracted": "", "generated": ""})
        assert metadata.final_title == ""

    def test_final_title_empty_when_all_none(self):
        """final_title should return empty string when all values are None."""
        metadata = self._make_base_metadata(
            {"filename": None, "extracted": None, "generated": None}
        )
        assert metadata.final_title == ""

    def test_final_title_strips_whitespace(self):
        """final_title should treat whitespace-only strings as empty."""
        metadata = self._make_base_metadata(
            {"filename": "  ", "extracted": "  ", "generated": "  "}
        )
        assert metadata.final_title == ""

    def test_final_title_string_title_backward_compat(self):
        """final_title should return string title as-is for backward compat."""
        metadata = self._make_base_metadata("Plain String Title")
        assert metadata.final_title == "Plain String Title"

    def test_final_title_string_title_empty(self):
        """final_title should return empty string for empty string title."""
        metadata = self._make_base_metadata("")
        assert metadata.final_title == ""

    def test_string_title_emits_warning(self, caplog):
        """String title should emit a deprecation warning."""
        with caplog.at_level("WARNING"):
            self._make_base_metadata("Plain String Title")
        assert any(
            "deprecated" in record.message.lower()
            for record in caplog.records
            if record.levelno >= 30
        )

    def test_string_title_warning_includes_filename(self, caplog):
        """String title deprecation warning should include the source filename."""
        with caplog.at_level("WARNING"):
            metadata = self._make_base_metadata("Plain String Title")
        string_title_warnings = [
            record
            for record in caplog.records
            if record.levelno >= 30
            and "deprecated" in record.message.lower()
            and "string titles are deprecated" in record.message.lower()
        ]
        assert string_title_warnings, "Expected a string title deprecation warning"
        assert metadata.filename in string_title_warnings[0].message

    def test_title_dict_missing_keys_emits_warning(self, caplog):
        """Title dict missing keys should emit a warning."""
        with caplog.at_level("WARNING"):
            self._make_base_metadata({"filename": "my file"})
        assert any(
            "missing keys" in record.message.lower()
            for record in caplog.records
            if record.levelno >= 30
        )

    def test_title_dict_serialization_excludes_computed(self):
        """to_json should not include final_title or feed_eligible."""
        metadata = self._make_base_metadata(
            {"filename": "my file", "extracted": "Ext", "generated": "Gen"}
        )
        data = json.loads(metadata.to_json())
        assert "final_title" not in data
        assert "feed_eligible" not in data

    def test_title_dict_serialization_includes_all_keys(self):
        """to_json should include all three title dict keys."""
        metadata = self._make_base_metadata(
            {"filename": "my file", "extracted": "Ext", "generated": "Gen"}
        )
        data = json.loads(metadata.to_json())
        assert data["title"]["filename"] == "my file"
        assert data["title"]["extracted"] == "Ext"
        assert data["title"]["generated"] == "Gen"

    def test_title_dict_serialization_excludes_none_values(self):
        """to_json should exclude None values from title dict."""
        metadata = self._make_base_metadata(
            {"filename": "my file", "extracted": None, "generated": None}
        )
        data = json.loads(metadata.to_json())
        # None values are excluded by exclude_none=True
        assert data["title"]["filename"] == "my file"
        assert "extracted" not in data["title"]
        assert "generated" not in data["title"]

    def test_final_title_precedence_full_chain(self):
        """final_title should follow generated > extracted > filename precedence."""
        # Test: generated wins over everything
        metadata = self._make_base_metadata(
            {"filename": "fname", "extracted": "ext", "generated": "gen"}
        )
        assert metadata.final_title == "gen"

        # Test: extracted wins over filename when generated is empty
        metadata = self._make_base_metadata(
            {"filename": "fname", "extracted": "ext", "generated": ""}
        )
        assert metadata.final_title == "ext"

        # Test: filename is the fallback
        metadata = self._make_base_metadata({"filename": "fname", "extracted": "", "generated": ""})
        assert metadata.final_title == "fname"

    def test_title_dict_with_extra_keys_keeps_required(self):
        """Title dict with extra keys should still have the three required keys."""
        metadata = self._make_base_metadata(
            {"filename": "my file", "extracted": "Ext", "generated": "Gen", "extra": "ignored"}
        )
        title = metadata.title
        assert isinstance(title, dict)
        assert title["filename"] == "my file"
        assert title["extracted"] == "Ext"
        assert title["generated"] == "Gen"


class TestModulePrefix:
    """Tests for module prefix application in final_title.

    The module prefix (parent directory name) is prepended to the resolved
    title when the title itself does not already start with the module name.
    This prevents duplicate prefixes when the LLM-generated or extracted
    title already includes the module name.
    """

    def _make_metadata(
        self,
        title: dict[str, str | None] | str,
        filename: str,
        filepath: str = "",
    ) -> Metadata:
        """Create a Metadata instance with the given title and filepath."""
        metadata = Metadata(
            extract=StepMap(status="complete", input_size=100, output_size=50),
            title=title,
            description="Test",
            filename=filename,
            filetype="pdf",
            date=1728931200,
            size=100,
        )
        metadata._source_filepath = filepath
        return metadata

    def test_module_prefix_applied_when_title_does_not_start_with_module(self):
        """Module prefix should be applied when title doesn't start with folder name."""
        metadata = self._make_metadata(
            {"filename": "lecture", "extracted": None, "generated": "My Lecture"},
            filename="lecture.mp4",
            filepath="videos/lecture.mp4",
        )
        assert metadata.final_title == "videos: My Lecture"

    def test_module_prefix_not_applied_when_title_starts_with_module(self):
        """Module prefix should NOT be applied when title already starts with folder name."""
        metadata = self._make_metadata(
            {"filename": "lecture", "extracted": None, "generated": "Videos My Lecture"},
            filename="lecture.mp4",
        )
        # Title "Videos My Lecture" starts with "Videos" (case-insensitive),
        # so no prefix is applied to avoid duplication.
        assert metadata.final_title == "Videos My Lecture"

    def test_module_prefix_applied_to_generated_title(self):
        """Module prefix should be applied after precedence resolution (generated)."""
        metadata = self._make_metadata(
            {"filename": "lecture", "extracted": "Extracted", "generated": "Generated"},
            filename="lecture.mp4",
            filepath="course1/lecture.mp4",
        )
        assert metadata.final_title == "course1: Generated"

    def test_module_prefix_applied_to_extracted(self):
        """Module prefix should be applied after precedence resolution (extracted)."""
        metadata = self._make_metadata(
            {"filename": "lecture", "extracted": "Extracted", "generated": ""},
            filename="lecture.mp4",
            filepath="course1/lecture.mp4",
        )
        assert metadata.final_title == "course1: Extracted"

    def test_module_prefix_applied_to_filename_title(self):
        """Module prefix should be applied after precedence resolution (filename)."""
        metadata = self._make_metadata(
            {"filename": "lecture", "extracted": "", "generated": ""},
            filename="lecture.mp4",
            filepath="course1/lecture.mp4",
        )
        assert metadata.final_title == "course1: lecture"

    def test_module_prefix_not_applied_at_root_level(self):
        """Module prefix should not be applied when file is at root level (no parent dir)."""
        metadata = self._make_metadata(
            {"filename": "lecture", "extracted": None, "generated": "My Lecture"},
            filename="lecture.mp4",
        )
        assert metadata.final_title == "My Lecture"

    def test_module_prefix_handles_underscores_in_folder_name(self):
        """Module prefix should handle underscores in folder name (replaced with spaces)."""
        metadata = self._make_metadata(
            {"filename": "lecture", "extracted": None, "generated": "My Lecture"},
            filename="lecture.mp4",
            filepath="my_course/lecture.mp4",
        )
        assert metadata.final_title == "my course: My Lecture"

    def test_module_prefix_case_insensitive_title_match(self):
        """Module prefix not applied when title starts with module name (case-insensitive).

        Title "Videos My Lecture" starts with "Videos" (case-insensitive),
        so no prefix is applied to avoid duplication.
        """
        metadata = self._make_metadata(
            {"filename": "lecture", "extracted": None, "generated": "Videos My Lecture"},
            filename="lecture.mp4",
            filepath="Videos/lecture.mp4",
        )
        # Title "Videos My Lecture" starts with "Videos" (case-insensitive),
        # so no prefix is applied to avoid duplication.
        assert metadata.final_title == "Videos My Lecture"

    def test_module_prefix_not_applied_when_title_starts_with_module_underscored(self):
        """Module prefix not applied when title starts with module name (underscores in title).

        Title "Videos_my_topic" → normalized "videos my topic" starts with
        "videos", so no prefix is applied to avoid duplication.
        """
        metadata = self._make_metadata(
            {"filename": "lecture", "extracted": None, "generated": "Videos_my_topic"},
            filename="lecture.mp4",
            filepath="videos/lecture.mp4",
        )
        # Title "Videos_my_topic" → normalized "videos my topic" starts with "videos",
        # so no prefix is applied to avoid duplication.
        assert metadata.final_title == "Videos_my_topic"

    def test_module_prefix_with_string_title(self):
        """Module prefix should be applied to string titles (backward compat)."""
        metadata = self._make_metadata(
            "Plain Title",
            filename="lecture.mp4",
            filepath="videos/lecture.mp4",
        )
        assert metadata.final_title == "videos: Plain Title"

    def test_module_prefix_not_applied_when_string_title_starts_with_module(self):
        """Module prefix should NOT be applied when string title starts with module name."""
        metadata = self._make_metadata(
            "Videos Lecture",
            filename="lecture.mp4",
        )
        assert metadata.final_title == "Videos Lecture"

    def test_extended_code_used_as_module_prefix(self):
        """An extended filename code should be used instead of the folder prefix."""
        metadata = self._make_metadata(
            {
                "filename": "H103RA Decline of Napoleon (FS)",
                "extracted": None,
                "generated": "Decline of Napoleon",
            },
            filename="H103RA_Decline_of_Napoleon_(FS).docx",
            filepath="H103/H103RA_Decline_of_Napoleon_(FS).docx",
        )
        assert metadata.final_title == "H103RA: Decline of Napoleon"

    def test_multi_letter_extended_code_used_as_module_prefix(self):
        """An extended code with multiple extra letters should be preserved."""
        metadata = self._make_metadata(
            {
                "filename": "H102DLORC The Operational Variables",
                "extracted": None,
                "generated": "PMESII-PT for Historical Analysis",
            },
            filename="H102DLORC_The Operational Variables.docx",
            filepath="H102/H102DLORC_The Operational Variables.docx",
        )
        assert metadata.final_title == "H102DLORC: PMESII-PT for Historical Analysis"

    def test_extended_code_detected_before_space_delimiter(self):
        """An extended filename code followed by a space should be recognized."""
        metadata = self._make_metadata(
            {
                "filename": "H101RA Military History Still Practicable",
                "extracted": None,
                "generated": "Military History Still Practicable",
            },
            filename="H101RA Military History Still Practicable.html",
            filepath="H101/H101RA Military History Still Practicable.html",
        )
        assert metadata.final_title == "H101RA: Military History Still Practicable"

    def test_extended_code_detected_before_hyphen_delimiter(self):
        """An extended filename code followed by a hyphen should be recognized."""
        metadata = self._make_metadata(
            {
                "filename": "H103RE French Military Problems-CPY (FS)",
                "extracted": None,
                "generated": "French Military Problems",
            },
            filename="H103RE_French_Military_Problems-CPY_(FS).docx",
            filepath="H103/H103RE_French_Military_Problems-CPY_(FS).docx",
        )
        assert metadata.final_title == "H103RE: French Military Problems"

    def test_filename_token_equal_to_module_uses_module_prefix(self):
        """A filename token equal to the folder name does not extend the prefix."""
        metadata = self._make_metadata(
            {"filename": "H103 Decline", "extracted": None, "generated": "Decline"},
            filename="H103_Decline.docx",
            filepath="H103/H103_Decline.docx",
        )
        assert metadata.final_title == "H103: Decline"

    def test_extended_code_detection_preserves_filename_case(self):
        """Extended code detection is case-insensitive but retains filename casing."""
        metadata = self._make_metadata(
            {"filename": "h103ra Decline", "extracted": None, "generated": "Decline"},
            filename="h103ra_Decline.docx",
            filepath="H103/h103ra_Decline.docx",
        )
        assert metadata.final_title == "h103ra: Decline"

    def test_extended_code_prefix_not_duplicated_in_title(self):
        """A title already starting with the extended code should remain unchanged."""
        metadata = self._make_metadata(
            {"filename": "H103RA Decline", "extracted": None, "generated": "H103RA Decline"},
            filename="H103RA_Decline.docx",
            filepath="H103/H103RA_Decline.docx",
        )
        assert metadata.final_title == "H103RA Decline"

    def test_extended_looking_filename_has_no_root_prefix(self):
        """An extended-looking root filename should not receive any prefix."""
        metadata = self._make_metadata(
            {"filename": "H103RA Decline", "extracted": None, "generated": "Decline"},
            filename="H103RA_Decline.docx",
            filepath="H103RA_Decline.docx",
        )
        assert metadata.final_title == "Decline"

    def test_extended_code_prefix_applied_to_extracted_title(self):
        """An extended code should prefix an extracted title when generated is empty."""
        metadata = self._make_metadata(
            {"filename": "H103RA Decline", "extracted": "Extracted", "generated": ""},
            filename="H103RA_Decline.docx",
            filepath="H103/H103RA_Decline.docx",
        )
        assert metadata.final_title == "H103RA: Extracted"
