"""Tests for audio transcription XML formatting."""

from datetime import UTC, datetime

from blockether_foundation.asr import (
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionSegmentWord,
    format_transcription_for_context,
)

EXPECTED_TOTAL_DURATION = 5.0


class TestTranscriptionXMLFormatting:
    """Test cases for transcription XML formatting."""

    def test_word_to_xml_dict(self):
        """Test Word to_xml_dict method."""
        word = TranscriptionSegmentWord(
            word="hello", start=0.0, end=0.5, score=0.95, speaker="Speaker A"
        )

        xml_dict = word.to_xml_dict()

        assert xml_dict["start"] == "0.000"
        assert xml_dict["end"] == "0.500"
        assert xml_dict["score"] == "0.950"
        assert xml_dict["speaker"] == "Speaker A"
        assert xml_dict["text"] == "hello"

    def test_word_to_xml_dict_no_speaker(self):
        """Test Word to_xml_dict method with no speaker."""
        word = TranscriptionSegmentWord(word="test", start=1.0, end=1.3, score=0.98)

        xml_dict = word.to_xml_dict()

        assert xml_dict["speaker"] == "unknown"

    def test_segment_to_xml_dict(self):
        """Test TranscriptionSegment to_xml_dict method."""
        words = [
            TranscriptionSegmentWord(word="Hello", start=0.0, end=0.5, score=0.98),
            TranscriptionSegmentWord(word="world", start=0.5, end=0.9, score=0.95),
        ]

        segment = TranscriptionSegment(
            start=0.0, end=1.0, text="Hello world", words=words, speaker="Speaker A"
        )

        xml_dict = segment.to_xml_dict()

        assert xml_dict["start"] == "0.000"
        assert xml_dict["end"] == "1.000"
        assert xml_dict["duration"] == "1.000"
        assert xml_dict["speaker"] == "Speaker A"
        assert xml_dict["text"] == "Hello world"
        assert len(xml_dict["words"]) == 2

    def test_transcription_result_to_xml_dict(self):
        """Test TranscriptionResult to_xml_dict method."""
        segments = [
            TranscriptionSegment(
                start=0.0,
                end=2.0,
                text="First segment",
                words=[TranscriptionSegmentWord(word="First", start=0.0, end=0.5, score=0.98)],
                speaker="A",
            ),
            TranscriptionSegment(
                start=2.0,
                end=4.0,
                text="Second segment",
                words=[TranscriptionSegmentWord(word="Second", start=2.0, end=2.5, score=0.95)],
                speaker="B",
            ),
        ]

        result = TranscriptionResult(
            segments=segments,
            language="en",
            language_probability=0.99,
            created_at=datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC),
        )

        xml_dict = result.to_xml_dict()

        assert xml_dict["language"] == "en"
        assert xml_dict["language_probability"] == "0.990"
        assert xml_dict["total_duration"] == "4.000"
        assert xml_dict["segment_count"] == "2"
        assert xml_dict["word_count"] == "2"
        assert xml_dict["created_at"] == "2025-01-01T12:00:00+00:00"
        assert len(xml_dict["segments"]) == 2

    def test_transcription_result_properties(self):
        """Test TranscriptionResult properties."""
        segments = [
            TranscriptionSegment(
                start=0.0,
                end=2.5,
                text="First",
                words=[TranscriptionSegmentWord(word="First", start=0.0, end=0.5, score=0.98)],
                speaker="A",
            ),
            TranscriptionSegment(
                start=3.0,
                end=5.0,
                text="Second",
                words=[TranscriptionSegmentWord(word="Second", start=3.0, end=3.5, score=0.95)],
                speaker="A",
            ),
        ]

        result = TranscriptionResult(segments=segments, language="en", language_probability=0.99)

        # Test text property
        assert result.text == "First Second"

        # Test total_duration property
        # The actual duration should be from min(start) to max(end)
        assert result.total_duration == EXPECTED_TOTAL_DURATION

        # Test word_count property
        assert result.word_count == 2

    def test_format_transcription_for_context_full(self):
        """Test format_transcription_for_context with all options."""
        segments = [
            TranscriptionSegment(
                start=0.0,
                end=1.5,
                text="Hello world",
                words=[
                    TranscriptionSegmentWord(word="Hello", start=0.0, end=0.5, score=0.98),
                    TranscriptionSegmentWord(word="world", start=0.6, end=1.0, score=0.95),
                ],
                speaker="A",
            )
        ]

        result = TranscriptionResult(
            segments=segments,
            language="en",
            language_probability=0.97,
            created_at=datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC),
        )

        xml = format_transcription_for_context(result)

        # Verify XML structure
        assert xml.startswith("<transcription>")
        assert xml.endswith("</transcription>")
        assert 'language="en"' in xml
        assert 'language_probability="0.970"' in xml
        assert 'total_duration="1.500"' in xml
        assert 'segment_count="1"' in xml
        assert 'word_count="2"' in xml
        assert 'created_at="2025-01-15T10:30:00+00:00"' in xml
        assert 'speaker="A"' in xml
        assert "<text>Hello world</text>" in xml
        assert '<word start="0.000" end="0.500" score="0.980">Hello</word>' in xml
        assert '<word start="0.600" end="1.000" score="0.950">world</word>' in xml

    def test_format_transcription_for_context_no_words(self):
        """Test format_transcription_for_context without word timestamps."""
        segments = [
            TranscriptionSegment(start=0.0, end=2.0, text="No words here", words=[], speaker="B")
        ]

        result = TranscriptionResult(segments=segments, language="es", language_probability=0.85)

        xml = format_transcription_for_context(result, include_word_timestamps=False)

        assert ">No words here</segment>" in xml
        assert "<words>" not in xml
        assert "<word " not in xml

    def test_format_transcription_for_context_truncated(self):
        """Test format_transcription_for_context with truncated content."""
        segments = [
            TranscriptionSegment(
                start=0.0,
                end=3.0,
                text="This is a very long segment that should be truncated",
                words=[],
                speaker="C",
            ),
            TranscriptionSegment(
                start=3.0, end=5.0, text="This should not appear", words=[], speaker="D"
            ),
        ]

        result = TranscriptionResult(segments=segments, language="fr", language_probability=0.90)

        xml = format_transcription_for_context(result, max_segment_content=20, max_segments=1)

        assert ">This is a very long ...</segment>" in xml
        assert "This should not appear" not in xml
        assert 'index="2"' not in xml

    def test_format_transcription_for_context_empty(self):
        """Test format_transcription_for_context with empty transcription."""
        result = TranscriptionResult(segments=[], language="unknown", language_probability=0.0)

        xml = format_transcription_for_context(result)

        assert xml.startswith("<transcription>")
        assert xml.endswith("</transcription>")
        assert 'segment_count="0"' in xml
        assert 'word_count="0"' in xml
        assert 'total_duration="0.000"' in xml

    def test_xml_escaping(self):
        """Test that special characters are properly escaped in XML."""
        segments = [
            TranscriptionSegment(
                start=0.0,
                end=2.0,
                text='Text with <special> & "quoted" characters',
                words=[],
                speaker="Speaker & Test",
            )
        ]

        result = TranscriptionResult(segments=segments, language="en", language_probability=0.95)

        xml = format_transcription_for_context(result)

        # Check that special characters are escaped
        assert "&lt;special&gt;" in xml
        assert "&amp;" in xml
        # Note: escape() escapes < > & in text content, and also quotes in attributes
        assert 'speaker="Speaker &amp; Test"' in xml
        assert '>Text with &lt;special&gt; &amp; "quoted" characters</segment>' in xml
