"""Test Timerange formatting functionality."""

from src.blockether_foundation.agents.transcriber import Timerange


class TestTimerangeFormatting:
    """Test Timerange timestamp formatting."""

    def test_format_short_duration(self):
        """Test formatting for a short duration (< 1 minute)."""
        timerange = Timerange(start=62.68, end=91.66)

        assert timerange.start_formatted == "00:01:02"
        assert timerange.end_formatted == "00:01:31"
        assert timerange.duration_formatted == "00:00:28"

    def test_format_with_hours(self):
        """Test formatting with hours included."""
        timerange = Timerange(start=3661.5, end=3675.8)

        assert timerange.start_formatted == "01:01:01"
        assert timerange.end_formatted == "01:01:15"
        assert timerange.duration_formatted == "00:00:14"

    def test_format_long_duration(self):
        """Test formatting for longer duration."""
        timerange = Timerange(start=4525.0, end=5400.0)

        assert timerange.start_formatted == "01:15:25"
        assert timerange.end_formatted == "01:30:00"
        assert timerange.duration_formatted == "00:14:35"

    def test_format_zero_duration(self):
        """Test formatting with zero duration."""
        timerange = Timerange(start=0.0, end=0.0)

        assert timerange.start_formatted == "00:00:00"
        assert timerange.end_formatted == "00:00:00"
        assert timerange.duration_formatted == "00:00:00"

    def test_format_same_time_start_end(self):
        """Test formatting where start and end are the same second."""
        timerange = Timerange(start=120.0, end=120.0)

        assert timerange.start_formatted == "00:02:00"
        assert timerange.end_formatted == "00:02:00"
        assert timerange.duration_formatted == "00:00:00"
