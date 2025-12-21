"""Test deterministic speaker statistics calculation."""

import pytest

from blockether_foundation.agents.transcriber import (
    ConversationStatistics,
    DialogueLine,
    SpeakerStatistics,
    Timerange,
    TranscriptionResult,
)

EXPECTED_TOTAL_DURATION = 30.0
EXPECTED_SPEAKER_A_TIME = 20.0


class TestSpeakerStatistics:
    @pytest.mark.unit
    def test_calculate_conversation_statistics(self) -> None:
        conversation: list[DialogueLine] = [
            DialogueLine(
                speaker="Speaker A", text="Hello", timerange=Timerange(start=0.0, end=10.0)
            ),
            DialogueLine(speaker="Speaker B", text="Hi", timerange=Timerange(start=10.0, end=20.0)),
            DialogueLine(speaker="Speaker A", text="Ok", timerange=Timerange(start=20.0, end=30.0)),
        ]

        stats: ConversationStatistics = TranscriptionResult(
            participants=[],
            conversation=conversation,
            date=None,
        ).statistics

        assert stats.total_duration == EXPECTED_TOTAL_DURATION
        assert stats.most_active_speaker == "Speaker A"
        assert len(stats.speaker_stats) == 2

        # Sorted by total_time desc
        speaker_a: SpeakerStatistics = stats.speaker_stats[0]
        speaker_b: SpeakerStatistics = stats.speaker_stats[1]

        assert speaker_a.name == "Speaker A"
        assert speaker_a.total_time == EXPECTED_SPEAKER_A_TIME
        assert speaker_a.message_count == 2
        assert speaker_a.percentage == pytest.approx(66.666666, rel=1e-3)  # type: ignore

        assert speaker_b.name == "Speaker B"
        assert speaker_b.total_time == 10.0
        assert speaker_b.message_count == 1
        assert speaker_b.percentage == pytest.approx(33.333333, rel=1e-3)  # type: ignore
