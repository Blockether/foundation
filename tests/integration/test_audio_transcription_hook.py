"""Test the audio transcription hook functionality."""

from pathlib import Path

import pytest
from agno.agent import Agent
from agno.media import Audio
from agno.run.agent import RunInput
from agno.session import AgentSession

from blockether_foundation.agents.hooks import TranscriptionHooksConfig
from blockether_foundation.asr import LocalWhisperAudioTranscriber

# Get the path to the test resources relative to the test file
TEST_RESOURCES = Path(__file__).parent.parent / "test-resources"

# Constants for test assertions
MIN_TRANSCRIPT_LENGTH = 20


@pytest.mark.integration
async def test_audio_transcription_hook_with_sample_ogg():
    """Test that the audio transcription hook correctly transcribes sample.ogg."""

    # Verify sample.ogg exists
    audio_file = TEST_RESOURCES / "sample.ogg"
    assert audio_file.exists(), f"Test audio file not found: {audio_file}"

    # Create the audio transcription hook
    config = TranscriptionHooksConfig(transcriber=LocalWhisperAudioTranscriber(model_id="tiny"))
    hook: AudioHookType = config.pre_hook()  # type: ignore

    # Create a RunInput with the sample.ogg audio
    run_input = RunInput(
        input_content="Please transcribe this audio sample", audios=[Audio(filepath=audio_file)]
    )

    # Verify the audio file is initially present
    assert run_input.audios is not None and len(run_input.audios) == 1
    assert run_input.audios[0].filepath == audio_file

    # Execute the hook
    await hook(
        Agent(),  # Mock agent
        run_input,
        AgentSession(session_id="test_session"),  # Create a valid session
        "test_user",
        True,
    )

    # Verify the audio was transcribed and removed from audios
    assert run_input.audios is not None and len(run_input.audios) == 0, (
        "Audio files should be cleared after transcription"
    )

    # Get the input content and verify the transcript was injected
    content = run_input.input_content_string()
    assert "--- Audio Transcription |" in content, (
        "Transcript header should be injected into input content"
    )
    assert "<transcription>" in content, "Transcript should be in XML format"
    assert "language=" in content, "XML should contain language attribute"

    # Extract and verify the transcript content
    transcript_start = content.find("<transcription>")
    transcript = content[transcript_start : transcript_start + 1000]  # Limit to first 1000 chars

    # Verify the XML transcript format
    assert "<metadata" in transcript, "Should have metadata element"
    assert "<segment" in transcript, "Should have segment elements"
    # Text can be either directly in segment or in word elements
    assert "1, 2, 3, 4, 5." in content, "Should transcribe the numbers from sample.ogg"

    # Verify file metadata is included in the header and XML
    assert "sample.ogg" in content, "Should include filename in header"
    assert "--- Audio Transcription | 1 file(s):" in content, "Should show file count in header"
    assert "--- Audio Transcription | 1 file(s): sample.ogg ---" in content, (
        "Should show full header format"
    )


@pytest.mark.integration
async def test_audio_transcription_hook_no_audio():
    """Test that the audio transcription hook handles input with no audio gracefully."""

    # Create the audio transcription hook
    config = TranscriptionHooksConfig(transcriber=LocalWhisperAudioTranscriber(model_id="tiny"))
    hook: AudioHookType = config.pre_hook()  # type: ignore

    # Create a RunInput with no audio
    run_input = RunInput(input_content="This is a text-only message", audios=[])

    # Execute the hook
    await hook(
        Agent(),  # Mock agent
        run_input,
        AgentSession(session_id="test_session"),  # Create a valid session
        "test_user",
        True,
    )

    # Verify nothing changed
    assert run_input.audios is not None and len(run_input.audios) == 0
    content = run_input.input_content_string()
    assert content == "This is a text-only message", "Content should remain unchanged"


@pytest.mark.integration
async def test_audio_transcription_hook_multiple_audios():
    """Test that the audio transcription hook handles multiple audio files."""

    # Create the audio transcription hook
    config = TranscriptionHooksConfig(transcriber=LocalWhisperAudioTranscriber(model_id="tiny"))
    hook: AudioHookType = config.pre_hook()  # type: ignore

    # Use sample.ogg twice to simulate multiple audio files
    audio_file = TEST_RESOURCES / "sample.ogg"
    assert audio_file.exists(), f"Test audio file not found: {audio_file}"

    # Create a RunInput with multiple audio files
    run_input = RunInput(
        input_content="Please transcribe these audio samples",
        audios=[
            Audio(filepath=audio_file),
            Audio(filepath=audio_file),  # Same file twice for testing
        ],
    )

    # Verify initial state
    assert run_input.audios is not None and len(run_input.audios) == 2

    # Execute the hook
    await hook(
        Agent(),  # Mock agent
        run_input,
        AgentSession(session_id="test_session"),  # Create a valid session
        "test_user",
        True,
    )

    # Verify all audios were processed
    assert run_input.audios is not None and len(run_input.audios) == 0, (
        "All audio files should be cleared after transcription"
    )

    # Get the input content and verify transcripts were injected
    content = run_input.input_content_string()

    # Should have multiple transcripts in XML format
    xml_count = content.count("<transcription>")
    assert xml_count == 2, f"Should have 2 XML transcripts, got {xml_count}"

    # Each transcript should contain the numbers
    assert content.count("1, 2, 3, 4, 5.") == 2, (
        "Each transcript should contain the transcribed numbers"
    )
