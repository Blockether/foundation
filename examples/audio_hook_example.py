import asyncio
import logging
import os
from pathlib import Path

from agno.agent import Agent
from agno.media import Audio
from agno.models.openai import OpenAIChat

from blockether_foundation.agents.hooks.audio import AudioHooksConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

# Ensure you have BLOCKETHER_WHISPER_MODEL set or it defaults to "turbo"
# os.environ["BLOCKETHER_WHISPER_MODEL"] = "tiny"

MODEL_BASE_URL = os.getenv("BLOCKETHER_LLM_API_BASE_URL")
MODEL_API_KEY = os.getenv("BLOCKETHER_LLM_API_KEY")

# Validate required environment variables
if not MODEL_BASE_URL:
    raise ValueError("BLOCKETHER_LLM_API_BASE_URL environment variable is not set.")

print(
    f"Initializing blockether_model... model_base_url={MODEL_BASE_URL}, model_api_key={MODEL_API_KEY}"
)

blockether_model = OpenAIChat(
    id="gpt-4.1", base_url=MODEL_BASE_URL, api_key=MODEL_API_KEY, modalities=["text"]
)


async def main():
    """Run the audio transcription example."""
    # Create an agent with the transcription hook
    config = AudioHooksConfig(effort=0.5)
    agent = Agent(
        model=blockether_model,
        description="You are a helpful assistant that can understand audio.",
        pre_hooks=[config.pre_hook()],
        markdown=True,
        debug_mode=True,
    )

    # Use the OGG file from tests/test-resources
    project_root = Path(__file__).parent.parent
    audio_path = project_root / "tests" / "test-resources" / "sample.ogg"

    # Verify the audio file exists
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found at {audio_path}")

    print(f"Processing audio from: {audio_path}")

    # Run the agent with audio input
    # The hook will automatically transcribe the audio and populate audio.transcript
    response = await agent.arun(  # type: ignore
        "What do you hear in this audio? Please summarize the transcript.",
        audio=[Audio(filepath=str(audio_path))],
    )

    print("\nResponse:")
    print(response.content)


if __name__ == "__main__":
    asyncio.run(main())