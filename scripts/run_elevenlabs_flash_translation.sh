#!/bin/bash
# Movie translation script with ElevenLabs Flash TTS
#
# This script processes video files to generate:
# - Transcriptions
# - Multi-language subtitles
# - TTS voice-over audio using ElevenLabs Flash v2.5
# - Audio ducking and mixing
# - Final video with burned subtitles
#
# Requirements:
# - BLOCKETHER_LLM_API_BASE_URL environment variable
# - BLOCKETHER_LLM_API_KEY environment variable
# - BLOCKETHER_ELEVENLABS_API_KEY environment variable (for ElevenLabs)
# - Install dependencies: uv sync --all-extras

set -e  # Exit on error

# Configuration
INPUT_DIR="poszukiwany_poszukiwana_input"
OUTPUT_DIR="poszukiwany_poszukiwana_output"
SOURCE_LANGUAGE="pl"
TARGET_LANGUAGE="en"
SUBTITLES_LANGUAGES="pl"

# TTS Configuration
TTS_BACKEND="elevenlabs"  # Use ElevenLabs Flash instead of Coqui
ELEVEN_MODEL_ID="eleven_flash_v2_5"  # Flash v2.5 multilingual (32 languages, 75ms latency)
# Alternative: "eleven_flash_v2" for English-only Flash
ENABLE_TTS=true
BURN_SUBS=true

# Command
echo "========================================="
echo "MOVIE TRANSLATION WITH ELEVENLABS FLASH"
echo "========================================="
echo ""
echo "Configuration:"
echo "  Input dir:       $INPUT_DIR"
echo "  Output dir:      $OUTPUT_DIR"
echo "  Source language:  $SOURCE_LANGUAGE"
echo "  Target language:  $TARGET_LANGUAGE"
echo "  Subtitles:        $SUBTITLES_LANGUAGES"
echo "  TTS backend:      $TTS_BACKEND"
echo "  ElevenLabs model: $ELEVEN_MODEL_ID"
echo "  Burn subtitles:   $BURN_SUBS"
echo ""
echo "========================================="
echo ""

# Check required environment variables
if [[ -z "${BLOCKETHER_ELEVENLABS_API_KEY}" ]]; then
    echo "ERROR: BLOCKETHER_ELEVENLABS_API_KEY environment variable is not set!"
    echo ""
    echo "Please set your ElevenLabs API key:"
    echo "  export BLOCKETHER_ELEVENLABS_API_KEY='your-api-key-here'"
    echo ""
    exit 1
fi

if [[ -z "${BLOCKETHER_LLM_API_BASE_URL}" ]]; then
    echo "ERROR: BLOCKETHER_LLM_API_BASE_URL environment variable is not set!"
    exit 1
fi

if [[ -z "${BLOCKETHER_LLM_API_KEY}" ]]; then
    echo "ERROR: BLOCKETHER_LLM_API_KEY environment variable is not set!"
    exit 1
fi

# Run translation pipeline
uv run python3 ./scripts/translate_movie.py \
    --input-dir "$INPUT_DIR" \
    --source-language "$SOURCE_LANGUAGE" \
    --target-language "$TARGET_LANGUAGE" \
    --output-dir "$OUTPUT_DIR" \
    --generate-subs \
    --subtitles-languages "$SUBTITLES_LANGUAGES" \
    --tts-backend "$TTS_BACKEND" \
    --eleven-model-id "$ELEVEN_MODEL_ID" \
    --enable-tts \
    --burn-subs

echo ""
echo "========================================="
echo "Pipeline completed!"
echo "========================================="
echo ""
echo "Output files:"
echo "  Transcriptions:   $OUTPUT_DIR/translations/"
echo "  Subtitles:        $OUTPUT_DIR/subtitles/"
echo "  TTS segments:    $OUTPUT_DIR/segments/"
echo "  Mixed audio:      $OUTPUT_DIR/partials/*_mixed.wav"
echo "  Final video:      $OUTPUT_DIR/final/*_final.mp4"
echo "========================================="
