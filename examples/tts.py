import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Initialize TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# List speakers
print(tts.speakers)

# Run TTS
# ❗ XTTS supports both, but many models allow only one of the `speaker` and
# `speaker_wav` arguments

# TTS with list of amplitude values as output, clone the voice from `speaker_wav`
# wav = tts.tts(text="Hello world!", language="en")  # type: ignore

# TTS to a file, use a preset speaker
tts.tts_to_file(  # type: ignore
    text="You, I don't know what's up with your scumbag, a fucking horny bear who whores under a bridge for a soldier and sits on a Vanish bottle and drives the whore around the neighborhood in a wheelbarrow drunk, you know what I mean, you can't talk me down because you eat my cum out of my cock, you horny loser, you loser cloud, fuck your mom and fuck your old man.",
    speaker_wav=["jerzy1_2-12.wav", "jerzy2_2-12.wav"],
    language="en",
    file_path="output.wav",
    split_sentences=True,
)
