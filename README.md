# Whisper Transcript Generator

This repository holds a simple wrapper on top of two audio processing models:
- [Whisper](https://github.com/openai/whisper): for the main transcription work
- [Pyannote.audio](https://github.com/pyannote/pyannote-audio): for the diarization (speaker identification)

This is for example useful to generate a transcript from a Zoom recording, improving on what Zoom is able to generate.

## Requirements

- [`ffmpeg`](https://ffmpeg.org/download.html) for audio conversion to `.wav`
- Dependent packages: `pip install -r requirements.txt`

## Pre-requisites

Copy the `.env.sample` file to `.env`:
```bash
cp .env.sample .env
```

You'll need to accept `pyannote.audio`'s conditions on those two pages:
- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/speaker-diarization-3.1

Once accepted, get a [HuggingFace Access Token](https://huggingface.co/settings/tokens).  
Put this token in your `.env` for the `HUGGINGFACE_TOKEN` env var.

## Usage

- First, put an `.m4a` or `.wav` audio file in `data/`, e.g. `test-jane-john`.
- Then call `transcript.py`, with the audio file name & speaker names:

    ```bash
    python3 transcript.py test-jane-john Jane John
    ```
