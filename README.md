# Whisper Transcript Generator

This repository holds a simple wrapper on top of two audio processing models:
- [Whisper](https://github.com/openai/whisper): for the main transcription work
- [Pyannote.audio](https://github.com/pyannote/pyannote-audio): for the diarization (speaker identification)

This is for example useful to generate a transcript from a Zoom recording, improving on what Zoom is able to generate.

## Requirements

- Dependent packages: `pip install -r requirements.txt`
- (Optional) [`ffmpeg`](https://ffmpeg.org/download.html) if your audio files aren't `.wav`.

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

- First, put an audio file in `data/`.
- Then call `transcript.py`, with the audio file name & speaker names:

    ```bash
    ./transcript.py test-jane-john.wav Jane John
    ```

### Example output

```txt
Jane
 Hey John.
 How are you doing?

John
 Hey Jane, I'm doing really good, thanks!
 What's great with this transcript tool
 is that even if I pause in my sentence
 the whole sentence ends up properly grouped!

...
```
