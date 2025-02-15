import os
import re
import subprocess
import sys
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from dotenv import load_dotenv
import whisper
import json

load_dotenv()


def find_best_speaker(diarization, start_time, end_time):
    """
    Find the speaker with the most overlap with the given time range.
    """
    speaker_counts = {}
    for segment in diarization:
        speaker = segment["speaker"]
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + max(
            0, min(segment["end"], end_time) - max(segment["start"], start_time)
        )
    best_speaker = max(speaker_counts, key=speaker_counts.get)
    best_speaker = int(re.search(r"\d+", best_speaker).group())
    return best_speaker


input_file_base = sys.argv[1]
m4a_file = f"data/{input_file_base}.m4a"
wav_file = f"data/{input_file_base}.wav"
diarization_file = f"data/{input_file_base}_diarization.json"
transcription_file = f"data/{input_file_base}_transcription.json"
speakers = sys.argv[2:]

# Convert to wav
if not os.path.exists(wav_file):
    command = ["ffmpeg", "-i", m4a_file, wav_file]
    subprocess.run(command, check=True)

# Diarization
if not os.path.exists(diarization_file):
    auth_token = os.getenv("HUGGINGFACE_TOKEN")
    dz_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.getenv("HUGGINGFACE_TOKEN"),
    )
    diarization = None
    with ProgressHook() as hook:
        diarization = dz_pipeline(wav_file, num_speakers=2, hook=hook)
    diarization_arr = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diarization_arr.append(
            {"start": turn.start, "end": turn.end, "speaker": speaker}
        )
    with open(diarization_file, "w") as text_file:
        json.dump(diarization_arr, text_file, ensure_ascii=False, indent=4)

# Captions generation
if not os.path.exists(transcription_file):
    transcription_model = whisper.load_model("large")
    transcription = transcription_model.transcribe(
        wav_file, language="fr", verbose=False
    )
    with open(transcription_file, "w", encoding="utf-8") as text_file:
        json.dump(transcription["segments"], text_file, ensure_ascii=False, indent=4)

# Combining both the diarization and transcription
diarization = []
with open(diarization_file, "r") as file:
    diarization = json.load(file)
transcription = []
with open(transcription_file, "r") as file:
    transcription = json.load(file)
last_speaker = -1
for segment in transcription:
    start_time = segment["start"]
    end_time = segment["end"]
    text = segment["text"]
    speaker = find_best_speaker(diarization, start_time, end_time)
    if speaker != last_speaker:
        print()
        print(f"{speakers[speaker]}:")
        last_speaker = speaker
    print(f"{text}")
