#! /usr/bin/env python3

import os
import re
import subprocess
import sys
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from dotenv import load_dotenv
import whisper
import json
import argparse

load_dotenv()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Transcribe a video file and diarize the speakers.",
        usage=f"{sys.argv[0]} -i <input_file> -l <language> -s <speaker1> -s <speaker2> ...",
    )
    parser.add_argument("-i", "--input", required=True, help="Input video file")
    parser.add_argument("-l", "--language", required=True, help="Language of the audio")
    parser.add_argument(
        "-s", "--speaker", action="append", required=True, help="Speakers in the audio."
    )
    args = parser.parse_args()
    return args.input, args.language, args.speaker


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


input_file, language, speakers = parse_arguments()
input_file_base, input_file_ext = os.path.splitext(os.path.basename(input_file))
source_file = f"data/{input_file_base}{input_file_ext}"
wav_file = f"data/{input_file_base}.wav"
diarization_file = f"data/{input_file_base}_diarization.json"
transcription_file = f"data/{input_file_base}_transcription.json"
final_transcript_file = f"data/{input_file_base}_transcript.txt"

print(f"Transcribing {source_file} in language '{language}' with speakers:")
for speaker in speakers:
    print(f"- {speaker}")
print(f"Intermediate output files:")
print(f"- {diarization_file}")
print(f"- {transcription_file}")
print(f"Final output file: {final_transcript_file}")

# Convert to wav
if not os.path.exists(wav_file):
    command = ["ffmpeg", "-i", source_file, wav_file]
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
        diarization = dz_pipeline(wav_file, num_speakers=len(speakers), hook=hook)
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
        wav_file, language=language, verbose=False
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

with open(final_transcript_file, "w", encoding="utf-8") as transcript_file:
    last_speaker = -1
    for segment in transcription:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]
        speaker = find_best_speaker(diarization, start_time, end_time)
        if speaker != last_speaker:
            transcript_file.write("\n")
            transcript_file.write(f"{speakers[speaker]}:\n")
            last_speaker = speaker
        transcript_file.write(f"{text}\n")

# Print the final transcript to stdout
with open(final_transcript_file, "r", encoding="utf-8") as transcript_file:
    print(transcript_file.read())
print(f"\nTranscription complete.\nTranscript saved to {final_transcript_file}")
