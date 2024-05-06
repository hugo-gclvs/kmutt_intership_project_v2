import json
import os
from pathlib import Path
import librosa
import numpy as np
from typing import Tuple, Dict, List

DATASET_PATH = "original_dataset"
JSON_PATH = "data_25ms_551.json"
SAMPLE_RATE = 22050
SEGMENT_DURATION = 0.025  # duration of each segment in seconds
OVERLAP_DURATION = 0.01   # overlap duration in seconds
NUM_MFCC = 16
# N_FFT = 2 ** int(np.ceil(np.log2(SEGMENT_DURATION * SAMPLE_RATE)))
N_FFT = 551
HOP_LENGTH = int(np.floor(OVERLAP_DURATION * SAMPLE_RATE))

def extract_mfcc(file_path: str, num_mfcc: int, n_fft: int, hop_length: int) -> List[List[float]]:
    """Extract MFCCs from an audio file."""
    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
    track_duration = librosa.get_duration(y=signal, sr=SAMPLE_RATE)
    samples_per_track = SAMPLE_RATE * track_duration
    samples_per_segment = int(SAMPLE_RATE * SEGMENT_DURATION)
    num_segments = int(samples_per_track / samples_per_segment)

    n_fft = min(len(signal), n_fft)

    mfccs = []
    for d in range(num_segments):
        start = samples_per_segment * d
        finish = start + samples_per_segment
        if finish > len(signal):
            finish = len(signal)
        mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=SAMPLE_RATE, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfccs.append(mfcc.T.tolist())
    return mfccs


def save_mfcc(dataset_path: str, json_path: str, voc_only: bool = True, num_mfcc: int = 13, n_fft: int = 2048, hop_length: int = 512) -> None:
    """Extracts MFCCs from audio dataset and saves them into a json file along with class labels."""
    data: Dict[str, List] = {"mapping": [], "labels": [], "mfcc": [], "files": []}
    dataset_path = Path(dataset_path)

    for i, (dirpath, _, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath != dataset_path:
            semantic_label = Path(dirpath).name
            data["mapping"].append(semantic_label)
            print(f"\nProcessing: {semantic_label}")

            for f in filenames:
                if ('voc' in f) == voc_only:
                    file_path = str(Path(dirpath) / f)
                    mfccs = extract_mfcc(file_path, num_mfcc, n_fft, hop_length)
                    data["mfcc"].extend(mfccs)
                    data["labels"].extend([i - 1] * len(mfccs))
                    data["files"].extend([file_path] * len(mfccs))
                    print(f"{file_path}, segments: {len(mfccs)}")

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, voc_only=True, num_mfcc=NUM_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
