import json
import os
import math
import librosa

DATASET_PATH = "original_dataset"
JSON_PATH = "data_25ms.json"
SAMPLE_RATE = 22050
SEGMENT_DURATION = 0.025 # duration of each segment in secondes

def save_mfcc(dataset_path, json_path, voc_only=True, num_mfcc=13, n_fft=2048, hop_length=512):
    """Extracts MFCCs from audio dataset and saves them into a json file along witgh class labels.

        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param voc_only (bool): If True, only process files with 'voc' in their name. If False, process files without 'voc' in their name.
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :return:
        """

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": ["DPR", "RMT", "HRK"],
        "labels": [],
        "mfcc": []
    }

    # loop through all class sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a class sub-folder level
        if dirpath is not dataset_path:

            # save class label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in class sub-dir
            for f in filenames:

                # check if 'voc' is in the file name
                if voc_only and 'voc' in f or not voc_only and 'voc' not in f:

                    # load audio file
                    file_path = os.path.join(dirpath, f)
                    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                    # get track duration and calculate samples per track
                    track_duration = librosa.get_duration(y=signal, sr=SAMPLE_RATE)
                    samples_per_track = SAMPLE_RATE * track_duration
                    samples_per_segment = int(SAMPLE_RATE * SEGMENT_DURATION)
                    num_segments = int(samples_per_track / samples_per_segment)
                    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

                    # process all segments of audio file
                    for d in range(num_segments):

                        # calculate start and finish sample for current segment
                        start = samples_per_segment * d
                        finish = start + samples_per_segment

                        # extract mfcc
                        mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                        mfcc = mfcc.T

                        # store only mfcc feature with expected number of vectors
                        if len(mfcc) == num_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i-1)
                            print("{}, segment:{}".format(file_path, d+1))

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
        
if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, voc_only=True)
