import json
import numpy as np
from typing import Any, Dict, Tuple, List

def load_data(data_path):
    """Loads training dataset from json file.

    :param data_path (str): Path to json file containing data
    :return X (ndarray): Inputs
    :return y (ndarray): Targets

    """
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    print("Training sets loaded!")
    return X, y

def load_data_with_mapping(data_path):
    """Loads training dataset from json file.

    :param data_path (str): Path to json file containing data
    :return X (ndarray): Inputs
    :return y (ndarray): Targets
    :return mapping (list): Class mapping

    """
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    mapping = np.array(data["mapping"])
    
    print("Training sets loaded!")
    return X, y, mapping


def load_patient_with_mapping(data_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Loads dataset from json file for visualization.

    :param data_path (str): Path to json file containing data
    :return X (ndarray): Inputs (MFCC features)
    :return y (ndarray): Targets (labels)
    :return mapping (list): Class mapping
    """
    with open(data_path, "r") as fp:
        data = json.load(fp)

    mfcc_list = []
    labels_list = []
    
    for key, value in data["patients"].items():
        mfcc_list.extend(value["mfcc"])
        labels_list.extend([value["label"]] * len(value["mfcc"]))
    
    X = np.array(mfcc_list)
    y = np.array(labels_list)
    mapping = data["mapping"]
    
    print("Dataset loaded !")
    return X, y, mapping
