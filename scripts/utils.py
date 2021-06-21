import json
import numpy as np
import pickle

def load_data(path: str) -> dict:
    """
    Load dict from json file
    :param path: data path.
    :return: read dict from json file.
    """
    with open(path) as f:
        data = json.load(f)
    return data

def save_mapping(mapping: dict, path: str):
    """
    Save mapping dict to file.
    :param mapping: Mapping dictionary.
    :param path: File Location.
    :return:
    """
    with open(path, 'wb') as handle:
        pickle.dump(mapping, handle)

def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

def save_embedding(embedding: np.array, path: str):
    """
    Save embedding to file
    :param embedding: embedding to be saved
    :param path: location to save to
    :return:
    """
    np.save(path, embedding)


