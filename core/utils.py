import os

from timm.utils.misc import natural_key


def get_class_names_sorted(folder: str) -> list:
    labels = []
    for base, dirs, files in os.walk(folder):
        for directory in dirs:
            labels.append(directory)

    unique_labels = set(labels)
    sorted_labels = list(sorted(unique_labels, key=natural_key))
    return sorted_labels