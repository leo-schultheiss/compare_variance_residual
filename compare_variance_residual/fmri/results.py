import os


def get_result_path(modality, subject):
    path = os.path.join("results", modality, f"subject{subject:02}")
    os.makedirs(path, exist_ok=True)
    return path
