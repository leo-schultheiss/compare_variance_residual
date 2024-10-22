import requests
import os
import logging

from tqdm import tqdm
from urllib.request import urlopen

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(file_url, download_path="data"):
    # extract file name from url
    file_name = os.path.basename(file_url)

    # Save the file in the data folder
    file_path = os.path.join(download_path, file_name)

    # check if an existing file has correct size
    if os.path.exists(file_path):
        response = urlopen(file_url)
        total_size = int(response.info().get('Content-Length').strip())
        if os.path.getsize(file_path) == total_size:
            logger.info(f"{file_name} already exists.")
            return
        else:
            logger.info(f"{file_name} is corrupted. Downloading again.")

    # Ensure the data directory exists
    os.makedirs(download_path, exist_ok=True)

    download_data(file_path, file_url)


def download_data(file_path, file_url):
    # Streaming, so we can iterate over the response.
    response = requests.get(file_url, stream=True)
    # Sizes in bytes.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(file_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Could not download file")


# English1000 is a 985-dimensional word
# embedding feature space based on word co-occurrence in English text
english1000_encodings = "https://github.com/HuthLab/deep-fMRI-dataset/raw/refs/heads/master/em_data/english1000sm.hf5"

# choose subjects to download fmri data for
# fMRI data for the six subjects in the experiment
# in arrays of (time x voxels) for each data collection run
# (10 stories of training data, 1 story repeated 2 times as validation data).
subjects = [
    "01",
    # "02",
    # "03",
    # "04",
    # "05",
    # "06",
    # "07",
    # "08",
    # "09"
]
modalities = [
    "reading",
    "listening"
]

if __name__ == "__main__":
    urls = [english1000_encodings]
    for subject in subjects:
        for modality in modalities:
            for data_type in ["trn", "val"]:
                url = "https://gin.g-node.org/denizenslab/narratives_reading_listening_fmri/raw/master/responses/subject{}_{}_fmri_data_{}.hdf".format(
                    subject, modality, data_type)
                urls.append(url)

    for url in urls:
        download_file(url)
