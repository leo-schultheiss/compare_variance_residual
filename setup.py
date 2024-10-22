import requests
import os

from tqdm import tqdm


def download_data(url):
    # extract file name from url
    file_name = os.path.basename(url)

    # Save the file in the data folder
    file_path = os.path.join('data', file_name)

    if os.path.exists(file_path):
        print(f"{file_name} already exists.")
        return

    # Ensure the data directory exists
    os.makedirs('data', exist_ok=True)

    # Streaming, so we can iterate over the response.
    response = requests.get(url, stream=True)

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
urls = [english1000_encodings]

if __name__ == "__main__":
    for url in urls:
        download_data(url)
