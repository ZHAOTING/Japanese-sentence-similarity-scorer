import os
import gzip
import shutil
from urllib.request import urlretrieve

from config import Config


if __name__ == "__main__":
    if not os.path.exists(Config.data_dir):
        os.makedirs(Config.data_dir)

    if not os.path.exists(Config.embedding_zip_filepath):
        print(f'Downloading {Config.embedding_download_url} to {Config.embedding_zip_filepath}')
        urlretrieve(Config.embedding_download_url, Config.embedding_zip_filepath)
        print(f'Successfully downloaded {Config.embedding_zip_filepath}')

    if not os.path.exists(Config.embedding_filepath):
        with gzip.open(Config.embedding_zip_filepath, "rb") as f_in:
            with open(Config.embedding_filepath, "wb") as f_out:
                print(f"Extracting {Config.embedding_zip_filepath} to {Config.embedding_filepath}")
                shutil.copyfileobj(f_in, f_out)
                print(f"Extracted {Config.embedding_filepath}")

    
        
