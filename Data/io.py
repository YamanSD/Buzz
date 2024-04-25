from dataclasses import asdict
from json import dumps
from opendatasets import download
from os import path, rename, remove
from pandas import DataFrame, read_csv
from typing import Literal

from Config import config
from Utils import read_json


# Directory path
dir_path: str = path.dirname(path.realpath(__file__))


def load_croissant(dir_name: str) -> DataFrame:
    """

    If not present locally, the dataset is downloaded.

    Args:
        dir_name: Directory of the CroissantML metadata.json file to load.

    Returns:
        The loaded DataFrame.

    """

    # Path to the Data file
    dest_path: str = path.join(dir_path, dir_name)
    metadata_path: str = path.join(dir_path, dir_name, "metadata.json")
    meta_data: dict = read_json(metadata_path)
    encoding: str = meta_data['distribution'][1]['encodingFormat'].split('/')[-1]
    cl_path: str = path.join(dest_path, f"data.{encoding}")

    # If we have already downloaded the file, then return it.
    if path.exists(cl_path):
        return read_csv(cl_path)

    url: str = meta_data['url']
    folder_name: str = url.split('/')[-1]
    file_name: str = meta_data['distribution'][1]['name']

    # Create the kaggle.json config. Needed here due to implementation of opendatasets
    with open('./kaggle.json', 'w') as file:
        file.write(dumps(asdict(config.kaggle)))

    # Download the file
    download(url, data_dir=dest_path)

    rename(
        path.join(dest_path, folder_name, file_name),
        path.join(dest_path, f"data.{encoding}")
    )

    # Remove the downloaded folder
    remove(path.join(dest_path, folder_name))

    # Return re-read the file
    return load_croissant(dir_name)


def load_headlines() -> DataFrame:
    """
    Loads the ABC headlines data from 2003 to 2021 and returns it as a DataFrame.
    If not present locally, the dataset is downloaded.

    Returns:
        DataFrame containing the training headlines data.

    """
    return load_croissant("headlines")


def save_parquet(
        df: DataFrame,
        folder: Literal['headlines'] = '',
        file_name: str = None
) -> None:
    """
    The DataFrame is saved in the appropriate location as a parquet file.

    Args:
        df: DataFrame to save.
        folder: Folder to save the file in. From a list of predefined folders.
        file_name: File name to save the data in, no extension. Default is clean.

    """
    df.to_parquet(path.join(dir_path, folder, f"{file_name if file_name else 'clean'}.parquet"))
