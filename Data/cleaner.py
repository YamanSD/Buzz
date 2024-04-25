from os import path
from pandas import DataFrame, to_datetime, read_parquet

from .io import load_headlines, save_parquet, dir_path


def clean_headlines() -> DataFrame:
    """

    Cleans the ABC headlines dataset and saves it.

    Returns:
        DataFrame containing the cleaned 'headlines' dataset.

    """
    df: DataFrame = load_headlines()

    # Drop missing values
    df.dropna(inplace=True)

    # Timestamp column name
    ts: str = 'publish_date'

    # Convert string timestamps to DateTime
    df[ts] = to_datetime(df[ts], format='%Y%m%d')

    # Set the timestamp column as the index
    df.set_index(ts, inplace=True)

    save_parquet(
        df,
        "headlines"
    )

    return df


def load_clean_headlines() -> DataFrame:
    """

    Loads the clean ABC headlines data from 2003 to 2021 and returns it as a DataFrame.
    If not present, the data is downloaded and cleaned.

    Returns:
        DataFrame containing the cleaned headline data.

    """
    cl_path: str = path.join(dir_path, "headlines", "clean.parquet")

    # If we have already cleaned the file, then return it.
    if path.exists(cl_path):
        return read_parquet(cl_path)

    return clean_headlines()
