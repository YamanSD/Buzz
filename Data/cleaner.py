from os import path
from pandas import DataFrame, Series, Timedelta, to_datetime, \
    read_parquet
from pandas.core.dtypes.common import is_numeric_dtype

from .io import load_headlines, save_parquet, dir_path


def clean_headlines() -> DataFrame:
    """

    Cleans the ABC headlines dataset and saves it.

    Returns:
        DataFrame containing the cleaned 'headlines' dataset.

    """
    df: DataFrame = load_headlines()
    stats: dict = {}

    # TODO implement

    # Initial data size
    init_size: int = len(df.index)

    stats['init_size'] = init_size
    stats['init_shape'] = df.shape

    # Timestamp column name
    ts: str = 'timestamp'

    # Clean the data
    stats['missing_vals'] = df.isna().sum()

    # Remove negative values
    has_neg: bool = False

    for column in df.columns:
        if not is_numeric_dtype(df[column]):
            continue

        if any(df[column] < 0):
            has_neg = True
            df = df[df[column] >= 0]

    stats['has_invalid'] = has_neg

    # Convert string timestamps to DateTime
    df[ts] = to_datetime(df[ts])

    prev_sz: int = len(df.index)

    # Remove duplicated timestamps
    df = df.drop_duplicates(ts, keep="last")

    # Number of duplicates
    dup_sz: int = prev_sz - len(df.index)

    stats['dup_timestamps'] = dup_sz

    # Sort the entries by their timestamp
    df.sort_values(ts, ascending=True, inplace=True)

    ts_diff: Series | DataFrame = df[ts].diff()
    invalid_ts: int = (ts_diff != Timedelta(minutes=1)).sum()

    stats['invalid_timestamps'] = invalid_ts
    stats['max_timestamp_dif'] = ts_diff.max()
    stats['bad_perc'] = ((invalid_ts + dup_sz) / init_size) * 100

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
