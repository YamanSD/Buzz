from .cleaner import *
from .io import save_parquet, dir_path


def get_data(refresh: bool = False) -> DataFrame:
    """

    Args:
        refresh: If True, re-cleans the datasets.

    Returns:
        The merged DataFrame of all other DataFrames.
        Suitable for use in training.

    """
    cl_path: str = path.join(dir_path, "clean.parquet")

    # If we have already cleaned the file, then return it.
    if not refresh and path.exists(cl_path):
        return read_parquet(cl_path)

    df: DataFrame = load_clean_headlines() if not refresh else clean_headlines()

    # Remove invalid rows
    df.dropna(inplace=True)

    # Save the cleaned parquet
    save_parquet(df)

    return df
