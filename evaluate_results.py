import argparse
import pandas as pd

from pathlib import Path
from typing import List, Tuple

from utils import BASE_PATH

# Put the "all_results.csv" in a common folder (here: ./results/all_results/) - the script calculates the mean, standard
# deviation and count, which are saved in ./results/.

# Configuration
this_parse = argparse.ArgumentParser(description="Merge the test result all_results.csv")
this_parse.add_argument(
    "--subfolder", default="all_results", type=str, help="Subfolder of ./results/ where all all_results.csv are stored"
)
this_args = this_parse.parse_args()


def list_subdirs(base_path: Path, min_depth: int = 1) -> List[Path]:
    """
    Crawl through the base path and return all folders with the given minimum depth
    :param base_path: path where crawling should start
    :param min_depth: folders of depth to the base path which should be considered for the evaluation
    :return: list of subfolder paths
    """

    # Get all subdirectories
    all_subdirs = [cur_path for cur_path in base_path.iterdir() if cur_path.is_dir()] if min_depth > 0 else []
    # Stop if nothing is left
    if min_depth < 1 and len(all_subdirs) < 1:
        return all_subdirs

    # Go one deeper
    for cur_path in base_path.iterdir():
        # We only consider directories
        if not cur_path.is_dir():
            continue

        next_subdirs = list_subdirs(base_path=cur_path, min_depth=min_depth-1)
        all_subdirs.extend(next_subdirs)

    return all_subdirs


def open_and_combine(
        base_path: Path, index_cols: List[str], file_suffix: str = ".csv"
) -> tuple:
    """
    Open all (result) files in the path and combine to a pandas DataFrame
    :param base_path: path where the files are
    :param index_cols: names of the columns used to index the files
    :param file_suffix: suffix of the files that should be opened
    :return: DataFrame containing the mean, stddev, and data count
    """

    # Get all files in the path
    all_files = [
        cur_file for cur_file in base_path.iterdir() if cur_file.suffix == file_suffix
    ]
    if len(all_files) < 1:
        raise FileNotFoundError("No suitable files found")

    # Open all files and combine
    overall_pd = pd.concat([pd.read_csv(cur_file, index_col=0) for cur_file in all_files])
    overall_pd = overall_pd.groupby(overall_pd.index)

    return (
        overall_pd.mean(),
        overall_pd.std(),
        overall_pd.count()
    )


if __name__ == '__main__':
    # Open all results
    all_results = open_and_combine(base_path=BASE_PATH / "results" / this_args.subfolder, index_cols=[])

    # Save all files
    file_names = ["mean", "std", "count"]
    for cur_file, cur_name in zip(all_results, file_names):
        cur_file.to_csv((BASE_PATH / "results" / f"{this_args.subfolder}_{cur_name}").with_suffix(".csv"))

