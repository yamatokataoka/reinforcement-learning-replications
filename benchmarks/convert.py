import argparse
import os
from enum import Enum
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class AlgorithmNames(Enum):
    VPG = "vpg"
    TRPO = "trpo"
    PPO = "ppo"
    DDPG = "ddpg"
    TD3 = "td3"


RETURN_TAGS: Dict[str, str] = {
    "vpg": "training/average_episode_return",
    "trpo": "training/average_episode_return",
    "ppo": "training/average_episode_return",
    "ddpg": "evaluation/average_episode_return",
    "td3": "evaluation/average_episode_return",
}


def main(input_dir: Path, output_dir: Path) -> None:
    environment_name: str = input_dir.name

    os.makedirs(output_dir, exist_ok=True)

    data_dfs: List[pd.DataFrame] = []
    for subdir in input_dir.iterdir():
        algorithm_name: str = subdir.name

        tfevent_paths: List[Path] = list(subdir.rglob("events.out.tfevents.*"))

        tfevents_df: pd.DataFrame = convert_tensorboards(
            tfevent_paths, RETURN_TAGS[algorithm_name]
        )

        mean_df = (
            tfevents_df.groupby("step", as_index=False)["return"]
            .mean()
            .rename(columns={"return": "mean_return"})
        )
        std_df = (
            tfevents_df.groupby("step", as_index=False)["return"]
            .std()
            .rename(columns={"return": "return_std"})
        )

        smopthed_mean_series = mean_df["mean_return"].rolling(10, min_periods=1).mean()
        smopthed_std_series = std_df["return_std"].rolling(10, min_periods=1).mean()

        data_df: pd.DataFrame = pd.DataFrame(
            {
                "step": mean_df["step"],
                "mean_return": smopthed_mean_series,
                "upper_return": smopthed_mean_series + smopthed_std_series,
                "lower_return": smopthed_mean_series - smopthed_std_series,
                "algorithm": algorithm_name,
            }
        )

        data_dfs.append(data_df)

    min_size: int = min([df["step"].iloc[-1] for df in data_dfs])

    for i in range(len(data_dfs)):
        data_dfs[i] = data_dfs[i][data_dfs[i].step <= min_size]

    concatted_data_df = pd.concat(data_dfs)

    output_path: Path = output_dir.joinpath(f"{environment_name}.csv")
    print(f"Save concatted DataFrame in {output_path}")
    with open(output_path, "w+") as f:
        concatted_data_df.to_csv(f, index=False)


def convert_tensorboards(input_paths: List[Path], tag: str) -> pd.DataFrame:
    tfevent_dfs: List[pd.DataFrame] = []
    for input_path in input_paths:
        aboslute_input_path: str = str(input_path.resolve())
        print(f"Convert {aboslute_input_path} to DataFrame")
        single_df: pd.DataFrame = convert_single_tensorboard(aboslute_input_path, tag)
        tfevent_dfs.append(single_df)

    return pd.concat(tfevent_dfs)


def convert_single_tensorboard(input_path: str, tag: str) -> pd.DataFrame:
    """
    Convert tensorboard to csv for visualiation

    From https://github.com/theRealSuperMario/supermariopy/blob/master/scripts/tflogs2pandas.py
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,
        "histograms": 1,
    }

    tfevent_df: pd.DataFrame
    try:
        event_acc = EventAccumulator(input_path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        event_scalars = event_acc.Scalars(tag)
        values = [x.value for x in event_scalars]
        steps = [x.step for x in event_scalars]
        tfevent_df = pd.DataFrame({"return": values, "step": steps})
    except Exception:
        print("Couldn't read the file: {}".format(input_path))
        raise

    return tfevent_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)

    args = parser.parse_args()

    main(Path(args.input_dir), Path(args.output_dir))
