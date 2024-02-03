from typing import Any, Dict, List, Tuple
import pandas as pd
import surprise

import logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s',)
logger = logging.getLogger(__name__)


class Dataset:
    def __init__(self, file_name: str) -> None:
        self.df = self.download_from_csv(file_name)
        self.format_to_surprise(self.df)
        self.compute_user_item_statistics(self.raw_ratings)

    def download_from_csv(self, file_name):
        df = pd.read_csv(file_name)
        df = df[df["user_id"] != "anonymous"]
        return df

    def format_to_surprise(self, df: pd.DataFrame):
        reader = surprise.Reader(rating_scale=(0, 5))
        self.dataset_for_surprise = surprise.Dataset.load_from_df(
            df[["user_id", "item_id", "rating"]], reader)
        # [('.B8l96iDeCvYGIKBOAjXGz4-/', 322250, 5.0, None), ... ]
        self.raw_ratings = self.dataset_for_surprise.raw_ratings

    def compute_user_item_statistics(self, train_raw_ratings: List[Tuple[Any, Any, Any, Any]]) -> None:
        self.user_list, self.user_index, self.user_evaluations_number = self._compute_statistics(
            [rating[0] for rating in train_raw_ratings])
        self.item_list, self.item_index, self.item_evaluations_number = self._compute_statistics(
            [rating[1] for rating in train_raw_ratings])

    def _compute_statistics(self, ids: List[Any]) -> Tuple[List[Any], Dict[Any, int], Dict[Any, int]]:
        unique_list, index, evaluations_number = [], {}, {}
        for id in ids:
            if id not in index:
                unique_list.append(id)
                index[id] = len(unique_list) - 1
                evaluations_number[id] = 1
            else:
                evaluations_number[id] += 1
        unique_list.sort()
        return unique_list, index, evaluations_number

    def construct_data(self, type: str = "test"):
        if type == "test":
            return self.dataset_for_surprise.construct_testset(self.raw_ratings)
        elif type == "train":
            return self.dataset_for_surprise.build_full_trainset()
        else:
            raise ValueError("type must be 'test' or 'train'")
