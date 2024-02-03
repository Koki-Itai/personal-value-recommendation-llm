from typing import Dict, List
from collections import defaultdict
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import minmax_scale

from DataProcessor import Dataset

import logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s',)
logger = logging.getLogger(__name__)


class PersonalValueModel:
    def __init__(self, train_dataset: Dataset, aspects: List[str], is_implicit: bool) -> None:
        self.aspects = aspects
        self.train_dataset_df = train_dataset.df
        self.user_list = train_dataset.user_list
        self.rmrate_matrix = self.calc_rmrates_matrix(
            train_dataset=train_dataset)
        # logger.debug(f"rmrate_matrix: {self.rmrate_matrix[0]}")

        if is_implicit:
            # tf-ifd行列の計算
            self.tf_idf_matrix = np.array([list(user.values())
                                           for user in self.tf_idf().values()])

            # tf-irsf行列の計算
            self.tf_irsf_matrix = np.array([list(user.values())
                                            for user in self.tf_irsf().values()])

            # tf-idf基づく評価一致率行列の計算
            self.rmrate_tf_idf_matrix = minmax_scale(
                self.rmrate_matrix * self.tf_idf_matrix, axis=1) + 1

            # tf-irsf基づく評価一致率行列の計算
            self.rmrate_tf_irsf_matrix = minmax_scale(
                self.rmrate_matrix * self.tf_irsf_matrix, axis=1) + 1

    def __calc_rmrates_per_user(self, df_user: pd.DataFrame) -> Dict:
        """ユーザごとにRMrateを計算し、属性ごとにrmrateを持つ辞書型で返還する"""
        # TODO: 明示的な属性評価の場合、評価幅が1~5なので、1,3,5に変換してから計算する
        O_i_j = defaultdict(int)
        target_counts = defaultdict(int)
        RMrates = defaultdict(int)
        for _, row in df_user.iterrows():
            for aspect in self.aspects:
                target_counts[aspect] += 1
                if row[f"{aspect}"] == row["rating"]:
                    O_i_j[aspect] += 1

                RMrates[aspect] = O_i_j[aspect] / target_counts[aspect]
        return RMrates

    def calc_rmrates_matrix(self, train_dataset: Dataset) -> np.ndarray:
        rmrate_matrix = np.zeros(
            (len(self.user_list), len(self.aspects)))  # 価値観行列の初期化
        for i, user in enumerate(self.user_list):
            df_user = self.train_dataset_df[self.train_dataset_df["user_id"] == user]
            RMrates = self.__calc_rmrates_per_user(df_user=df_user)
            rmrate_matrix[i] = np.array(list(RMrates.values()))
        return rmrate_matrix

    def simirarity(self, rmrate_matrix: np.ndarray, type: str = "pearson") -> np.ndarray:
        """属性評価を考慮したユーザ間類似度の計算"""
        if type == "pearson":
            aspects_pearon_matrix = np.zeros(
                (len(self.user_list), len(self.user_list)))  # ピアソン相関係数行列
            for i in range(len(self.user_list)):  # 活動ユーザ
                for j in range(len(self.user_list)):  # 探索対象ユーザ
                    if i == j:
                        aspects_pearon_matrix[i][j] = 1
                        continue
                    # 有効なユーザのマスク
                    valid_mask = (
                        rmrate_matrix[i] != -1) & (rmrate_matrix[j] != -1)
                    values_i = rmrate_matrix[i][valid_mask]
                    values_j = rmrate_matrix[j][valid_mask]
                    if len(values_i) == 0 or len(values_j) == 0:
                        continue

                    # ピアソン相関係数の計算
                    # 平均
                    mean_i = np.mean(values_i)
                    mean_j = np.mean(values_j)

                    # 分母
                    denominator_pearson = np.sqrt(
                        np.sum((values_i - mean_i)**2) * np.sum((values_j - mean_j)**2))

                    # 類似度の計算
                    aspects_pearon_matrix[i, j] = np.sum((values_i - mean_i) * (
                        values_j - mean_j)) / denominator_pearson if denominator_pearson != 0 else 0

            return aspects_pearon_matrix

        elif type == "cosine":
            aspects_cosine_matrix = np.zeros(
                (len(self.user_list), len(self.user_list)))
            for i in range(len(self.user_list)):
                for j in range(len(self.user_list)):
                    if i == j:
                        aspects_cosine_matrix[i][j] = 1
                        continue
                    valid_mask = (
                        rmrate_matrix[i] != -1) & (rmrate_matrix[j] != -1)
                    values_i = rmrate_matrix[i][valid_mask]
                    values_j = rmrate_matrix[j][valid_mask]
                    if len(values_i) == 0 or len(values_j) == 0:
                        continue

                    # コサイン類似度の計算
                    numerator_cosine = np.dot(values_i, values_j)
                    denominator_cosine = np.linalg.norm(
                        values_i) * np.linalg.norm(values_j)
                    aspects_cosine_matrix[i, j] = numerator_cosine / \
                        denominator_cosine if denominator_cosine != 0 else 0
            return aspects_cosine_matrix

    def tf_idf(self) -> Dict:
        tf = self.calc_tf()
        idf = self.calc_idf()
        tf_idf = {}
        user_ids = self.train_dataset_df["user_id"].unique()
        tf_idf = {}
        for user_id in user_ids:
            tf_idf_per_user = {}
            for aspect in self.aspects:
                # tf_idf_per_user[aspect] = tf[user_id][aspect] * idf[aspect]
                tf_idf_per_user[aspect] = (
                    tf[user_id][aspect] * idf[aspect]) + 1
            tf_idf[user_id] = tf_idf_per_user
        return tf_idf

    def tf_irsf(self) -> Dict:
        tf = self.calc_tf()
        irsf = self.calc_inverse_review_set_frequncy()
        tf_irsf = {}
        user_ids = self.train_dataset_df["user_id"].unique()
        for user_id in user_ids:
            tmp = []
            for aspect in self.aspects:
                tmp.append(tf[user_id][aspect] * irsf[aspect] + 1)
            af_irsf = {}
            for i, aspect in enumerate(self.aspects):
                af_irsf[aspect] = tmp[i]
            tf_irsf[user_id] = af_irsf
        return tf_irsf

    def calc_tf(self) -> Dict:
        """(あるユーザのレビュー内の属性tへの言及数 + 1) / そのユーザのレビュー内の総言及数"""
        tf = {}
        user_ids = self.train_dataset_df["user_id"].unique()
        for user_id in user_ids:
            user_df = self.train_dataset_df[self.train_dataset_df["user_id"] == user_id]
            user_mentioned = {}
            n_all_mentioned = 0
            # それぞれの属性が何回言及されたかを計算
            for aspect in self.aspects:
                n_mentioned_aspect = len(user_df[user_df[aspect] != 3]) + 1
                n_all_mentioned += n_mentioned_aspect
                user_mentioned[aspect] = n_mentioned_aspect
            # ユーザごとの総言及回数で割る
            tf_per_user = {}
            for aspect in self.aspects:
                tf_per_user[aspect] = user_mentioned[aspect] / n_all_mentioned
            tf[user_id] = tf_per_user
        # logger.debug(f"tf: {tf['.B8l96iDeCvYGIKBOAjXGz4-/']}")
        return tf

    def calc_idf(self) -> Dict:
        """log(全ユーザ数 / ある属性を評価したユーザ数)"""
        idf = defaultdict(dict)
        user_ids = self.train_dataset_df["user_id"].unique()

        # 属性へ言及しているユーザ数を計算
        n_mentioned = {}
        for aspect in self.aspects:
            n_mentioned[aspect] = len(
                self.train_dataset_df[self.train_dataset_df[aspect] != 3]["user_id"].unique())

        # IDFを計算
        for aspect in self.aspects:
            idf[aspect] = math.log(len(user_ids) / n_mentioned[aspect]) + 1
        # logger.debug(f"idf: {dict(idf)}")
        return idf

    def calc_inverse_review_set_frequncy(self) -> Dict:
        inverse_review_set_frequncy = {}
        irsf = []
        for aspect in self.aspects:
            tmp = len(
                self.train_dataset_df[self.train_dataset_df[aspect] != 3])
            tmp = math.log(len(self.train_dataset_df) / tmp) + 1
            irsf.append(tmp)
        # nomalized_irsf = preprocessing.minmax_scale(irsf) + 1

        for i, aspect in enumerate(self.aspects):
            inverse_review_set_frequncy[aspect] = irsf[i]
        # logger.debug(f"irsf: {dict(inverse_review_set_frequncy)}")
        return inverse_review_set_frequncy


# from typing import Dict, List
# import pandas as pd
# import numpy as np

# from DataProcessor import Dataset

# import logging
# logging.basicConfig(level=logging.DEBUG, format='%(message)s')
# logger = logging.getLogger(__name__)


# class PersonalValueModel:
#     def __init__(self, train_dataset: Dataset, aspects: List[str], is_implicit: bool) -> None:
#         self.aspects = aspects
#         self.train_dataset_df = train_dataset.df
#         self.user_list = train_dataset.user_list
#         self.rmrate_matrix = self.calc_rmrates_matrix()
#         logger.debug(f"{self.rmrate_matrix[0]=}")

#         if is_implicit:
#             # LLMによる評価値予測の場合のみ、言及の有無を考慮した計算を行う
#             self.tf_idf_matrix = self.calculate_tf_matrix(self.tf_idf)
#             self.tf_irsf_matrix = self.calculate_tf_matrix(self.tf_irsf)
#             logger.debug(f"{self.tf_idf_matrix[0]=}")
#             logger.debug(f"{self.tf_irsf_matrix[0]=}")

#             self.rmrate_tf_idf_matrix = self.rmrate_matrix * self.tf_idf_matrix
#             self.rmrate_tf_irsf_matrix = self.rmrate_matrix * self.tf_irsf_matrix
#             logger.debug(f"{self.rmrate_tf_idf_matrix[0]=}")
#             logger.debug(f"{self.rmrate_tf_irsf_matrix[0]=}")

#     def calc_rmrates_matrix(self) -> np.ndarray:
#         rmrate_matrix = np.zeros((len(self.user_list), len(self.aspects)))
#         for i, user in enumerate(self.user_list):
#             df_user = self.train_dataset_df[self.train_dataset_df["user_id"] == user]
#             RMrates = self.__calc_rmrates_per_user(df_user)
#             rmrate_matrix[i] = np.array(list(RMrates.values()))
#         return rmrate_matrix

#     def __calc_rmrates_per_user(self, df_user: pd.DataFrame) -> Dict:
#         O_i_j = df_user[self.aspects].eq(df_user['rating'], axis=0).sum()
#         target_counts = df_user[self.aspects].count()
#         RMrates = O_i_j / target_counts
#         return RMrates.to_dict()

#     def calculate_tf_matrix(self, method) -> np.ndarray:
#         return np.array([list(user.values()) for user in method().values()])

#     def tf_idf(self) -> Dict:
#         tf = self.calc_tf()
#         idf = self.calc_idf()
#         return {user: {aspect: tf[user][aspect] * idf[aspect] + 1 for aspect in self.aspects}
#                 for user in self.train_dataset_df["user_id"].unique()}

#     def tf_irsf(self) -> Dict:
#         tf = self.calc_tf()
#         irsf = self.calc_inverse_review_set_frequncy()
#         return {user: {aspect: tf[user][aspect] * irsf[aspect] + 1 for aspect in self.aspects}
#                 for user in self.train_dataset_df["user_id"].unique()}

#     def calc_tf(self) -> Dict:
#         user_mentioned = self.train_dataset_df[self.aspects].ne(
#             3).groupby(self.train_dataset_df['user_id']).sum() + 1
#         n_all_mentioned = user_mentioned.sum(axis=1)
#         return (user_mentioned.div(n_all_mentioned, axis=0)).to_dict(orient='index')

#     def calc_idf(self) -> Dict:
#         n_mentioned = self.train_dataset_df[self.aspects].ne(3).sum()
#         return (np.log(len(self.user_list) / n_mentioned) + 1).to_dict()

#     def calc_inverse_review_set_frequncy(self) -> Dict:
#         mentioned_count = self.train_dataset_df[self.aspects].ne(3).sum()
#         irsf = np.log(len(self.train_dataset_df) / mentioned_count) + 1
#         return irsf.to_dict()

#     def simirarity(self, rmrate_matrix: np.ndarray, type: str = "pearson") -> np.ndarray:
#         """属性評価を考慮したユーザ間類似度の計算"""
#         aspects_pearon_matrix = np.zeros(
#             (len(self.user_list), len(self.user_list)))  # ピアソン相関係数行列
#         for i in range(len(self.user_list)):  # 活動ユーザ
#             for j in range(len(self.user_list)):  # 探索対象ユーザ
#                 if i == j:
#                     aspects_pearon_matrix[i][j] = 1
#                     continue
#                 # 有効なユーザのマスク
#                 valid_mask = (
#                     rmrate_matrix[i] != -1) & (rmrate_matrix[j] != -1)
#                 values_i = rmrate_matrix[i][valid_mask]
#                 values_j = rmrate_matrix[j][valid_mask]
#                 if len(values_i) == 0 or len(values_j) == 0:
#                     continue

#                 # ピアソン相関係数の計算
#                 # 平均
#                 mean_i = np.mean(values_i)
#                 mean_j = np.mean(values_j)

#                 # 分母
#                 denominator_pearson = np.sqrt(
#                     np.sum((values_i - mean_i)**2) * np.sum((values_j - mean_j)**2))

#                 # 類似度の計算
#                 aspects_pearon_matrix[i, j] = np.sum((values_i - mean_i) * (
#                     values_j - mean_j)) / denominator_pearson if denominator_pearson != 0 else 0

#         return aspects_pearon_matrix
