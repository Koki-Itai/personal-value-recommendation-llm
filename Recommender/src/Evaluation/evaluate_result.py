from typing import Dict, List, Tuple
import surprise
from surprise import accuracy
from collections import defaultdict

import logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s',)
logger = logging.getLogger(__name__)


class Evaluation:
    def __init__(self, predicted: surprise.Prediction, at_k: int, threshold: float) -> None:
        self.precisions_per_user, self.recalls_per_user = self.__precision_recall_at_k(
            predictions=predicted, k=at_k, threshold=threshold)
        self.precision = sum(prec for prec in self.precisions_per_user.values(
        )) / len(self.precisions_per_user)
        self.recall = sum(rec for rec in self.recalls_per_user.values(
        )) / len(self.recalls_per_user)
        self.f1 = 2 * self.precision * self.recall / \
            (self.precision + self.recall)
        self.rmse = accuracy.rmse(predicted, verbose=False)
        self.mae = accuracy.mae(predicted, verbose=False)
        self.nDCG = self.calc_nDCG(predicted, k=at_k)

    def __get_top_n(self,
                    predictions: List[surprise.Prediction],
                    n: int = 10):
        """Return the top-N recommendation for each user from a set of predictions.

        Args:
            predictions(list of Prediction objects): The list of predictions, as
                returned by the test method of an algorithm.
            n(int): The number of recommendation to output for each user. Default
                is 10.

        Returns:
        A dict where keys are user (raw) ids and values are lists of tuples:
            [(raw item id, rating estimation), ...] of size n.
        """

        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        return top_n

    def __precision_recall_at_k(self,
                                predictions: List[surprise.Prediction],
                                k: int = 10,
                                threshold: float = 3.5) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Return precision and recall at k metrics for each user"""

        # First map the predictions to each user.
        user_est_true = defaultdict(list)
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))

        precisions = dict()
        recalls = dict()
        for uid, user_ratings in user_est_true.items():

            # Sort user ratings by estimated value
            user_ratings.sort(key=lambda x: x[0], reverse=True)

            # Number of relevant items
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

            # Number of recommended items in top k
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

            # Number of relevant and recommended items in top k
            n_rel_and_rec_k = sum(
                ((true_r >= threshold) and (est >= threshold))
                for (est, true_r) in user_ratings[:k]
            )

            # Precision@K: Proportion of recommended items that are relevant
            # When n_rec_k is 0, Precision is undefined. We here set it to 0.

            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

            # Recall@K: Proportion of relevant items that are recommended
            # When n_rel is 0, Recall is undefined. We here set it to 0.

            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

        return precisions, recalls

    def calc_nDCG(self, predictions: List[surprise.Prediction], k: int = 10) -> float:
        return 0.0
