from DataProcessor import Dataset
from Evaluation import Evaluation
from PersonalValues import PersonalValueModel

import logging
import surprise

logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger(__name__)

# Args
NEIGHBOR_K = 10
MIN_K = 3
AT_K = 5
THRESHOLD = 3.7
N_TRAIN_USERS = 500
N_TRAIN_ITEMS = 20
N_TRAIN = N_TRAIN_USERS * N_TRAIN_ITEMS
N_TEST = 7658  # 7658 or 5000
aspects = ['story', 'casting', 'direction', 'images', 'music']
simirarity_type = "pearson"  # "pearson" or "cosine"
prompt_type = "few_shot_ja"  # "zero_shot_ja" or "few_shot_ja"

# Train Files
baseline_train_file_name = f"./data/train/explicit/explicit_train_u{N_TRAIN_USERS}.csv"
explicit_train_file_name = f"./data/train/explicit/explicit_train_u{N_TRAIN_USERS}.csv"
implicit_train_file_name = f"./data/train/implicit/implicit_train_{prompt_type}_u{N_TRAIN_USERS}.csv"

# Test Files
test_file_name = f"./data/test/test_existing_users_r{N_TEST}.csv"


def main():
    """ Preprocess """
    # Preprocess Train Dataset
    baseline_train_dataset = Dataset(file_name=baseline_train_file_name)
    baseline_trainset = baseline_train_dataset.construct_data(type="train")

    explicit_train_dataset = Dataset(file_name=explicit_train_file_name)
    explicit_trainset = explicit_train_dataset.construct_data(type="train")

    implicit_train_dataset = Dataset(file_name=implicit_train_file_name)
    implicit_trainset = implicit_train_dataset.construct_data(type="train")

    logger.info("Completed Preprocess of Train Dataset\n")

    # Preprocess Test Dataset
    test_dataset = Dataset(file_name=test_file_name)
    testset = test_dataset.construct_data(type="test")
    logger.info("Completed Preprocess of Test Dataset\n")

    """ Personal Value Model """
    # Calculate RMrate Matrix
    explicit_pv_model = PersonalValueModel(train_dataset=explicit_train_dataset,
                                           aspects=aspects,
                                           is_implicit=False)
    implicit_pv_model = PersonalValueModel(train_dataset=implicit_train_dataset,
                                           aspects=aspects,
                                           is_implicit=True)
    logger.info("Completed Create RMrate Matrix\n")

    # Calculate Simirarity Matrix
    implicit_sim_matrix = implicit_pv_model.simirarity(
        rmrate_matrix=implicit_pv_model.rmrate_matrix, type=simirarity_type)
    explicit_sim_matrix = explicit_pv_model.simirarity(
        rmrate_matrix=explicit_pv_model.rmrate_matrix, type=simirarity_type)
    logger.info("Completed Calculate User Simirarity\n")

    """ Recommend Models """
    # Baseline Model: Pearson KNN
    baseline_pearson_knn = surprise.KNNWithMeans(k=NEIGHBOR_K,
                                                 min_k=MIN_K,
                                                 sim_options={'name': 'pearson',
                                                              'user_based': True},
                                                 verbose=False)
    baseline_pearson_knn.fit(baseline_trainset)
    baseline_pearson_knn_predicted = baseline_pearson_knn.test(testset)
    baseline_pearson_evaluation = Evaluation(predicted=baseline_pearson_knn_predicted,
                                             at_k=AT_K,
                                             threshold=THRESHOLD)

    # Baseline Model: Ecplicit Personal Value Model
    explicit_pv_knn = surprise.KNNWithMeans(k=NEIGHBOR_K,
                                            min_k=MIN_K,
                                            sim_options={'name': simirarity_type,
                                                         'user_based': True},
                                            verbose=False)
    explicit_pv_knn.fit(explicit_trainset)
    explicit_pv_knn_predicted = explicit_pv_knn.pv_test(testset=testset,
                                                            sim=explicit_sim_matrix)
    explicit_pv_knn_evaluation = Evaluation(predicted=explicit_pv_knn_predicted,
                                            at_k=AT_K,
                                            threshold=THRESHOLD)

    # Proposed Model: Implicit Personal Value Model
    implicit_pv_knn = surprise.KNNWithMeans(k=NEIGHBOR_K,
                                            min_k=MIN_K,
                                            sim_options={'name': simirarity_type,
                                                         'user_based': True},
                                            verbose=False)
    implicit_pv_knn.fit(implicit_trainset)
    implicit_pv_predicted = implicit_pv_knn.pv_test(testset=testset,
                                                        sim=implicit_sim_matrix)
    implicit_pv_evaluation = Evaluation(predicted=implicit_pv_predicted,
                                        at_k=AT_K,
                                        threshold=THRESHOLD)

if __name__ == "__main__":
    main()
