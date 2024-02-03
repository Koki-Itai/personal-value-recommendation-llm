import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

columns = ['item_id', 'user_id', 'story', 'casting', 'direction',
           'images', 'music', 'overall', 'created', 'title', 'review']

aspects = ['story', 'casting', 'direction', 'images', 'music']

users_review_data = pd.read_csv(
    "../../movie/data/review_data_polarity_u-100_i-20.csv", names=columns)
user_ids = list(users_review_data['user_id'].unique())


def visulize_polarity_hist(exp_results: pd.DataFrame, title: str, prefix: str, is_save: bool) -> None:
    """各属性の予測頻度を可視化"""
    to_visual_exp = exp_results[[
        'story', 'casting', 'direction', 'images', 'music']].apply(pd.value_counts)
    data_dict = to_visual_exp.to_dict()

    categories = ["positive", "neutral", "negative"]
    barWidth = 0.15
    positions = np.arange(len(categories))

    plt.figure(figsize=(10, 6))

    for idx, (label, values) in enumerate(data_dict.items()):
        plt.bar(positions + idx * barWidth, list(values.values()),
                width=barWidth, label=label)

    plt.title(title, fontweight='bold')
    plt.xlabel('Reviews', fontweight='bold')
    plt.xticks(positions + barWidth, categories)
    plt.legend()
    title = title.replace(" ", "_")
    if is_save:
        plt.savefig(f"../LangChain/results/polarity_dist/{title}_{prefix}.png")
    plt.show()


def visualize_review_length(users_review_data: pd.DataFrame):
    reviews_length = users_review_data["review"].str.len().to_list()
    reviews_length = sorted(reviews_length)

    # reviews_lengthをヒストグラムで可視化, 幅は10
    plt.figure(figsize=(10, 6))
    plt.hist(reviews_length, bins=100)
    plt.xlabel('Reviews', fontweight='bold')
    plt.ylabel('Number of Reviews', fontweight='bold')
    plt.show()


def calculate_metrics(cm, positive_class_index):
    TP = cm[positive_class_index, positive_class_index]
    FP = cm[:, positive_class_index].sum() - TP
    FN = cm[positive_class_index, :].sum() - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision +
                                     recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def calculate_f1_score(cm):
    negative_metrics = calculate_metrics(cm, 0)
    neutral_metrics = calculate_metrics(cm, 1)
    positive_metrics = calculate_metrics(cm, 2)

    macro_avg_f1 = np.mean(
        [negative_metrics[2], neutral_metrics[2], positive_metrics[2]])

    TP_total = cm.diagonal().sum()
    FP_total = cm.sum(axis=0) - cm.diagonal()
    FN_total = cm.sum(axis=1) - cm.diagonal()
    micro_avg_precision = TP_total / \
        (TP_total + FP_total.sum()) if (TP_total + FP_total.sum()) > 0 else 0
    micro_avg_recall = TP_total / \
        (TP_total + FN_total.sum()) if (TP_total + FN_total.sum()) > 0 else 0
    micro_avg_f1 = 2 * (micro_avg_precision * micro_avg_recall) / (micro_avg_precision +
                                                                   micro_avg_recall) if (micro_avg_precision + micro_avg_recall) > 0 else 0

    return macro_avg_f1, micro_avg_f1


def accuracy_from_cm(cm):
    """
    Calculate accuracy from confusion matrix.
    """
    total = sum(sum(cm))
    correct = sum([cm[i][i] for i in range(len(cm))])
    accuracy = correct / total
    return accuracy


# 評価一致率
def calc_RMrates(df_user: pd.DataFrame, aspects: list, mode: str, only_PN: bool) -> dict:
    O_i_j = defaultdict(int)
    target_counts = defaultdict(int)
    RMrates = defaultdict(int)
    for _, row in df_user.iterrows():
        for aspect in aspects:
            if only_PN:
                target_counts[aspect] += 1
                if row[f"{aspect}_{mode}"] != "neutral":
                    if row[f"{aspect}_{mode}"] == row["overall"]:
                        O_i_j[aspect] += 1
            else:
                target_counts[aspect] += 1
                if row[f"{aspect}_{mode}"] == row["overall"]:
                    O_i_j[aspect] += 1

            RMrates[aspect] = O_i_j[aspect] / target_counts[aspect]

    return RMrates


def RMSE_RMrate(RMrate_dict1, RMrate_dict2):
    """評価一致率のRMSEを計算する"""
    rmse = 0
    for key in RMrate_dict1.keys():
        rmse += (RMrate_dict1[key] - RMrate_dict2[key]) ** 2
    rmse = (rmse / len(RMrate_dict1)) ** (1/2)
    rmse = round(rmse, 3)
    return rmse


# こだわり属性

def commitments_per_user(df: pd.DataFrame, aspects: list, mode: str, only_PN: bool) -> dict:
    user_ids = list(df["user_id"].unique())
    user_commitments = {}  # 最もこだわりの強い属性を格納する辞書
    for u_id in user_ids:
        if only_PN:
            rm_rate_dict = calc_RMrates(
                df[df["user_id"] == u_id], aspects, mode, only_PN=True)
        else:
            rm_rate_dict = calc_RMrates(
                df[df["user_id"] == u_id], aspects, mode, only_PN=False)
        most_commitment = list(max(rm_rate_dict.items(), key=lambda x: x[1]))
        most_commitment[0] = most_commitment[0].replace(f"_{mode}", "")
        # most_commitment -> ['casting', 0.9]
        user_commitments[u_id] = most_commitment
    return user_commitments


def commitment_accuracy(commitments_true: dict, commitments_pred: dict) -> float:
    match_count = 0
    for u_id in user_ids:
        if commitments_true[u_id][0] == commitments_pred[u_id][0]:  # こだわりのaspectが一致するか確認
            match_count += 1

    accuracy = match_count / len(user_ids)
    return accuracy


def RMSE_commmitment_score(commitments_true: dict, commitments_pred: dict) -> float:
    """こだわり属性の予測誤差"""
    rmse = 0
    for u_id in user_ids:
        if commitments_true[u_id][0] == commitments_pred[u_id][0]:
            rmse += (commitments_true[u_id][1] -
                     commitments_pred[u_id][1]) ** 2
        else:
            rmse += 1
    rmse = (rmse / len(user_ids)) ** (1/2)
    return rmse


# 極性予測誤差 Accuracy
def polarity_accuracy(df: pd.DataFrame, only_PN: bool = False) -> dict:
    accuracy_dict = defaultdict(int)
    attributes = ["story", "casting", "direction", "images", "music"]

    for attr in attributes:
        count_target = 0  # 分母
        count_match = 0  # 分子

        for _, row in df.iterrows():
            if not only_PN or row[f"{attr}_pred"] != "neutral":
                count_target += 1
                if row[f"{attr}_pred"] == row[f"{attr}_true"]:
                    count_match += 1

        accuracy_dict[attr] = round(
            count_match / count_target, 3) if count_target > 0 else 0.0

    return accuracy_dict

# Precision, Recall


def calculate_tp_fp_fn(y_true, y_pred, class_label):
    tp = sum((yt == class_label and yp == class_label)
             for yt, yp in zip(y_true, y_pred))
    fp = sum((yt != class_label and yp == class_label)
             for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == class_label and yp != class_label)
             for yt, yp in zip(y_true, y_pred))
    return tp, fp, fn


def calculate_precision_recall(y_true, y_pred, class_label):
    tp, fp, fn = calculate_tp_fp_fn(y_true, y_pred, class_label)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall


def calculate_total_tp_fp_fn(y_true, y_pred):
    total_tp = total_fp = total_fn = 0
    for class_label in set(y_true + y_pred):
        tp, fp, fn = calculate_tp_fp_fn(y_true, y_pred, class_label)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    return total_tp, total_fp, total_fn


def calculate_overall_precision_recall(y_true, y_pred):
    total_tp, total_fp, total_fn = calculate_total_tp_fp_fn(y_true, y_pred)
    overall_precision = total_tp / \
        (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / \
        (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    return overall_precision, overall_recall


def calculate_accuracy(y_true, y_pred):
    correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))
    return correct / len(y_true)
