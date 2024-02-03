<div align="center">
    <img height=200 src="./.github/images/icon.jpeg" alt="News Contents on Smartphone">
</div>

<h1 align="center">Personal Value based Recommendation System from Review Texts Using Large Language Models </h1>
<!-- <p align="center"><strong>Recommendation System from Review Texts Using Large Language Models</strong></p> -->

## Overview
This proposed system is a method that constructs recommendation systems requiring explicit attribute evaluations by implicitly extracting user attribute evaluations from review texts using LLMs.

The system consists of the following two modules:
- Polarity Extraction Module (PolarityExtractor)

    This module determines the polarity of the evaluation attributes mentioned in the review texts through prompting with LLMs.
    Conducted verification with GPT-3.5-turbo[1] and BERT(cl-tohoku/bert-large-japanese-v2 [2]).
- Recommendation System(Recommender)

    A recommendation system is constructed based on the attribute evaluations obtained from the Polarity Extraction Module.

    This repository aims to build a value-based information recommendation system.


## ProjectStructure
```
├── movie
│   ├── data
│   │   ├── all.csv
│   │   └── review_data_u-500_i-20.csv
├── PolarityExtractor
│   └── src
│       ├── Evaluator
│       │   └── utils
│       │       └── metrics.py
│       └── gpt-3.5-turbo-1106
│           ├── prompts
│           │   ├── __init__.py
│           │   └── PolarityExtract.py
│           └── main.py
├── Recommender
│   ├── src
│   │   ├── data
│   │   │   ├── test
│   │   │   │   └── test_existing_users_r7658.csv
│   │   │   └── train
│   │   │       ├── explicit
│   │   │       │   ├── explicit_train_u500.csv
│   │   │       ├── implicit
│   │   │           ├── BERT_predicted_u500.csv
│   │   │           ├── implicit_train_few_shot_u500.csv
│   │   │           ├── implicit_train_zero_shot_u500.csv
│   │   ├── DataProcessor
│   │   │   ├── __init__.py
│   │   │   └── dataset.py
│   │   ├── Evaluation
│   │   │   ├── __init__.py
│   │   │   └── evaluate_result.py
│   │   ├── PersonalValues
│   │   │   ├── __init__.py
│   │   │   └── personal_value_model.py
│   │   └── main.py
│   └── README.md
├── .gitignore
└── README.md
```

## Model Performance
Conducted comparative experiments on the following four methods.

1. KNN-pearson

    User-based Collaborative Filtering using KNN with Pearson Correlation Coefficient
2. KNN-cosine

    User-based Collaborative Filtering using KNN with Cosine Similarity

3. ECFPV: Explicit Collaborative Filtering employing Personal Values

   Personal-Value-based User-based Collaborative Filtering using KNN with Explicit Attribute Evaluations
4. **ICFPV: Implicit Collaborative Filtering employing Personal Values** 

   Personal-Value-based User-based Collaborative Filtering using KNN with Implicit Attribute Evaluations extracted by LLMs

### Experimental Result

|                      | **Precision@5** | **Recall@5** | **F1@5**  | **RMSE**  |
| -------------------- | --------------- | ------------ | --------- | --------- |
| KNN-pearson          | 0.273           | 0.146        | 0.191     | **1.057** |
| KNN-cosine           | 0.555           | 0.234        | 0.330     | 1.066     |
| ECFPV                | **0.618**       | **0.270**    | **0.376** | 1.058     |
| **ICFPV(Zero-shot)** | 0.603           | 0.258        | 0.361     | 1.063     |
| **ICFPV(Few-shot)**  | 0.588           | 0.251        | 0.352     | 1.071     |


## Reference
[1] gpt-3.5-tubo-1106, https://platform.openai.com/docs/models/gpt-3-5

[2] cl-tohoku/bert-large-japanese-v2, https://huggingface.co/cl-tohoku/bert-large-japanese-v2

[3] S. Hattori, and Y. Takama.: Proposal of user modeling method employing reputation analysis on user reviews based on personal values. {\it Proceeding of 27th Annual Conf. of Japanese Society for Artificial Intelligence}, Vol. 27, No. 1A3-IOS-3a-4, pp. 1-6, 2013.

[4] 三澤 遼理, 服部 俊一, 高間 康史： 価値観に基づくユーザモデルによる協調フィルタリングの拡張手法の提案. 人工知能学会全国大会論文集, Vol. 28, pp. 1-2, No. 1H4-NFC-01a-5, 2014.

