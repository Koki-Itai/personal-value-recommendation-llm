# environment
- OS: Mac OS 12.5
- CPU: Apple M1 Pro
- メモリ: 16GB

- versions
  - python: 3.11.6
  - pip: 23.3.2 (from /opt/homebrew/lib/python3.11/site-packages/pip)
  - scikit-surprise: 1.1.3

# surpriseの変更
ユーザ間類似度計算を評価一致率行列に対応させるためsurpriseの以下のように修正
- prediction_algorithms/algo_base.pyのAlgobaseクラスにpv_predictの追加
- prediction_algorithms/algo_base.pyのAlgobaseクラスにpv_testの追加
- prediction_algorithms/knns.pyのKNNWithMeansクラスにpv_estmateを追加

# 属性別評価を考慮した価値観モデルの構築
## LLMによる属性評価
LLMによりレビュー文から属性別評価の極性(positive, negative, neutral)を予測する．

ユーザ100人, レビュー20件の合計2000件の評価アイテム．

-> KNNの学習データとする

## 学習データ・テストデータ
- 学習データ
  - LLMにより抽出された100ユーザ×20アイテム評価データ
  - ユーザid, アイテムid, 属性別評価，総合評価
- テストデータ
  - 学習データのユーザ100人で学習データ内で未評価のアイテム評価データをサンプリング: 6088件

## main処理の流れ
- `Dataset`クラスにより，学習データとテストデータのsurpriseの入力形式データを作成．
    rating_scaleは0~5としている
- RMrate行列を作成

  `PersonalValueModel.calc_RMrates_matrix`により，(ユーザ数, 属性数)のRMrate行列を作成
- ユーザ間類似度の計算

  `PersonalValueModel.simirarity`により，(ユーザ数, ユーザ数)のユーザ間類似度行列の作成
- 通常KNNの構築

   ベースラインとして作成
- 属性評価値を考慮したKNNの構築

  `surprise.KNNWithMeans.pv_test`に`sim_matrix`を与え，作成したユーザ間類似度行列を使用してKNNによる推薦を行う