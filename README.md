# お菓子画像分類プロジェクト（Snack Image Classification）

ResNet50 を用いた転移学習により、お菓子の画像を **3 クラス（candy / chocolate / rice_snack）** に分類する深層学習モデルです。

---

## 各メンバーの分担

| メンバー | 担当 |
| --- | --- |
| （氏名） | データ収集・前処理 |
| （氏名） | モデル設計・学習 |
| （氏名） | 評価・資料作成 |

---

## プロジェクトの目的

市販のお菓子を撮影した画像から、その種類（candy / chocolate / rice_snack）を自動で判別するモデルを構築する。

---

## 問題設定

- **タスク種別**: 画像の多クラス分類（single-label / 3 クラス）
- **入力**: お菓子 1 個（または 1 種）が写った RGB 画像
- **出力**: `candy`, `chocolate`, `rice_snack` のいずれか 1 クラスと、その確信度（softmax 確率）
- **クラス**: ディレクトリ名がそのままラベルになる構成（`data_processed/<class_name>/`）

---

## 学習用データの説明

### 収集方法
- 生画像を `data/<class_name>/` に配置し、前処理スクリプト `scripts/convert_images.py` で `data_processed/` に変換。
- `convert_images.py` は `.jpg / .jpeg / .png / .webp / .avif / .jfif / .gif` を受け付け、すべて **RGB の JPEG（quality=95）** に統一して保存します。
  - 多様な拡張子に対応していることから、複数の入手元（Web 上の画像など）から収集したと推測されます。**【収集元の詳細は要記入】**

### 個数
| クラス | 枚数 |
| --- | --- |
| candy | 100 |
| chocolate | 98 |
| rice_snack | 155 |
| **合計** | **約 353** |

- クラス間に枚数の偏り（rice_snack が多い）があり、軽度のクラス不均衡があります。

### 特徴量
- モデルへの入力は生の画像ピクセル（**224 × 224 × 3**）。
- ResNet50 用の前処理 `tf.keras.applications.resnet50.preprocess_input` を通してから畳み込み特徴を抽出します（手作業での特徴量設計は行わず、CNN が特徴を学習）。

### ラベル
- candy, chocolate, rice_snackの3つ

### データ分割
`tf.keras.utils.image_dataset_from_directory`（`seed=42`, `validation_split=0.2`）で分割します。

- **訓練 (train)**: 全体の 80%
- **検証+テスト (val+test)**: 全体の 20% を読み込み、`.cache()` で順序を固定したうえで前半を **検証 (val)**、後半を **テスト (test)** に二分割
  - `val_test_ds.cache()` によりシャッフル順を確定させ、実行ごとに検証/テストの中身が入れ替わらないようにしています。

---

## 機械学習モデルの詳細

### 入力
- 形状: `(224, 224, 3)`
- バッチサイズ: `8`

### モデル構成
ImageNet 事前学習済み **ResNet50**（`include_top=False`）をベースにした転移学習モデル。

```
Input (224, 224, 3)
  → Data Augmentation（RandomFlip / Rotation / Zoom / Contrast / Translation, factor=0.2）
  → resnet50.preprocess_input
  → ResNet50（include_top=False, ImageNet 重み）
  → GlobalAveragePooling2D
  → Dense(256, ReLU)
  → Dropout(0.5)
  → Dense(3, softmax)
```

### 学習方法（2 段階の転移学習）
1. **第 1 段階：特徴抽出**
   - ResNet50 の重みを凍結（`trainable=False`）し、追加した分類ヘッドのみを学習。
2. **第 2 段階：ファインチューニング**
   - ResNet50 の最終ブロック（`conv5` で始まる層）のみ解凍。
   - ただし `BatchNormalization` 層は凍結したまま（統計量の破壊を防止）。
   - 学習率を 1/10 に下げて再学習。

学習の安定化のため、以下のコールバックを使用：
- `EarlyStopping`（`val_loss` 監視 / patience=10 / 最良重みを復元）
- `ReduceLROnPlateau`（`val_loss` 監視 / patience=5 / factor=0.5 / min_lr=1e-6）

### ハイパーパラメータ
| 項目 | 値 |
| --- | --- |
| 画像サイズ | 224 × 224 |
| バッチサイズ | 8 |
| エポック数（各段階） | 50（EarlyStopping により早期終了あり） |
| 最適化手法 | Adam |
| 学習率（第 1 段階） | 1e-4 |
| 学習率（第 2 段階） | 1e-5（= 1e-4 / 10） |
| 損失関数 | sparse_categorical_crossentropy |
| Dropout 率 | 0.5 |
| 乱数シード | 42 |

### 再現性
`tf.random.set_seed(42)` / `np.random.seed(42)` およびデータ分割の `seed=42` を固定。

---

## 評価結果

テストデータに対して `sklearn.metrics.classification_report` で評価します。

> **【要記入】** 以下の数値は学習を実行して得られた `classification_report` の出力を転記してください。

| クラス | 適合率 (Precision) | 再現率 (Recall) | F1 スコア |
| --- | --- | --- | --- |
| candy | — | — | — |
| chocolate | — | — | — |
| rice_snack | — | — | — |
| **正解率 (Accuracy)** | | | **—** |

---

## 開発したモデルの課題

> 実装・データから推測される課題（実際の結果を踏まえて加筆してください）

- **データ数が少ない**（合計約 353 枚）ため、過学習しやすく汎化性能に限界がある。
- **クラス不均衡**（rice_snack 155 枚 vs chocolate 98 枚）により、少数クラスの再現率が低下する恐れ。
- テストデータも少数（20% のさらに半分）と小さく、評価指標の分散が大きい（数枚の誤分類で数値が大きく変動）。
- 3 クラスのみで、実運用で想定される多様なお菓子には未対応。

---

## 今後の展望

- データの追加収集とクラス数の拡張。
- クラス不均衡への対策（クラス重み付け、オーバーサンプリング等）。
- 交差検証（k-fold）による評価の安定化。
- より軽量なモデル（MobileNet 等）との比較、推論の高速化。
- 混同行列・Grad-CAM 等による誤分類の可視化と分析。

---

## ディレクトリ構成 / 実行方法

```
data/            # 生画像（変換前）
data_processed/  # 前処理済み画像（学習に使用）
scripts/
  convert_images.py   # data/ → data_processed/ への変換
main_notebook.py      # 学習・評価（marimo ノートブック）
models/               # 保存済みモデル
```

```bash
# 依存関係のインストール（uv 使用）
uv sync

# 画像の前処理
uv run python scripts/convert_images.py

# ノートブックの起動
uv run marimo edit main_notebook.py
```

学習済みモデルは `snack_classifier_finetuned.keras` として保存されます。
