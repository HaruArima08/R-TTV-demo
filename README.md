# R-TTV-demo
日本語の要求仕様とテスト仕様間のトレーサビリティをBERTで分析するデモです。

## ファイル構成

```
R-TTV-demo/
├── README.md                
├── requirements.txt         # 依存関係
├── config.yaml             # 設定ファイル
├── dataset.json            # サンプルデータ
├── src/
│   ├── main.py             # メイン実行スクリプト
│   ├── core.py             # BERTエンベッダー + 評価エンジン
│   └── visualization.py    # 可視化機能
└── results/                # 実行結果
    ├── esults.json
    ├── *.png               # グラフ
    ├── *.npy              # 類似度マトリクス
    └── metrics_summary.csv # CSV形式サマリー
```

## クイックスタート

### 1. セットアップ

```bash
# 依存関係をインストール
pip install -r requirements.txt

# 日本語モデル用（オプション）
pip install protobuf fugashi ipadic
```

### 2. 基本実行

```bash
# サンプルデータで実行
python -m src.main

# 軽量モデルで実行（推奨）
python -m src.main --model bert-base-multilingual-cased

# 結果は ./results/ に保存されます
```

### 3. 結果確認

```bash
# ファイル確認
ls results/

# 主要結果
cat results/evaluation_results.json | grep f1_score
```

## 使用方法

### 基本コマンド

```bash
# デフォルト実行（完全評価）
python -m src.main

# カスタムデータセット
python -m src.main --dataset my_data.json

# クイック評価（閾値固定）
python -m src.main --quick --threshold 0.8

# 可視化なし（高速実行）
python -m src.main --no-plot

# カスタム設定ファイル
python -m src.main --config my_config.yaml

# BERTモデル変更
python -m src.main --model bert-base-multilingual-cased

# CPU強制使用
python -m src.main --device cpu
```

### 出力例

```
=== Requirements-Testcase Traceability Visualization Demo ===
設定ファイル読み込み: config.yaml
データセット読み込み: dataset.json
データ概要: 8仕様 × 10テスト, 10リンク
=== 完全評価開始 ===
デバイス: cuda
モデル読み込み中: bert-base-multilingual-cased
モデル読み込み完了
類似度マトリクス計算開始 (8 × 10)
テキストエンコード中: 8件
エンコード完了 (0.8秒)
テキストエンコード中: 10件
エンコード完了 (0.9秒)
類似度マトリクス計算完了: (8, 10)
最適閾値探索中 (候補: 5個)
最適閾値: 0.7 (F1: 0.800)
=== 完全評価完了 (15.2秒) ===
最適F1スコア: 0.800
カバレッジ: 0.875

評価結果サマリー
最適閾値: 0.7
F1スコア: 0.800
適合率: 0.800
再現率: 0.800
カバレッジ: 0.875
精度: 0.964
```

## 出力ファイル

### 1. 評価結果
- **results.json**: 詳細な評価結果
- **metrics_summary.csv**: 閾値別指標（Excel対応）
- **similarity_matrix.npy**: 類似度マトリクス（NumPy形式）
- **ground_truth_matrix.npy**: 正解マトリクス

### 2. 可視化
- **performance_metrics.png**: 閾値別性能グラフ
- **similarity_heatmap.png**: 類似度マトリクスヒートマップ
- **domain_analysis.png**: ドメイン別分析（該当する場合）

## カスタマイズ

### データセット形式

```json
{
  "metadata": {
    "name": "My Dataset",
    "description": "説明"
  },
  "specifications": [
    {
      "id": 0,
      "text": "ユーザーはログインできる",
      "domain": "認証"
    }
  ],
  "test_cases": [
    {
      "id": 0,
      "text": "ログイン機能をテストする",
      "domain": "認証"
    }
  ],
  "ground_truth": [
    {
      "spec_id": 0,
      "test_id": 0,
      "relevance": 3,
      "label": "完全一致"
    }
  ]
}
```

**関連度レベル:**
- `3`: 完全一致（直接対応）
- `2`: 高関連（間接的関連、異常系など）
- `1`: 部分関連（弱い関連）

### 設定変更（config.yaml）

```yaml
# モデル変更
model_name: "bert-base-multilingual-cased"

# 評価設定
thresholds: [0.6, 0.7, 0.8]
default_threshold: 0.75

# 性能設定
batch_size: 16      # GPU使用時は大きく
max_length: 512     # 長文の場合は大きく

# 出力設定
save_plots: true
show_plots: false   # サーバー実行時はfalse
```

## トラブルシューティング

### よくある問題

**1. GPU メモリ不足**
```bash
# CPU使用
python -m src.main --device cpu

# バッチサイズ削減
# config.yamlで batch_size: 4 に変更
```

**2. 日本語モデルの依存関係エラー**
```bash
# 依存関係インストール
pip install protobuf fugashi ipadic

# または軽量モデル使用
python -m src.main --model bert-base-multilingual-cased
```

**3. データセット形式エラー**
```bash
# サンプルデータセットで確認
python -c "import json; print(json.load(open('dataset.json')))"
```

### 推奨モデル

| モデル | 依存関係 | 日本語性能 | 速度 |
|--------|----------|------------|------|
| `bert-base-multilingual-cased` | 少ない | 中程度 | 高速 |
| `cl-tohoku/bert-base-japanese` | 多い | 高い | 中速 |
| `rinna/japanese-roberta-base` | 中程度 | 高い | 中速 |
| `distilbert-base-multilingual-cased` | 少ない | 中程度 | 最高速 |


## 今すぐ試してみる

```bash
# 1. セットアップ
pip install -r requirements.txt

# 2. 軽量モデルで実行
python -m src.main --model bert-base-multilingual-cased --quick --threshold 0.7

# 3. 完全実行
python -m src.main --model bert-base-multilingual-cased
```