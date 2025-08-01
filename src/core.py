"""
Requirements-Testcase Traceability Visualization Core Module
BERTエンベッダー + 評価機能を統合した軽量コア
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import time
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score

class SimpleTraceabilityEngine:
    """
    トレーサビリティエンジン
    BERTエンコーディングから評価まで一貫して処理
    """
    
    def __init__(self, model_name: str, device: str = "auto", 
                 batch_size: int = 8, max_length: int = 256):
        """
        エンジンの初期化
        
        Args:
            model_name: BERTモデル名
            device: 使用デバイス (auto, cpu, cuda)
            batch_size: バッチサイズ
            max_length: 最大トークン長
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        
        # デバイス設定
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f" デバイス: {self.device}")
        print(f" モデル読み込み中: {model_name}")
        
        # BERTモデルとトークナイザーの初期化
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            print(" モデル読み込み完了")
        except Exception as e:
            print(f" モデル読み込みエラー: {e}")
            raise
    
    def encode_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        テキストリストをBERTでエンコード
        
        Args:
            texts: エンコードするテキストのリスト
            show_progress: 進捗表示の有無
            
        Returns:
            np.ndarray: エンコード結果 (n_texts, hidden_size)
        """
        if not texts:
            raise ValueError("テキストリストが空です")
        
        embeddings = []
        start_time = time.time()
        
        if show_progress:
            print(f" テキストエンコード中: {len(texts)}件")
        
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                # トークン化
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                # BERT推論
                outputs = self.model(**inputs)
                
                # [CLS]トークンの隠れ状態を使用（文レベル表現）
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(cls_embeddings)
                
                # 進捗表示
                if show_progress and i % (self.batch_size * 2) == 0:
                    progress = min((i + self.batch_size) / len(texts) * 100, 100)
                    print(f"   進捗: {progress:.0f}%")
        
        encoding_time = time.time() - start_time
        if show_progress:
            print(f" エンコード完了 ({encoding_time:.1f}秒)")
        
        return np.array(embeddings)
    
    def compute_similarity_matrix(self, specs: List[str], tests: List[str]) -> np.ndarray:
        """
        仕様とテスト間の類似度マトリクスを計算
        
        Args:
            specs: 仕様テキストのリスト
            tests: テストテキストのリスト
            
        Returns:
            np.ndarray: 類似度マトリクス (n_specs, n_tests)
        """
        print(f" 類似度マトリクス計算開始 ({len(specs)} × {len(tests)})")
        
        # エンコード
        spec_embeddings = self.encode_texts(specs)
        test_embeddings = self.encode_texts(tests)
        
        # コサイン類似度計算
        similarity_matrix = cosine_similarity(spec_embeddings, test_embeddings)
        
        print(f" 類似度マトリクス計算完了: {similarity_matrix.shape}")
        return similarity_matrix
    
    def create_ground_truth_matrix(self, 
                                  num_specs: int, 
                                  num_tests: int, 
                                  ground_truth: List[Dict]) -> np.ndarray:
        """
        正解マトリクスの作成
        
        Args:
            num_specs: 仕様数
            num_tests: テスト数
            ground_truth: 正解データ
            
        Returns:
            np.ndarray: 正解マトリクス (関連度2以上を1とするバイナリ)
        """
        gt_matrix = np.zeros((num_specs, num_tests))
        
        for link in ground_truth:
            spec_id = link['spec_id']
            test_id = link['test_id']
            relevance = link['relevance']
            
            # 関連度2以上（高関連・完全一致）を正例とする
            if relevance >= 2 and 0 <= spec_id < num_specs and 0 <= test_id < num_tests:
                gt_matrix[spec_id][test_id] = 1
        
        return gt_matrix
    
    def evaluate_at_threshold(self, 
                             similarity_matrix: np.ndarray,
                             ground_truth_matrix: np.ndarray,
                             threshold: float) -> Dict[str, float]:
        """
        指定閾値での評価指標計算
        
        Args:
            similarity_matrix: 類似度マトリクス
            ground_truth_matrix: 正解マトリクス
            threshold: 類似度閾値
            
        Returns:
            Dict[str, float]: 評価指標
        """
        # 予測マトリクス作成
        pred_matrix = (similarity_matrix >= threshold).astype(int)
        
        # フラット化して評価
        gt_flat = ground_truth_matrix.flatten()
        pred_flat = pred_matrix.flatten()
        
        # 基本評価指標
        precision = precision_score(gt_flat, pred_flat, zero_division=0)
        recall = recall_score(gt_flat, pred_flat, zero_division=0)
        f1 = f1_score(gt_flat, pred_flat, zero_division=0)
        
        # カバレッジ計算（少なくとも1つのテストでカバーされる仕様の割合）
        covered_specs = np.sum(np.max(pred_matrix, axis=1))
        total_specs = similarity_matrix.shape[0]
        coverage = covered_specs / total_specs if total_specs > 0 else 0
        
        # 精度計算
        total_predictions = np.sum(pred_flat)
        total_actual = np.sum(gt_flat)
        accuracy = np.sum(gt_flat == pred_flat) / len(gt_flat) if len(gt_flat) > 0 else 0
        
        return {
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'coverage': coverage,
            'total_predictions': int(total_predictions),
            'total_actual': int(total_actual)
        }
    
    def find_best_threshold(self, 
                           similarity_matrix: np.ndarray,
                           ground_truth_matrix: np.ndarray,
                           thresholds: List[float]) -> Tuple[float, List[Dict]]:
        """
        最適閾値の探索
        
        Args:
            similarity_matrix: 類似度マトリクス
            ground_truth_matrix: 正解マトリクス
            thresholds: 試行する閾値のリスト
            
        Returns:
            Tuple[float, List[Dict]]: (最適閾値, 全結果)
        """
        print(f" 最適閾値探索中 (候補: {len(thresholds)}個)")
        
        results = []
        for threshold in thresholds:
            metrics = self.evaluate_at_threshold(
                similarity_matrix, ground_truth_matrix, threshold
            )
            results.append(metrics)
        
        # F1スコアが最大の閾値を選択
        best_result = max(results, key=lambda x: x['f1_score'])
        best_threshold = best_result['threshold']
        
        print(f" 最適閾値: {best_threshold} (F1: {best_result['f1_score']:.3f})")
        
        return best_threshold, results
    
    def predict_missing_links(self, 
                             similarity_matrix: np.ndarray,
                             ground_truth_matrix: np.ndarray,
                             threshold: float,
                             top_k: int = 5) -> List[Tuple[int, int, float]]:
        """
        欠損リンクの予測（既知の正解以外で高類似度のペア）
        
        Args:
            similarity_matrix: 類似度マトリクス
            ground_truth_matrix: 正解マトリクス
            threshold: 閾値
            top_k: 上位k件
            
        Returns:
            List[Tuple]: (spec_id, test_id, similarity)のリスト
        """
        predictions = []
        
        for i in range(similarity_matrix.shape[0]):
            for j in range(similarity_matrix.shape[1]):
                similarity = similarity_matrix[i, j]
                # 閾値以上かつ既知の正解ではないペア
                if similarity >= threshold and ground_truth_matrix[i, j] == 0:
                    predictions.append((i, j, similarity))
        
        # 類似度でソート
        predictions.sort(key=lambda x: x[2], reverse=True)
        
        return predictions[:top_k]
    
    def analyze_by_domain(self, 
                         specs: List[Dict], 
                         tests: List[Dict],
                         similarity_matrix: np.ndarray,
                         threshold: float) -> Dict[str, Any]:
        """
        ドメイン別分析
        
        Args:
            specs: 仕様データ（ドメイン情報付き）
            tests: テストデータ（ドメイン情報付き）
            similarity_matrix: 類似度マトリクス
            threshold: 閾値
            
        Returns:
            Dict[str, Any]: ドメイン別統計
        """
        domain_results = {}
        
        # 全ドメインを取得
        all_domains = set()
        for spec in specs:
            if spec.get('domain'):
                all_domains.add(spec['domain'])
        for test in tests:
            if test.get('domain'):
                all_domains.add(test['domain'])
        
        for domain in all_domains:
            # ドメイン別インデックス取得
            spec_indices = [i for i, spec in enumerate(specs) 
                           if spec.get('domain') == domain]
            test_indices = [i for i, test in enumerate(tests) 
                           if test.get('domain') == domain]
            
            if spec_indices and test_indices:
                # ドメイン内の類似度統計
                domain_similarities = []
                high_similarity_count = 0
                
                for spec_idx in spec_indices:
                    for test_idx in test_indices:
                        sim = similarity_matrix[spec_idx, test_idx]
                        domain_similarities.append(sim)
                        if sim >= threshold:
                            high_similarity_count += 1
                
                domain_results[domain] = {
                    'specs_count': len(spec_indices),
                    'tests_count': len(test_indices),
                    'mean_similarity': float(np.mean(domain_similarities)),
                    'max_similarity': float(np.max(domain_similarities)),
                    'min_similarity': float(np.min(domain_similarities)),
                    'std_similarity': float(np.std(domain_similarities)),
                    'high_similarity_pairs': high_similarity_count,
                    'total_pairs': len(domain_similarities)
                }
        
        return domain_results
    
    def run_complete_evaluation(self, dataset: Dict[str, Any], 
                               thresholds: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        完全な評価パイプライン実行
        
        Args:
            dataset: データセット
            thresholds: 評価する閾値のリスト
            
        Returns:
            Dict[str, Any]: 包括的な評価結果
        """
        if thresholds is None:
            thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        print(" === 完全評価開始 ===")
        start_time = time.time()
        
        # データ抽出
        specs = dataset['specifications']
        tests = dataset['test_cases']
        ground_truth = dataset['ground_truth']
        
        spec_texts = [spec['text'] for spec in specs]
        test_texts = [test['text'] for test in tests]
        
        print(f" データ概要: {len(specs)}仕様 × {len(tests)}テスト, {len(ground_truth)}リンク")
        
        # 類似度マトリクス計算
        similarity_matrix = self.compute_similarity_matrix(spec_texts, test_texts)
        
        # 正解マトリクス作成
        gt_matrix = self.create_ground_truth_matrix(
            len(specs), len(tests), ground_truth
        )
        
        # 最適閾値探索
        best_threshold, all_results = self.find_best_threshold(
            similarity_matrix, gt_matrix, thresholds
        )
        
        # 欠損リンク予測
        missing_links = self.predict_missing_links(
            similarity_matrix, gt_matrix, best_threshold
        )
        
        # ドメイン別分析
        domain_analysis = self.analyze_by_domain(
            specs, tests, similarity_matrix, best_threshold
        )
        
        # 実行時間
        execution_time = time.time() - start_time
        
        # 最適結果の取得
        best_metrics = next(r for r in all_results if r['threshold'] == best_threshold)
        
        # 結果統合
        result = {
            'similarity_matrix': similarity_matrix,
            'ground_truth_matrix': gt_matrix,
            'best_threshold': best_threshold,
            'best_metrics': best_metrics,
            'all_threshold_results': all_results,
            'missing_links': missing_links,
            'domain_analysis': domain_analysis,
            'dataset_info': {
                'name': dataset.get('metadata', {}).get('name', 'Unknown'),
                'num_specs': len(specs),
                'num_tests': len(tests),
                'num_ground_truth': len(ground_truth),
                'domains': list(set(spec.get('domain') for spec in specs if spec.get('domain')))
            },
            'execution_info': {
                'model_name': self.model_name,
                'device': str(self.device),
                'execution_time': execution_time,
                'batch_size': self.batch_size,
                'max_length': self.max_length
            }
        }
        
        print(f" === 完全評価完了 ({execution_time:.1f}秒) ===")
        print(f" 最適F1スコア: {best_metrics['f1_score']:.3f}")
        print(f" カバレッジ: {best_metrics['coverage']:.3f}")
        
        return result

# 便利関数
def quick_evaluate(dataset_path: str, 
                  model_name: str = "cl-tohoku/bert-base-japanese-whole-word-masking",
                  threshold: float = 0.7) -> Dict[str, float]:
    """
    クイック評価関数
    
    Args:
        dataset_path: データセットファイルパス
        model_name: BERTモデル名
        threshold: 類似度閾値
        
    Returns:
        Dict[str, float]: 基本評価指標
    """
    import json
    
    # データセット読み込み
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # エンジン初期化
    engine = SimpleTraceabilityEngine(model_name)
    
    # 基本評価
    specs = [spec['text'] for spec in dataset['specifications']]
    tests = [test['text'] for test in dataset['test_cases']]
    
    similarity_matrix = engine.compute_similarity_matrix(specs, tests)
    gt_matrix = engine.create_ground_truth_matrix(
        len(specs), len(tests), dataset['ground_truth']
    )
    
    return engine.evaluate_at_threshold(similarity_matrix, gt_matrix, threshold)

if __name__ == "__main__":
    # 簡単なテスト
    print(" === コア機能テスト ===")
    
    sample_specs = [
        "ユーザーはログインできる",
        "システムは高速に応答する"
    ]
    sample_tests = [
        "ログイン機能をテストする", 
        "応答時間をテストする"
    ]
    
    try:
        engine = SimpleTraceabilityEngine(
            "cl-tohoku/bert-base-japanese-whole-word-masking"
        )
        similarity_matrix = engine.compute_similarity_matrix(sample_specs, sample_tests)
        
        print(" 類似度マトリクス:")
        print(similarity_matrix)
        print(" テスト完了")
        
    except Exception as e:
        print(f" テストエラー: {e}")