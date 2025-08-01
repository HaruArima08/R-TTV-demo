"""
Requirements-Testcase Traceability Visualization Demo
メイン実行スクリプト - 全機能を統合したエントリーポイント
"""

import json
import yaml
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# プロジェクトモジュールのインポート
try:
    from .core import SimpleTraceabilityEngine
    from .visualization import quick_visualize, print_text_summary
except ImportError:
    # 直接実行の場合の絶対インポート
    try:
        from core import SimpleTraceabilityEngine
        from visualization import quick_visualize, print_text_summary
    except ImportError as e:
        print(f"モジュールインポートエラー: {e}")
        print("src/ ディレクトリから実行するか、プロジェクトルートから python -m src.main で実行してください")
        sys.exit(1)

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    設定ファイルの読み込み
    
    Args:
        config_path: 設定ファイルパス
        
    Returns:
        Dict[str, Any]: 設定辞書
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"設定ファイル読み込み: {config_path}")
        return config
    except FileNotFoundError:
        print(f"設定ファイルが見つかりません: {config_path}")
        print("デフォルト設定を使用します")
        return get_default_config()
    except yaml.YAMLError as e:
        print(f"設定ファイル読み込みエラー: {e}")
        sys.exit(1)

def get_default_config() -> Dict[str, Any]:
    """デフォルト設定を取得"""
    return {
        'model_name': 'cl-tohoku/bert-base-japanese-whole-word-masking',
        'device': 'auto',
        'default_threshold': 0.7,
        'thresholds': [0.5, 0.6, 0.7, 0.8, 0.9],
        'output_dir': './results',
        'save_plots': True,
        'show_plots': True,
        'default_dataset': 'sample_dataset.json',
        'batch_size': 8,
        'max_length': 256
    }

def load_dataset(dataset_path: str) -> Dict[str, Any]:
    """
    データセットの読み込み
    
    Args:
        dataset_path: データセットファイルパス
        
    Returns:
        Dict[str, Any]: データセット
    """
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"データセット読み込み: {dataset_path}")
        
        # 基本検証
        required_keys = ['specifications', 'test_cases', 'ground_truth']
        for key in required_keys:
            if key not in dataset:
                raise ValueError(f"必須キー '{key}' がデータセットにありません")
        
        specs_count = len(dataset['specifications'])
        tests_count = len(dataset['test_cases']) 
        links_count = len(dataset['ground_truth'])
        
        print(f"データ概要: {specs_count}仕様 × {tests_count}テスト, {links_count}リンク")
        
        return dataset
        
    except FileNotFoundError:
        print(f"データセットファイルが見つかりません: {dataset_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"JSONファイル形式エラー: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"データセット読み込みエラー: {e}")
        sys.exit(1)

def save_results(results: Dict[str, Any], output_dir: str) -> List[str]:
    """
    結果の保存
    
    Args:
        results: 評価結果
        output_dir: 出力ディレクトリ
        
    Returns:
        List[str]: 保存されたファイルパス
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    try:
        # 1. メイン結果をJSON形式で保存（NumPy配列は除外）
        results_copy = results.copy()
        results_copy.pop('similarity_matrix', None)
        results_copy.pop('ground_truth_matrix', None)
        
        results_path = output_dir / "evaluation_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_copy, f, ensure_ascii=False, indent=2)
        saved_files.append(str(results_path))
        
        # 2. NumPy配列を個別保存
        import numpy as np
        
        sim_matrix_path = output_dir / "similarity_matrix.npy"
        np.save(sim_matrix_path, results['similarity_matrix'])
        saved_files.append(str(sim_matrix_path))
        
        gt_matrix_path = output_dir / "ground_truth_matrix.npy"
        np.save(gt_matrix_path, results['ground_truth_matrix'])
        saved_files.append(str(gt_matrix_path))
        
        # 3. CSVサマリー（指標のみ）
        import pandas as pd
        
        metrics_data = []
        for result in results['all_threshold_results']:
            metrics_data.append({
                '閾値': result['threshold'],
                'F1スコア': result['f1_score'],
                '適合率': result['precision'],
                '再現率': result['recall'],
                'カバレッジ': result['coverage'],
                '精度': result['accuracy']
            })
        
        df = pd.DataFrame(metrics_data)
        csv_path = output_dir / "metrics_summary.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        saved_files.append(str(csv_path))
        
        print(f"結果保存完了: {len(saved_files)}ファイル ({output_dir})")
        
    except Exception as e:
        print(f"結果保存エラー: {e}")
    
    return saved_files

def run_quick_evaluation(dataset: Dict[str, Any], 
                        config: Dict[str, Any],
                        threshold: float) -> Dict[str, Any]:
    """
    クイック評価の実行
    
    Args:
        dataset: データセット
        config: 設定
        threshold: 類似度閾値
        
    Returns:
        Dict[str, Any]: 基本評価結果
    """
    print("=== クイック評価開始 ===")
    
    # エンジン初期化
    engine = SimpleTraceabilityEngine(
        model_name=config['model_name'],
        device=config['device'],
        batch_size=config.get('batch_size', 8),
        max_length=config.get('max_length', 256)
    )
    
    # 基本データ抽出
    specs = [spec['text'] for spec in dataset['specifications']]
    tests = [test['text'] for test in dataset['test_cases']]
    
    # 類似度計算
    similarity_matrix = engine.compute_similarity_matrix(specs, tests)
    gt_matrix = engine.create_ground_truth_matrix(
        len(specs), len(tests), dataset['ground_truth']
    )
    
    # 評価
    metrics = engine.evaluate_at_threshold(similarity_matrix, gt_matrix, threshold)
    
    print(f"=== クイック評価完了 ===")
    print(f"F1スコア: {metrics['f1_score']:.3f}")
    print(f"適合率: {metrics['precision']:.3f}")
    print(f"再現率: {metrics['recall']:.3f}")
    print(f"カバレッジ: {metrics['coverage']:.3f}")
    
    return {
        'type': 'quick_evaluation',
        'threshold': threshold,
        'metrics': metrics,
        'similarity_matrix': similarity_matrix,
        'ground_truth_matrix': gt_matrix,
        'dataset_info': {
            'num_specs': len(specs),
            'num_tests': len(tests),
            'num_ground_truth': len(dataset['ground_truth'])
        }
    }

def run_full_evaluation(dataset: Dict[str, Any], 
                       config: Dict[str, Any]) -> Dict[str, Any]:
    """
    完全評価の実行
    
    Args:
        dataset: データセット
        config: 設定
        
    Returns:
        Dict[str, Any]: 包括的評価結果
    """
    print("=== 完全評価開始 ===")
    
    # エンジン初期化
    engine = SimpleTraceabilityEngine(
        model_name=config['model_name'],
        device=config['device'],
        batch_size=config.get('batch_size', 8),
        max_length=config.get('max_length', 256)
    )
    
    # 完全評価実行
    results = engine.run_complete_evaluation(
        dataset, 
        thresholds=config.get('thresholds', [0.5, 0.6, 0.7, 0.8, 0.9])
    )
    
    return results

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="Requirements-Testcase Traceability Visualization Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  使用例:
  # 基本実行
  python main.py
  
  # カスタムデータセット
  python main.py --dataset my_data.json
  
  # クイック評価
  python main.py --quick --threshold 0.8
  
  # 可視化なし（高速）
  python main.py --no-plot
  
  # カスタム設定
  python main.py --config my_config.yaml
        """
    )
    
    parser.add_argument('--dataset', '-d', type=str,
                       help='データセットファイルパス')
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                       help='設定ファイルパス')
    parser.add_argument('--output', '-o', type=str,
                       help='結果出力ディレクトリ')
    parser.add_argument('--threshold', '-t', type=float,
                       help='類似度閾値（クイック評価用）')
    parser.add_argument('--quick', action='store_true',
                       help='クイック評価（閾値探索なし）')
    parser.add_argument('--no-plot', action='store_true',
                       help='グラフ表示をスキップ')
    parser.add_argument('--model', '-m', type=str,
                       help='BERTモデル名')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'],
                       help='使用デバイス')
    
    args = parser.parse_args()
    
    try:
        print(" === Requirements-Testcase Traceability Visualization Demo ===")
        
        # 設定読み込み
        config = load_config(args.config)
        
        # コマンドライン引数で設定をオーバーライド
        if args.model:
            config['model_name'] = args.model
        if args.device:
            config['device'] = args.device
        if args.output:
            config['output_dir'] = args.output
        if args.no_plot:
            config['show_plots'] = False
            config['save_plots'] = False
        
        # データセット読み込み
        dataset_path = args.dataset or config.get('default_dataset', 'sample_dataset.json')
        dataset = load_dataset(dataset_path)
        
        # 評価実行
        if args.quick:
            # クイック評価
            threshold = args.threshold or config.get('default_threshold', 0.7)
            results = run_quick_evaluation(dataset, config, threshold)
            
        else:
            # 完全評価
            results = run_full_evaluation(dataset, config)
            
            # 結果保存
            output_dir = config['output_dir']
            saved_files = save_results(results, output_dir)
            
            # 可視化
            if config.get('save_plots', True) or config.get('show_plots', True):
                try:
                    viz_files = quick_visualize(
                        results, 
                        output_dir=output_dir,
                        show_plots=config.get('show_plots', True)
                    )
                    print(f"可視化ファイル: {len(viz_files)}個作成")
                except Exception as e:
                    print(f"可視化エラー: {e}")
            else:
                # 可視化なしでもテキストサマリーは表示
                print_text_summary(results)
        
        print("\n === 実行完了 ===")
        
    except KeyboardInterrupt:
        print("\n 実行が中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\n 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()