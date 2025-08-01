"""
Visualization Module
評価結果の可視化機能
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from pathlib import Path

# 日本語フォント対応
try:
    import japanize_matplotlib
    JAPANESE_SUPPORT = True
except ImportError:
    JAPANESE_SUPPORT = False
    print(" japanize-matplotlib未インストール（日本語表示に制限あり）")

def create_performance_plot(results: List[Dict], 
                          output_path: Optional[str] = None,
                          show_plot: bool = True) -> Optional[str]:
    """
    閾値別性能評価グラフの作成
    
    Args:
        results: 閾値別評価結果のリスト
        output_path: 保存パス（Noneの場合は保存しない）
        show_plot: グラフ表示の有無
        
    Returns:
        str: 保存されたファイルパス（保存した場合）
    """
    if not results:
        print(" 評価結果が空です")
        return None
    
    # データ抽出
    thresholds = [r['threshold'] for r in results]
    f1_scores = [r['f1_score'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    coverages = [r['coverage'] for r in results]
    
    # 最適閾値の特定
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    # グラフ作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左側: 基本性能指標
    ax1.plot(thresholds, f1_scores, 'ro-', label='F1スコア', linewidth=2, markersize=6)
    ax1.plot(thresholds, precisions, 'bo-', label='適合率', linewidth=2, markersize=6)
    ax1.plot(thresholds, recalls, 'go-', label='再現率', linewidth=2, markersize=6)
    
    # 最適閾値をマーク
    ax1.axvline(best_threshold, color='red', linestyle='--', alpha=0.7, 
                label=f'最適閾値: {best_threshold}')
    
    ax1.set_xlabel('類似度閾値')
    ax1.set_ylabel('スコア')
    ax1.set_title('閾値別性能評価')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # 右側: カバレッジ
    ax2.plot(thresholds, coverages, 'mo-', label='カバレッジ', linewidth=2, markersize=6)
    ax2.axvline(best_threshold, color='red', linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('類似度閾値')
    ax2.set_ylabel('カバレッジ')
    ax2.set_title('閾値別カバレッジ')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    # 保存
    saved_path = None
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_path = output_path
        print(f" 性能グラフを保存: {output_path}")
    
    # 表示
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return saved_path

def create_similarity_heatmap(similarity_matrix: np.ndarray,
                            ground_truth_matrix: np.ndarray,
                            output_path: Optional[str] = None,
                            show_plot: bool = True) -> Optional[str]:
    """
    類似度マトリクスヒートマップの作成
    
    Args:
        similarity_matrix: 類似度マトリクス
        ground_truth_matrix: 正解マトリクス
        output_path: 保存パス
        show_plot: グラフ表示の有無
        
    Returns:
        str: 保存されたファイルパス
    """
    plt.figure(figsize=(12, 8))
    
    # ヒートマップ作成
    im = plt.imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto', 
                   vmin=0, vmax=1)
    
    # カラーバー
    cbar = plt.colorbar(im, label='コサイン類似度')
    cbar.ax.tick_params(labelsize=10)
    
    # 正解リンクをマーク
    gt_indices = np.where(ground_truth_matrix == 1)
    if len(gt_indices[0]) > 0:
        plt.scatter(gt_indices[1], gt_indices[0], 
                   c='red', s=100, marker='o', alpha=0.8, 
                   edgecolors='white', linewidth=2, label='正解リンク')
    
    # ラベル設定
    plt.xlabel('テストID')
    plt.ylabel('仕様ID')
    plt.title('仕様-テスト類似度マトリクス')
    
    # 軸の調整
    plt.xticks(range(similarity_matrix.shape[1]))
    plt.yticks(range(similarity_matrix.shape[0]))
    
    # 凡例
    if len(gt_indices[0]) > 0:
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    
    # 保存
    saved_path = None
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_path = output_path
        print(f" ヒートマップを保存: {output_path}")
    
    # 表示
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return saved_path

def create_domain_analysis_plot(domain_analysis: Dict[str, Any],
                              output_path: Optional[str] = None,
                              show_plot: bool = True) -> Optional[str]:
    """
    ドメイン別分析グラフの作成
    
    Args:
        domain_analysis: ドメイン別分析結果
        output_path: 保存パス
        show_plot: グラフ表示の有無
        
    Returns:
        str: 保存されたファイルパス
    """
    if not domain_analysis:
        print(" ドメイン分析データがありません")
        return None
    
    domains = list(domain_analysis.keys())
    mean_similarities = [domain_analysis[d]['mean_similarity'] for d in domains]
    specs_counts = [domain_analysis[d]['specs_count'] for d in domains]
    tests_counts = [domain_analysis[d]['tests_count'] for d in domains]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左側: 平均類似度
    bars1 = ax1.bar(domains, mean_similarities, alpha=0.7, color='skyblue')
    ax1.set_xlabel('ドメイン')
    ax1.set_ylabel('平均類似度')
    ax1.set_title('ドメイン別平均類似度')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 値をバーの上に表示
    for bar, value in zip(bars1, mean_similarities):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    # 右側: 仕様・テスト数
    x = np.arange(len(domains))
    width = 0.35
    
    bars2 = ax2.bar(x - width/2, specs_counts, width, label='仕様数', alpha=0.7)
    bars3 = ax2.bar(x + width/2, tests_counts, width, label='テスト数', alpha=0.7)
    
    ax2.set_xlabel('ドメイン')
    ax2.set_ylabel('数')
    ax2.set_title('ドメイン別仕様・テスト数')
    ax2.set_xticks(x)
    ax2.set_xticklabels(domains, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    saved_path = None
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_path = output_path
        print(f" ドメイン分析グラフを保存: {output_path}")
    
    # 表示
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return saved_path

def create_summary_plot(evaluation_result: Dict[str, Any],
                       output_dir: str = "./results",
                       show_plots: bool = True) -> List[str]:
    """
    評価結果の包括的可視化
    
    Args:
        evaluation_result: 完全評価結果
        output_dir: 出力ディレクトリ
        show_plots: グラフ表示の有無
        
    Returns:
        List[str]: 作成されたファイルパスのリスト
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    created_files = []
    
    print(" 可視化開始...")
    
    # 1. 性能評価グラフ
    performance_path = output_dir / "performance_metrics.png"
    saved_path = create_performance_plot(
        evaluation_result['all_threshold_results'],
        output_path=str(performance_path),
        show_plot=show_plots
    )
    if saved_path:
        created_files.append(saved_path)
    
    # 2. 類似度ヒートマップ
    heatmap_path = output_dir / "similarity_heatmap.png"
    saved_path = create_similarity_heatmap(
        evaluation_result['similarity_matrix'],
        evaluation_result['ground_truth_matrix'],
        output_path=str(heatmap_path),
        show_plot=show_plots
    )
    if saved_path:
        created_files.append(saved_path)
    
    # 3. ドメイン分析（データがある場合）
    if evaluation_result.get('domain_analysis'):
        domain_path = output_dir / "domain_analysis.png"
        saved_path = create_domain_analysis_plot(
            evaluation_result['domain_analysis'],
            output_path=str(domain_path),
            show_plot=show_plots
        )
        if saved_path:
            created_files.append(saved_path)
    
    print(f" 可視化完了: {len(created_files)}ファイル作成")
    return created_files

def print_text_summary(evaluation_result: Dict[str, Any]):
    """
    テキスト形式の結果サマリー表示
    
    Args:
        evaluation_result: 評価結果
    """
    print("\n" + "="*60)
    print(" 評価結果サマリー")
    print("="*60)
    
    # 基本指標
    best = evaluation_result['best_metrics']
    print(f" 最適閾値: {evaluation_result['best_threshold']}")
    print(f" F1スコア: {best['f1_score']:.3f}")
    print(f" 適合率: {best['precision']:.3f}")
    print(f" 再現率: {best['recall']:.3f}")
    print(f" カバレッジ: {best['coverage']:.3f}")
    print(f" 精度: {best['accuracy']:.3f}")
    
    # データセット情報
    info = evaluation_result['dataset_info']
    print(f"\n データセット情報:")
    print(f"   名前: {info['name']}")
    print(f"   仕様数: {info['num_specs']}")
    print(f"   テスト数: {info['num_tests']}")
    print(f"   正解リンク数: {info['num_ground_truth']}")
    print(f"   ドメイン: {', '.join(info['domains'])}")
    
    # 実行情報
    exec_info = evaluation_result['execution_info']
    print(f"\n 実行情報:")
    print(f"   モデル: {exec_info['model_name']}")
    print(f"   デバイス: {exec_info['device']}")
    print(f"   実行時間: {exec_info['execution_time']:.1f}秒")
    
    # 欠損リンク予測
    missing_links = evaluation_result.get('missing_links', [])
    if missing_links:
        print(f"\n 予測された欠損リンク (上位3件):")
        for i, (spec_id, test_id, sim) in enumerate(missing_links[:3]):
            print(f"   {i+1}. 仕様{spec_id} ↔ テスト{test_id} (類似度: {sim:.3f})")
    
    # ドメイン別統計
    domain_analysis = evaluation_result.get('domain_analysis', {})
    if domain_analysis:
        print(f"\n ドメイン別統計:")
        for domain, stats in domain_analysis.items():
            print(f"   {domain}: 平均類似度 {stats['mean_similarity']:.3f}, "
                  f"ペア数 {stats['total_pairs']}")
    
    print("="*60)

# 便利関数
def quick_visualize(evaluation_result: Dict[str, Any], 
                   output_dir: str = "./results",
                   show_plots: bool = True) -> List[str]:
    """
    クイック可視化関数
    
    Args:
        evaluation_result: 評価結果
        output_dir: 出力ディレクトリ
        show_plots: グラフ表示の有無
        
    Returns:
        List[str]: 作成されたファイルパス
    """
    # テキストサマリー表示
    print_text_summary(evaluation_result)
    
    # グラフ作成
    return create_summary_plot(evaluation_result, output_dir, show_plots)

if __name__ == "__main__":
    # テスト用のダミーデータ
    print(" 可視化モジュールテスト")
    
    # ダミー評価結果
    dummy_results = [
        {'threshold': 0.5, 'f1_score': 0.6, 'precision': 0.5, 'recall': 0.8, 'coverage': 0.9},
        {'threshold': 0.7, 'f1_score': 0.8, 'precision': 0.8, 'recall': 0.8, 'coverage': 0.8},
        {'threshold': 0.9, 'f1_score': 0.7, 'precision': 0.9, 'recall': 0.6, 'coverage': 0.6}
    ]
    
    # 性能グラフテスト
    create_performance_plot(dummy_results, show_plot=False)
    print(" 可視化モジュールテスト完了")