import os
import json
import argparse
import matplotlib.pyplot as plt

def load_metrics(results_path):
    """Load metrics from a JSON file.

    Args:
        results_path (str): Path to directory containing metrics.json

    Returns:
        dict: Loaded metrics data
    """
    metrics_path = os.path.join(results_path, 'metrics.json')
    with open(metrics_path, 'r') as f:
        return json.load(f)

def load_fps_results(results_path):
    """Load FPS results from a JSON file.

    Args:
        results_path (str): Path to directory containing fps_results_spfs.json

    Returns:
        dict: Loaded FPS results data
    """
    fps_path = os.path.join(results_path, 'fps_results_spfs.json')
    with open(fps_path, 'r') as f:
        return json.load(f)

def main(results_root):
    """Main function to generate FPS vs accuracy plot.

    Args:
        results_root (str): Path to root directory containing experiment subdirectories

    Returns:
        None: Generates and saves plot files (png/svg)
    """
    result_dirs = [d for d in os.listdir(results_root)
                   if os.path.isdir(os.path.join(results_root, d))]

    data = []
    dense_dir = 'results_mamba_spfs_ct_1.0_mcs_8'
    optimal_dir = 'results_mamba_spfs_ct_0.7_mcs_8'

    for dir_name in result_dirs:
        dir_path = os.path.join(results_root, dir_name)

        fps_data = load_fps_results(dir_path)
        if len(fps_data) > 1:
            avg_fps = sum(item['fps'] for item in fps_data[1:]) / (len(fps_data) - 1)
        else:
            continue

        metrics = load_metrics(dir_path)
        accuracy = metrics['performance']['within_4']

        print(f"Directory: {dir_path[16:]}, Average FPS: {avg_fps:.4f}, Accuracy: {accuracy:.4f}")

        is_dense = (dir_name == dense_dir)
        is_optimal = (dir_name == optimal_dir)
        data.append( (avg_fps, accuracy, is_dense, is_optimal) )

    # Sort data by FPS
    data.sort(key=lambda x: x[0])

    # Separate data
    dense_fps = None
    dense_acc = None
    optimal_fps = None
    optimal_acc = None
    spfs_fps = []
    spfs_acc = []

    for fps, acc, is_dense, is_optimal in data:
        spfs_fps.append(fps)
        spfs_acc.append(acc)
        if is_dense:
            dense_fps = fps
            dense_acc = acc
        if is_optimal:
            optimal_fps = fps
            optimal_acc = acc

    plt.figure(figsize=(10, 6))

    # Plot all SPFS line (including Dense and Optimal)
    plt.plot(spfs_fps, spfs_acc, 'g-', alpha=0.5, linewidth=2, marker='o', markerfacecolor='green', markersize=4, label='StreamMamba (SPFS)')

    # Plot Dense StreamMamba as filled circle
    if dense_fps is not None and dense_acc is not None:
        plt.scatter(dense_fps, dense_acc, color='blueviolet', marker='D', facecolor='blueviolet', s=45, label='StreamMamba (Dense)', zorder=9)

    # Plot Optimal SPFS with visual separation
    if optimal_fps is not None and optimal_acc is not None:
        # Padding
        plt.scatter(optimal_fps, optimal_acc, color='white', marker='o', s=250,
                    linewidth=3, zorder=5)

        plt.scatter(optimal_fps, optimal_acc, color='green', marker='*', s=160, zorder=9,
                    label='StreamMamba (SPFS optimal)')

    # Plot InternVideo2-B14 baseline
    plt.scatter(1.4059, 74.67, color='blue', marker='x', s=100, label='InternVideo2-B14')

    plt.axvline(x=30, color='black', linestyle='--', label='Real-time Threshold', alpha=0.4, zorder=8)

    max_fps = max([x[0] for x in data]) + 5

    xmin_current = plt.xlim()[0]
    plt.xlim(xmin_current, max_fps)
    plt.axvspan(xmin=30, xmax=max_fps, color='lightgreen', alpha=0.1, zorder=8)

    plt.xlabel('Average FPS')
    plt.ylabel('Accuracy within ±4 frames')
    plt.title('FPS vs Accuracy Tradeoff')
    plt.grid(False)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(results_root, 'fps_accuracy_plot.png'))
    plt.savefig(os.path.join(results_root, 'fps_accuracy_plot.svg'))
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-path', type=str, required=True,
                        help='Path to results directory containing experiment folders')
    args = parser.parse_args()

    main(args.results_path)
