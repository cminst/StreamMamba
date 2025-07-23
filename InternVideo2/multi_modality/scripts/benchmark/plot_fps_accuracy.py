import os
import json
import argparse
import matplotlib.pyplot as plt

def load_metrics(results_path):
    metrics_path = os.path.join(results_path, 'metrics.json')
    with open(metrics_path, 'r') as f:
        return json.load(f)

def load_fps_results(results_path):
    fps_path = os.path.join(results_path, 'fps_results_spfs.json')
    with open(fps_path, 'r') as f:
        return json.load(f)

def main(results_root):
    result_dirs = [d for d in os.listdir(results_root)
                   if os.path.isdir(os.path.join(results_root, d))]

    fps_values = []
    accuracy_values = []

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

        fps_values.append(avg_fps)
        accuracy_values.append(accuracy)

    plt.figure(figsize=(10, 6))
    plt.scatter(fps_values, accuracy_values)

    plt.xlabel('Average FPS')
    plt.ylabel('Accuracy within ±4 frames')
    plt.title('FPS vs Accuracy Tradeoff')

    sorted_indices = sorted(range(len(fps_values)), key=lambda k: fps_values[k])
    sorted_fps = [fps_values[i] for i in sorted_indices]
    sorted_accuracy = [accuracy_values[i] for i in sorted_indices]

    plt.plot(sorted_fps, sorted_accuracy, 'g-', alpha=0.5, linewidth=2, marker='o', markersize=6, label='StreamMamba (SPFS)')

    plt.grid(True)
    plt.tight_layout()

    # Add InternVideo2-B14 baseline point
    plt.scatter(1.4059, 74.67, color='blue', marker='*', s=150, label='InternVideo2-B14')
    plt.legend()
    plt.savefig(os.path.join(results_root, 'fps_accuracy_plot.png'))
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-path', type=str, required=True,
                        help='Path to results directory containing experiment folders')
    args = parser.parse_args()

    main(args.results_path)
