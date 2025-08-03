import os
import json
import argparse
import matplotlib.pyplot as plt

STYLES = {
    'streambm_spfs': {
        'color': 'green',
        'marker': 'o',
        'linestyle': '-',
        'label': 'StreamMamba (SPFS)',
        'zorder': 5
    },
    'streambm_usp': {
        'color': 'green',
        'marker': 's',
        'linestyle': '--',
        'label': 'StreamMamba (USP)',
        'zorder': 5
    },
    'streambm_dense': {
        'color': 'green',
        'marker': 'D',
        's': 70,
        'label': 'StreamMamba (Dense)',
        'zorder': 6
    },
    'streambm_optimal': {
        'color': 'gold',
        'marker': '*',
        's': 300,
        'label': 'StreamMamba (SPFS Optimal)',
        'edgecolors': 'black',
        'linewidth': 0.5,
        'zorder': 10
    },
    'internvideo2_b14': {
        'color': '#377eb8',
        'marker': 'x',
        's': 100,
        'label': 'InternVideo2-B14'
    },
    'internvideo2_6b': {
        'color': '#ff7f00',
        'marker': 'x',
        's': 100,
        'label': 'InternVideo2-6B'
    },
    'streaming_lstm': {
        'color': '#e41a1c',
        'marker': '^',
        's': 120,
        'label': 'Streaming LSTM (Dense)'
    }
}

MISC_DATA = {
    'dense_dir_name': 'results_mamba_spfs_ct_1.0_mcs_8',
    'optimal_dir_name': 'results_mamba_spfs_ct_0.85_mcs_8',
    'realtime_threshold_fps': 24,
    'max_fps_data_limit': 40,
    'baselines': {
        'internvideo2_b14': {'fps': 1.4059, 'accuracy': 74.67},
        'internvideo2_6b': {'fps': 0.0, 'accuracy': 84.00},
        'streaming_lstm': {'fps': 21.3665, 'accuracy': 64.00}
    }
}

def load_metrics(results_path):
    metrics_path = os.path.join(results_path, 'metrics.json')
    with open(metrics_path, 'r') as f:
        return json.load(f)

def load_fps_results(results_path):
    fps_path = os.path.join(results_path, 'fps_results_spfs.json')
    with open(fps_path, 'r') as f:
        return json.load(f)

def load_all_data(results_root):
    spfs_data = []
    result_dirs = [d for d in os.listdir(results_root)
                   if os.path.isdir(os.path.join(results_root, d))]

    for dir_name in result_dirs:
        dir_path = os.path.join(results_root, dir_name)
        try:
            fps_data = load_fps_results(dir_path)
            avg_fps = sum(item['fps'] for item in fps_data[1:]) / (len(fps_data) - 1) if len(fps_data) > 1 else 0
            if not avg_fps or avg_fps > MISC_DATA['max_fps_data_limit']:
                continue

            metrics = load_metrics(dir_path)
            accuracy = metrics['performance']['within_4']

            is_dense = (dir_name == MISC_DATA['dense_dir_name'])
            is_optimal = (dir_name == MISC_DATA['optimal_dir_name'])

            spfs_data.append({'fps': avg_fps, 'acc': accuracy, 'is_dense': is_dense, 'is_optimal': is_optimal})
            print(f"SPFS - Dir: {dir_name}, FPS: {avg_fps:.2f}, Acc: {accuracy:.2f}")

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Could not process SPFS dir {dir_name}: {e}")
            continue

    spfs_data.sort(key=lambda x: x['fps'])

    usp_data = []
    uniform_results_root = os.path.join(os.path.dirname(results_root), 'results_uniform')
    if os.path.exists(uniform_results_root):
        uniform_dirs = [d for d in os.listdir(uniform_results_root) if os.path.isdir(os.path.join(uniform_results_root, d))]

        for dir_name in uniform_dirs:
            dir_path = os.path.join(uniform_results_root, dir_name)
            try:
                fps_data = load_fps_results(dir_path)
                avg_fps = sum(item['fps'] for item in fps_data) / len(fps_data) if fps_data else 0
                if not avg_fps or avg_fps > MISC_DATA['max_fps_data_limit']:
                    continue

                metrics = load_metrics(dir_path)
                accuracy = metrics['performance']['within_4']

                usp_data.append({'fps': avg_fps, 'acc': accuracy})
                print(f"USP - Dir: {dir_name}, FPS: {avg_fps:.2f}, Acc: {accuracy:.2f}")

            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                print(f"Could not process USP dir {dir_name}: {e}")
                continue

    dense_point = next((p for p in spfs_data if p['is_dense']), None)
    if dense_point:
        usp_data.append({'fps': dense_point['fps'], 'acc': dense_point['acc']})

    usp_data.sort(key=lambda x: x['fps'])

    return spfs_data, usp_data

def generate_plot(spfs_data, usp_data, results_root):
    plt.figure(figsize=(6, 5))

    if spfs_data:
        spfs_style = STYLES['streambm_spfs']
        plt.plot(
            [d['fps'] for d in spfs_data],
            [d['acc'] for d in spfs_data],
            color=spfs_style['color'],
            marker=spfs_style['marker'],
            linestyle=spfs_style['linestyle'],
            label=spfs_style['label'],
            zorder=spfs_style['zorder'],
            linewidth=2.5,
            markersize=6
        )

    if usp_data:
        usp_style = STYLES['streambm_usp']
        plt.plot(
            [d['fps'] for d in usp_data],
            [d['acc'] for d in usp_data],
            color=usp_style['color'],
            marker=usp_style['marker'],
            linestyle=usp_style['linestyle'],
            label=usp_style['label'],
            zorder=usp_style['zorder'],
            linewidth=2,
            alpha=0.8
        )

    dense_point = next((p for p in spfs_data if p['is_dense']), None)
    if dense_point:
        style = STYLES['streambm_dense']
        plt.scatter(dense_point['fps'], dense_point['acc'], **style)

    optimal_point = next((p for p in spfs_data if p['is_optimal']), None)
    if optimal_point:
        style = STYLES['streambm_optimal']
        plt.scatter(optimal_point['fps'], optimal_point['acc'], **style)

    for name, data in MISC_DATA['baselines'].items():
        style = STYLES[name]
        plt.scatter(data['fps'], data['accuracy'], **style)

    plt.axvline(
        x=MISC_DATA['realtime_threshold_fps'],
        color='black',
        linestyle='--',
        label='Real-time Threshold (24 FPS)',
        alpha=0.5,
        zorder=2
    )

    max_fps_plot = MISC_DATA['max_fps_data_limit']
    plt.axvspan(
        xmin=MISC_DATA['realtime_threshold_fps'],
        xmax=max_fps_plot,
        color='lightgreen',
        alpha=0.2,
        zorder=0
    )

    plt.grid(False)
    plt.xlim(left=-2, right=max_fps_plot)
    plt.ylim(bottom=35)
    plt.xlabel('Average FPS (Throughput)', fontsize=12)
    plt.ylabel('Accuracy within ±4 frames (%)', fontsize=12)
    plt.title('Performance-Throughput Trade-off', fontsize=14)#, fontweight='bold')
    plt.legend(fontsize=10)
    plt.tight_layout()

    output_png = os.path.join(results_root, 'performance_throughput_plot.png')
    output_svg = os.path.join(results_root, 'performance_throughput_plot.svg')
    plt.savefig(output_png, dpi=300)
    plt.savefig(output_svg)
    print(f"Plot saved to {output_png} and {output_svg}")
    plt.close()

def main(results_root):
    spfs_data, usp_data = load_all_data(results_root)
    if not spfs_data:
        print("No valid SPFS data found. Aborting plot generation.")
        return
    generate_plot(spfs_data, usp_data, results_root)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate FPS vs. Accuracy plots from experiment results.")
    parser.add_argument('--results-path', type=str, required=True,
                        help='Path to results directory containing SPFS experiment folders')
    args = parser.parse_args()
    main(args.results_path)
