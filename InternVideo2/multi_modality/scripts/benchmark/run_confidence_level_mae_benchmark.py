import os
import json
import subprocess
import matplotlib.pyplot as plt
import argparse

def _generate_thresholds():
    """Generates a list of confidence thresholds from 0.0 to 1.0 with 0.05 increments."""
    thresholds = []
    for i in range(0, 101, 5):
        if i % 10 == 0:
            tenth = i // 10
            if tenth == 0:
                thresholds.append("0.0")
            else:
                thresholds.append(f"{tenth / 10:.1f}")
        else:
            thresholds.append(f"{i / 100:.2f}")
    return thresholds

def _run_single_benchmark(ct, benchmark_script, config_dir, config_name, max_consecutive_skips):
    """
    Runs a single benchmark instance with a given confidence threshold and returns
    the 'within_8' performance metric.
    """
    print(f"Running benchmark with confidence threshold = {ct}")

    command = [
        'python3',
        benchmark_script,
        config_dir,
        '--config-name', config_name,
        '--confidence-threshold', ct,
        '--max-consecutive-skips', str(max_consecutive_skips)
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during benchmark run with threshold {ct}: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return None

    # Construct expected folder name based on the benchmark script's output convention
    folder_name = f'results_mamba_spfs_ct_{ct}_mcs_{max_consecutive_skips}'
    metrics_path = os.path.join(folder_name, 'metrics.json')

    if not os.path.exists(metrics_path):
        print(f"Metrics file not found for threshold {ct} at {metrics_path}")
        return None

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    return metrics['performance']['within_8']

def _plot_performance(performance_data, output_prefix):
    """
    Plots performance data and adds a horizontal line for a benchmark.
    """
    performance_data.sort()
    x = [d[0] for d in performance_data]
    y = [d[1] for d in performance_data]

    plt.figure(figsize=(10, 6))

    # Using a different color (red) for the main plot to contrast with the new blue line
    plt.plot(x, y, marker='o', linestyle='-', color='g', label='SPFS Model Performance (±8 Frame Tolerance)')

    # Add InternVideo2 B14 performance line
    plt.axhline(y=0.88, color='b', linestyle='--', label='InternVideo2-B14 performance (88.00%)')

    plt.title('SPFS Performance vs. Confidence Threshold')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Performance (Accuracy within ±8 frames)')
    plt.grid(True)
    plt.ylim(bottom=min(y) - 0.05 if y else 0, top=1.0) # Ensure y-axis covers the full range
    plt.tight_layout()
    plt.legend() # This will now display both labels

    plt.savefig(f'{output_prefix}.png')
    plt.savefig(f'{output_prefix}.svg')

    print(f"Performance plot saved as '{output_prefix}.png' and '{output_prefix}.svg'")


def main():
    parser = argparse.ArgumentParser(
        description="Run SPFS benchmark across various confidence thresholds and plot performance."
    )
    parser.add_argument(
        "config_dir",
        type=str,
        help="Path to the configuration directory for benchmarking (e.g., scripts/spfs/clip/B14/)"
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="delta",
        help="Configuration name for the benchmark (default: 'delta')"
    )
    parser.add_argument(
        "--max-consecutive-skips",
        type=int,
        default=8,
        help="Maximum consecutive skips for the benchmark (default: 8)"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="spfs_performance",
        help="Prefix for the output plot files (e.g., 'spfs_performance.png', 'spfs_performance.svg')"
    )

    args = parser.parse_args()

    benchmark_script = "scripts/benchmark/mae/run_mae_benchmark_single.py"
    thresholds = _generate_thresholds()
    performance_data = []

    existing_folders = True
    missing_thresholds = []

    for ct_str in thresholds:
        folder_name = f'results_mamba_spfs_ct_{ct_str}_mcs_{args.max_consecutive_skips}'
        metrics_path = os.path.join(folder_name, 'metrics.json')

        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            performance_data.append((float(ct_str), metrics['performance']['within_8']))
        else:
            existing_folders = False
            missing_thresholds.append(ct_str)

    if not existing_folders:
        for ct_str in missing_thresholds:
            within_8 = _run_single_benchmark(
                ct_str,
                benchmark_script,
                args.config_dir,
                args.config_name,
                args.max_consecutive_skips
            )
            if within_8 is not None:
                performance_data.append((float(ct_str), within_8))

    if not performance_data:
        print("No performance data collected. Exiting.")
        return

    _plot_performance(performance_data, args.output_prefix)

if __name__ == "__main__":
    main()
