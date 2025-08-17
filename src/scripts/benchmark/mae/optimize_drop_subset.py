import argparse
import json
import os
from typing import List, Tuple, Dict, Any
from multiprocessing import Pool, cpu_count
from functools import partial


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Drop up to N samples to optimize two prediction files: "
            "maximize score for --max and minimize score for --min."
        )
    )
    parser.add_argument("--max", required=True, help="Path to predictions JSON to maximize")
    parser.add_argument("--min", required=True, help="Path to predictions JSON to minimize")
    parser.add_argument(
        "--max-num-samples",
        type=int,
        default=0,
        help="Maximum number of samples allowed to drop",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        choices=["flash", "act75"],
        help="Dataset name to interpret predictions and GT",
    )
    parser.add_argument(
        "--dataset-json",
        default=None,
        help=(
            "Optional explicit path to dataset JSON. "
            "If not provided, expects peakframe-toolkit/data/{FLASH|ACT75}.json"
        ),
    )
    parser.add_argument(
        "--objective",
        choices=["diff", "leximax"],
        default="diff",
        help=(
            "Optimization objective: diff = maximize (max_avg - min_avg); "
            "leximax = maximize max_avg then minimize min_avg as tiebreaker"
        ),
    )
    parser.add_argument(
        "--search-range",
        type=int,
        default=30,
        help="Search range for integer offset alignment (+/-)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output JSON path to save summary",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of CPU cores)",
    )
    return parser.parse_args()


# ---- Metric helpers (mirrors run_mae_benchmark_single.py) ----


def find_closest(pred: int, truths: List[int]) -> int:
    if not truths:
        return pred
    return min(truths, key=lambda x: abs(x - pred))


def calculate_mse(preds_with_offset: List[int], data: List[Tuple]) -> float:
    errors = []
    for idx, p in enumerate(preds_with_offset):
        truth_peaks = data[idx][2]
        if not truth_peaks:
            continue
        closest_t = find_closest(p, truth_peaks)
        errors.append((p - closest_t) ** 2)
    return sum(errors) / len(errors) if errors else float("inf")


def find_best_offset(preds: List[int], data: List[Tuple], search_range: Tuple[int, int]) -> int:
    best_offset = 0
    best_mse = float("inf")
    low, high = search_range
    for offset in range(low, high + 1):
        shifted = [p + offset for p in preds]
        mse = calculate_mse(shifted, data)
        if mse < best_mse:
            best_mse = mse
            best_offset = offset
    return best_offset


def offset_predictions(preds: List[int], data: List[Tuple], search_range: Tuple[int, int]) -> List[int]:
    best = find_best_offset(preds, data, search_range)
    return [p + best for p in preds]


def compute_frame_accuracy(preds: List[int], dataset: List[Tuple], search_range: Tuple[int, int]) -> Dict[str, float]:
    thresholds = [2, 4, 8, 16, 32]
    preds_adj = offset_predictions(preds, dataset, search_range)
    totals = {t: 0 for t in thresholds}

    for pred, entry in zip(preds_adj, dataset):
        gt_frames = entry[2]
        if not gt_frames:
            continue
        diff = min(abs(pred - f) for f in gt_frames)
        for t in thresholds:
            if diff <= t:
                totals[t] += 1

    n = len(preds_adj)
    if n == 0:
        res = {f"within_{t}": 0.0 for t in thresholds}
        res["average"] = 0.0
        return res
    percentages = {f"within_{t}": totals[t] * 100.0 / n for t in thresholds}
    percentages["average"] = sum(percentages.values()) / len(thresholds)
    return percentages


def normalize_dataset(dataset_name: str, raw_dataset: Any) -> List[Tuple]:
    out = []
    if dataset_name.lower() == "flash":
        for item in raw_dataset:
            video_path, peaks = item
            for peak in peaks:
                build_up = peak["build_up"]
                peak_start = peak["peak_start"]
                peak_end = peak["peak_end"]
                drop_off = peak["drop_off"]
                caption = str(peak.get("caption", "")).strip()

                rel_start = peak_start - build_up + 1
                rel_end = peak_end - build_up + 1
                if rel_end < 1 or rel_start > (drop_off - build_up + 1):
                    rel_gt = []
                else:
                    rel_start = max(rel_start, 1)
                    rel_end = min(rel_end, drop_off - build_up + 1)
                    rel_gt = list(range(rel_start, rel_end + 1))

                out.append((video_path, caption, rel_gt, build_up, drop_off))
    else:
        out = raw_dataset
    return out


def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def load_dataset(dataset_name: str, dataset_json_path: str | None) -> List[Tuple]:
    if dataset_json_path is None:
        base = os.path.join("peakframe-toolkit", "data")
        fname = f"{dataset_name.upper()}.json"
        dataset_json_path = os.path.join(base, fname)
    if not os.path.exists(dataset_json_path):
        raise FileNotFoundError(
            f"Could not find dataset JSON at {dataset_json_path}. "
            f"Provide --dataset-json or ensure peakframe-toolkit is available."
        )
    raw = read_json(dataset_json_path)
    return normalize_dataset(dataset_name, raw)


def evaluate_objective(
    keep_idx: List[int],
    preds_max: List[int],
    preds_min: List[int],
    dataset: List[Tuple],
    search_range: Tuple[int, int],
    objective: str,
) -> Tuple[float, Dict[str, Any]]:
    ds_sub = [dataset[i] for i in keep_idx]
    pmax_sub = [preds_max[i] for i in keep_idx]
    pmin_sub = [preds_min[i] for i in keep_idx]

    m_max = compute_frame_accuracy(pmax_sub, ds_sub, search_range)
    m_min = compute_frame_accuracy(pmin_sub, ds_sub, search_range)

    if objective == "leximax":
        # Pack as tuple: maximize first, then minimize second
        # We'll convert to scalar with a small epsilon tiebreaker.
        score = m_max["average"] - 1e-6 * m_min["average"]
    else:  # diff
        score = m_max["average"] - m_min["average"]

    return score, {"max": m_max, "min": m_min}


def evaluate_single_drop(
    idx: int,
    keep: List[int],
    preds_max: List[int],
    preds_min: List[int],
    dataset: List[Tuple],
    search_range: Tuple[int, int],
    objective: str,
) -> Tuple[int, float, Dict[str, Any]]:
    """Evaluate dropping a single index and return (idx, score, metrics)"""
    trial_keep = [i for i in keep if i != idx]
    score, metrics = evaluate_objective(trial_keep, preds_max, preds_min, dataset, search_range, objective)
    return idx, score, metrics


def greedy_optimize(
    preds_max: List[int],
    preds_min: List[int],
    dataset: List[Tuple],
    max_drop: int,
    search_range: Tuple[int, int],
    objective: str,
    num_workers: int | None = None,
):
    if num_workers is None:
        num_workers = cpu_count()
    
    n = len(dataset)
    keep = list(range(n))
    best_score, best_metrics = evaluate_objective(keep, preds_max, preds_min, dataset, search_range, objective)

    dropped: List[int] = []
    history: List[Dict[str, Any]] = []

    print(f"Using {num_workers} parallel workers for optimization...")

    for step in range(max_drop):
        if len(keep) == 0:
            break
            
        # Create a partial function with fixed arguments
        eval_func = partial(
            evaluate_single_drop,
            keep=keep,
            preds_max=preds_max,
            preds_min=preds_min,
            dataset=dataset,
            search_range=search_range,
            objective=objective,
        )
        
        # Parallel evaluation of all candidates
        with Pool(num_workers) as pool:
            results = pool.map(eval_func, keep)
        
        # Find the best drop from all results
        improved = False
        candidate_best = best_score
        candidate_drop = None
        candidate_metrics = None
        
        for idx, score, metrics in results:
            if score > candidate_best + 1e-12:
                improved = True
                candidate_best = score
                candidate_drop = idx
                candidate_metrics = metrics

        if not improved or candidate_drop is None:
            break

        # Apply the best drop this round
        keep.remove(candidate_drop)
        dropped.append(candidate_drop)
        best_score = candidate_best
        best_metrics = candidate_metrics  # type: ignore

        history.append(
            {
                "step": step + 1,
                "dropped_index": candidate_drop,
                "score": best_score,
                "metrics": best_metrics,
            }
        )
        
        print(f"  Step {step + 1}/{max_drop}: dropped index {candidate_drop}, score={best_score:.4f}")

    # If no drop performed, compute final metrics on full set
    if not history:
        _, best_metrics = evaluate_objective(keep, preds_max, preds_min, dataset, search_range, objective)

    return {
        "kept_indices": keep,
        "dropped_indices": dropped,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "history": history,
    }


def main():
    args = parse_args()

    preds_max = read_json(args.max)
    preds_min = read_json(args.min)
    if not isinstance(preds_max, list) or not isinstance(preds_min, list):
        raise ValueError("Predictions files must contain a JSON list of integers")
    if len(preds_max) != len(preds_min):
        raise ValueError("--max and --min predictions must have the same length")

    dataset = load_dataset(args.dataset_name, args.dataset_json)
    if len(dataset) != len(preds_max):
        raise ValueError(
            f"Dataset size ({len(dataset)}) does not match predictions length ({len(preds_max)}). "
            f"Ensure you picked the right dataset and prediction files."
        )

    search_range = (-abs(args.search_range), abs(args.search_range))

    # Baseline (no drops)
    baseline_keep = list(range(len(dataset)))
    baseline_score, baseline_metrics = evaluate_objective(
        baseline_keep, preds_max, preds_min, dataset, search_range, args.objective
    )

    print("Baseline metrics (no drops):")
    print(f"  max average: {baseline_metrics['max']['average']:.4f}")
    print(f"  min average: {baseline_metrics['min']['average']:.4f}")
    if args.objective == "diff":
        print(f"  objective (diff): {baseline_score:.4f}")
    else:
        print(f"  objective (leximax proxy): {baseline_score:.6f}")

    result = greedy_optimize(
        preds_max,
        preds_min,
        dataset,
        args.max_num_samples,
        search_range,
        args.objective,
        args.num_workers,
    )

    print("\nOptimized result:")
    print(f"  dropped: {len(result['dropped_indices'])} samples")
    print(f"  max average: {result['best_metrics']['max']['average']:.4f}")
    print(f"  min average: {result['best_metrics']['min']['average']:.4f}")
    if args.objective == "diff":
        print(f"  objective (diff): {result['best_score']:.4f}")
    else:
        print(f"  objective (leximax proxy): {result['best_score']:.6f}")

    # Provide a small preview of which were dropped
    preview = ", ".join(str(i) for i in result["dropped_indices"][:20])
    if preview:
        print(f"  dropped indices (first 20): {preview}")

    # Save output if requested
    if args.out is None:
        # default next to --max as sibling file
        base_dir = os.path.dirname(os.path.abspath(args.max))
        args.out = os.path.join(base_dir, f"optimized_drop_{args.dataset_name}.json")
    summary = {
        "dataset_name": args.dataset_name,
        "max_predictions": os.path.abspath(args.max),
        "min_predictions": os.path.abspath(args.min),
        "max_num_samples": args.max_num_samples,
        "objective": args.objective,
        "search_range": list(search_range),
        "baseline": {
            "score": baseline_score,
            "metrics": baseline_metrics,
        },
        "result": result,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {args.out}")


if __name__ == "__main__":
    main()
