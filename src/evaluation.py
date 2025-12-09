import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

from .config import RESULTS_DIR, FIGS_DIR


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return {"accuracy": acc, "f1": f1}


def save_metrics(metrics: Dict, filename: str) -> Path:
    out_path = RESULTS_DIR / filename
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {out_path}")
    return out_path


def load_all_results(pattern: str = "sst2_*.json") -> List[Dict]:
    """Load all result JSONs matching a glob pattern under RESULTS_DIR."""
    results = []
    for path in RESULTS_DIR.glob(pattern):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data["_filename"] = path.name
        results.append(data)
    return results


def plot_accuracy_by_noise_type(results_pattern: str = "sst2_*.json", title: str = ""):
    """Assumes each file has 'noise_type' and 'accuracy'."""
    results = load_all_results(results_pattern)
    if not results:
        print("No results found for pattern:", results_pattern)
        return

    by_noise = {}
    for r in results:
        nt = r.get("noise_type", "unknown")
        by_noise.setdefault(nt, []).append(r["accuracy"])

    noise_types = sorted(by_noise.keys())
    acc_means = [np.mean(by_noise[nt]) for nt in noise_types]

    plt.figure()
    plt.bar(noise_types, acc_means)
    plt.ylabel("Accuracy")
    plt.xlabel("Noise type")
    plt.title(title or "Accuracy by noise type")
    plt.xticks(rotation=45)
    plt.tight_layout()

    out_path = FIGS_DIR / "accuracy_by_noise_type.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot to {out_path}")


def plot_intensity_curve(
    noise_type: str,
    results_pattern: str = "sst2_*.json",
    title: str = "",
):
    """Plot accuracy vs intensity for a single noise_type."""
    results = load_all_results(results_pattern)
    filtered = [r for r in results if r.get("noise_type") == noise_type]
    if not filtered:
        print(f"No results found for noise_type={noise_type}")
        return

    # group by intensity
    xs = []
    ys = []
    for r in sorted(filtered, key=lambda x: x["intensity"]):
        xs.append(r["intensity"])
        ys.append(r["accuracy"])

    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Intensity")
    plt.ylabel("Accuracy")
    plt.title(title or f"Accuracy vs intensity ({noise_type})")
    plt.grid(True)
    plt.tight_layout()

    out_path = FIGS_DIR / f"intensity_curve_{noise_type}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot to {out_path}")