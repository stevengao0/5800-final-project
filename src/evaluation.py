import json
import argparse
from pathlib import Path
from typing import List
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

from .config import RESULTS_DIR, FIGS_DIR


def compute_metrics(y_true: List[int], y_pred: List[int]) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {
        "accuracy": acc,
        "f1": f1,
    }


def save_metrics(
    metrics: dict,
    filename: str,
):
    path = RESULTS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {path}")


def load_all_results():
    """
    Load all .json metric files in RESULTS_DIR.
    Expect each file to contain at least:
      - noise_type
      - intensity
      - accuracy
      - f1
    """
    all_rows = []
    for path in RESULTS_DIR.glob("*.json"):
        with open(path) as f:
            data = json.load(f)
        data["file"] = path.name
        all_rows.append(data)
    return all_rows


def plot_accuracy_by_noise_type(rows: list[dict]):
    # use intensity=0.3 (as an example) or average across intensities
    agg = {}
    for r in rows:
        if r.get("noise_type") == "clean":
            continue
        key = r["noise_type"]
        agg.setdefault(key, []).append(r["accuracy"])

    labels = sorted(agg.keys())
    accs = [np.mean(agg[k]) for k in labels]

    plt.figure()
    plt.bar(labels, accs)
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Noise Type")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path = FIGS_DIR / "accuracy_by_noise_type.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


def plot_accuracy_vs_intensity(rows: list[dict], noise_type: str = "spelling"):
    xs = []
    ys = []
    for r in rows:
        if r.get("noise_type") == noise_type:
            xs.append(r["intensity"])
            ys.append(r["accuracy"])

    if not xs:
        print(f"No results for noise_type={noise_type}")
        return

    xs, ys = zip(*sorted(zip(xs, ys), key=lambda x: x[0]))

    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Intensity")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs Intensity ({noise_type})")
    plt.tight_layout()
    out_path = FIGS_DIR / f"accuracy_vs_intensity_{noise_type}.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--make_plots", action="store_true")
    parser.add_argument("--noise_type_for_curve", type=str, default="spelling")
    args = parser.parse_args()

    if args.make_plots:
        rows = load_all_results()
        if not rows:
            print("No results found in", RESULTS_DIR)
            return
        plot_accuracy_by_noise_type(rows)
        plot_accuracy_vs_intensity(rows, noise_type=args.noise_type_for_curve)


if __name__ == "__main__":
    main()
