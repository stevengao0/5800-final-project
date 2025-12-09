import argparse
from typing import List

from .data_utils import load_sst2_split
from .model_inference import predict_labels
from .evaluation import compute_metrics, save_metrics
from .perturbations import NOISE_FUNCS
from .normalization import apply_normalization_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--noise_type",
        type=str,
        required=True,
        choices=list(NOISE_FUNCS.keys()),
    )
    parser.add_argument("--intensity", type=float, required=True)
    parser.add_argument(
        "--norm_steps",
        type=str,
        default="lower,collapse",
        help="Comma-separated normalization steps, e.g. 'lower,collapse'",
    )
    parser.add_argument("--split", type=str, default="validation")
    args = parser.parse_args()

    noise_type = args.noise_type
    intensity = args.intensity
    norm_steps: List[str] = [s.strip() for s in args.norm_steps.split(",") if s.strip()]

    noise_func = NOISE_FUNCS[noise_type]

    texts, labels = load_sst2_split(split=args.split)

    perturbed_texts = [noise_func(t, intensity) for t in texts]
    normalized_texts = [
        apply_normalization_pipeline(t, norm_steps) for t in perturbed_texts
    ]

    preds = predict_labels(normalized_texts)
    metrics = compute_metrics(labels, preds)
    metrics.update(
        {
            "noise_type": noise_type,
            "intensity": intensity,
            "n_samples": len(labels),
            "dataset": "sst2",
            "norm_steps": norm_steps,
        }
    )

    steps_tag = "-".join(norm_steps) if norm_steps else "none"
    filename = f"sst2_{noise_type}_int{intensity:.2f}_norm-{steps_tag}.json"
    save_metrics(metrics, filename)
    print(metrics)


if __name__ == "__main__":
    main()