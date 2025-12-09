import argparse
from .data_utils import load_sst2_split
from .model_inference import predict_labels
from .evaluation import compute_metrics, save_metrics
from .perturbations import NOISE_FUNCS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_type", type=str, required=True,
                        choices=list(NOISE_FUNCS.keys()))
    parser.add_argument("--intensity", type=float, required=True)
    parser.add_argument("--split", type=str, default="validation")
    args = parser.parse_args()

    noise_type = args.noise_type
    intensity = args.intensity

    noise_func = NOISE_FUNCS[noise_type]

    texts, labels = load_sst2_split(split=args.split)

    perturbed_texts = [noise_func(t, intensity) for t in texts]

    preds = predict_labels(perturbed_texts)
    metrics = compute_metrics(labels, preds)
    metrics.update(
        {
            "noise_type": noise_type,
            "intensity": intensity,
            "n_samples": len(labels),
        }
    )

    filename = f"sst2_{noise_type}_int{intensity:.2f}.json"
    save_metrics(metrics, filename)
    print(metrics)


if __name__ == "__main__":
    main()
