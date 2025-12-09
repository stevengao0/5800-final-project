from .data_utils import load_sst2_split
from .model_inference import predict_labels
from .evaluation import compute_metrics, save_metrics


def main():
    texts, labels = load_sst2_split(split="validation")
    preds = predict_labels(texts)
    metrics = compute_metrics(labels, preds)
    metrics.update(
        {
            "noise_type": "clean",
            "intensity": 0.0,
            "n_samples": len(labels),
        }
    )
    save_metrics(metrics, "baseline_sst2_clean.json")
    print(metrics)


if __name__ == "__main__":
    main()
