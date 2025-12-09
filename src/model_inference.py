import os
from typing import List

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

from transformers import pipeline
from .config import MODEL_NAME, PIPELINE_TASK, DEVICE

_classifier = None


def get_classifier():
    """
    Lazily create a HuggingFace pipeline for sentiment classification.
    We force it to use PyTorch (no TensorFlow) to avoid Keras 3 issues.
    """
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            PIPELINE_TASK,
            model=MODEL_NAME,
            device=DEVICE,
            framework="pt",
        )
    return _classifier


def predict_labels(texts: List[str]) -> List[int]:
    """
    Run the classifier on a list of texts.
    Returns integer labels (0 = negative, 1 = positive).
    """
    texts = [str(t) for t in texts]

    clf = get_classifier()

    batch_size = 32
    preds: List[int] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        outputs = clf(batch, truncation=True)

        for out in outputs:
            label = out["label"]
            if "NEG" in label.upper():
                preds.append(0)
            else:
                preds.append(1)

    return preds
