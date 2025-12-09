from typing import List, Tuple, Optional
from datasets import load_dataset
from .config import DATASET_NAME, DATASET_CONFIG, TEXT_FIELD, LABEL_FIELD, MAX_SAMPLES


def load_sst2_split(
    split: str = "validation",
    max_samples: Optional[int] = MAX_SAMPLES,
) -> Tuple[List[str], List[int]]:
    """
    Load SST-2 data from the GLUE benchmark via `datasets`.

    Returns:
        texts: list of sentences (as plain Python strings)
        labels: list of 0/1 labels (0 = negative, 1 = positive)
    """
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    texts = [str(t) for t in ds[TEXT_FIELD]]
    labels = list(ds[LABEL_FIELD])
    return texts, labels
