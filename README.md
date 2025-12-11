# Evaluating LLM Robustness Under Social Media Text Perturbations  
**Zihao Huang (zh291), Jiacheng Gao**  
**ANLY-5800 – Fall 2025**

---

## 1. Overview

Social-media text is noisy, informal, and often contains emojis, misspellings, and other perturbations that do not exist in standard NLP benchmarks. This project evaluates how such noise affects the robustness of a strong pretrained sentiment classifier.

We study two types of perturbations:

1. **Emoji noise** – random insertion of emojis  
2. **Spelling noise** – character-level swaps simulating typos  

We also test a simple normalization defense (lowercasing + repetition collapse) to determine whether lightweight preprocessing can recover robustness.

**Key Findings**

- Emoji noise causes **only mild accuracy degradation**  
- Spelling noise causes **large accuracy drops**  
- Normalization **does not improve** performance  

This project highlights a robustness gap between clean benchmark performance and real-world, noisy text environments.

---

## 2. Repository Structure
5800-final-project/
│
├── experiments/
│   ├── figs/
│   │   ├── accuracy_by_noise_type.png
│   │   └── accuracy_vs_intensity_spelling.png
│   └── results/   # JSON/CSV experiment outputs
│
├── report/
│   ├── Proposal.md
│   └── Report.md   # IMRaD-style final report
│
├── scripts/
│   ├── download_data.sh
│   └── run_all.sh   # Runs all perturbation experiments
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_utils.py
│   ├── evaluation.py
│   ├── model_inference.py
│   ├── normalization.py
│   ├── perturbations.py
│   ├── run_baseline.py
│   ├── run_perturbed.py
│   └── run_perturbed_normalized.py
│
├── requirements.txt
└── README.md

## 3. Installation

Create environment and install dependencies:

```bash
python3 -m venv env
source env/bin/activate 
pip install -r requirements.txt

```

### Noise experiments (unnormalized)

***Emoji noise:***
```
python src/run_perturbed.py --noise_type emoji --intensity 0.3
```
***Spelling noise:***
```
python src/run_perturbed.py --noise_type spelling --intensity 0.5
```

***Run normalization experiments:***
```
python src/run_perturbed_normalized.py --noise_type spelling --intensity 0.5
```

***Run all experiments (recommended):***
```
bash scripts/run_all.sh
```

## 4.Reproducing All Results


```bash
pip install -r requirements.txt

# Clean baseline
python src/run_baseline.py

# All noise experiments
bash scripts/run_all.sh
```

***This will regenerate:***

- accuracy_by_noise_type.png
- accuracy_vs_intensity_spelling.png
- JSON/CSV metrics for all runs

## 5. Methods Summary

### Model
- `distilbert-base-uncased-finetuned-sst-2-english`

### Perturbations
- **Emoji noise** — random emoji insertion  
- **Spelling noise** — adjacent-character swap  

**Noise intensity values:**  
- 0.1  
- 0.3  
- 0.5  

### Defense: Normalization
- Lowercase  
- Collapse repeated characters (e.g., “goooood” → “good”)  

### Metrics
- Accuracy  
- Macro F1  


## 6. Key Results

- Emoji noise → **minor, predictable performance drop**  
- Spelling noise → **sharp degradation even at moderate intensity**  
- Normalization → **does not recover accuracy**  

## 8. Limitations & Future Work

- Only two perturbation types tested  
- Only one pretrained model evaluated  
- Normalization is simple  

### Potential extensions:
- Slang handling  
- Hashtags and code-switching  
- Adversarial noise  
- Robustness-aware finetuning  