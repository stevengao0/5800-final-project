# Evaluating LLM Robustness Under Social Media Text Perturbations

This project is for **Georgetown ANLY-5800 (Fall 2025)**.  
We study how **social media–style noise** (emoji, spelling errors, slang, hashtags, repetition, and code-switching) affects the performance of a transformer-based sentiment classifier.

The core questions:

1. How much does accuracy drop under different types of perturbations?
2. Which noise types are most harmful?
3. How does performance change as we increase noise intensity?

---

## 1. Repository Structure

```text
5800-final-project/
├── README.md
├── requirements.txt
├── .gitignore
├── report/
│   └── report.md              # IMRaD-style project report (to export as PDF)
├── src/
│   ├── __init__.py
│   ├── config.py              # configuration: model name, noise settings, paths
│   ├── data_utils.py          # load SST-2 / other datasets
│   ├── perturbations.py       # social-media-style text noise functions
│   ├── model_inference.py     # model loading and batched prediction
│   ├── evaluation.py          # metrics + plotting
│   ├── run_baseline.py        # evaluate model on clean test set
│   └── run_perturbed.py       # evaluate model under specified noise type/intensity
├── experiments/
│   ├── results/               # JSON/CSV metrics for each experiment
│   └── figs/                  # plots (accuracy vs noise type, etc.)
├── notebooks/
│   ├── eda_sst2.ipynb         # optional: EDA notebook
│   └── error_analysis.ipynb   # optional: manual error analysis
└── scripts/
    ├── run_all.sh             # run all perturbation experiments
    └── download_data.sh       # optional: data download helper
