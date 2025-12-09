# ANLY-5800 Final Project Proposal

**Student Name:** Zihao Huang(zh291), jiacheng gao 
**Project Title:** Evaluating LLM Robustness Under Social Media Text Perturbations
**Course:** Georgetown ANLY-5800 (Fall 2025)
---

## 1. Problem statement & motivation
Large language models (LLMs) are increasingly used to analyze social media text, yet these texts contain heavy noise—misspellings, abbreviations, emojis, repeated characters, inconsistent casing, and other informal variations. However, most evaluations test models on clean text, which does not reflect real usage conditions.

**Goal:** Evaluate how robust different sentiment classifiers (including a zero-shot LLM) are when social-media-style perturbations are applied to input text.

**Success outcome:**
A clear measurement of performance degradation across perturbation types, plus a simple normalization method that partially restores robustness.


## 2. Dataset

**Primary dataset: TweetEval — Sentiment**
- Source: HuggingFace TweetEval benchmark
- Size: ~45k tweets (train/val/test combined)
- Labels: positive / negative / neutral
- Domain: real Twitter/X content with natural noise

**Preprocessing needed:**
- Remove URLs and usernames (standard TweetEval cleaning).
- Tokenization handled by each model’s tokenizer.
- For LLM evaluation: wrap tweet text in a short classification prompt.

## 3. Baseline

**Baseline model:** 
1. A strong social-media-trained encoder model
    - cardiffnlp/twitter-roberta-base-sentiment-latest
2. A general-domain classifier trained on SST-2

**Baseline metrics:**
1. Accuracy
2. Macro F1

## 4. Approach (beyond baseline)

We will design controlled social-media-style perturbations, including:
- Typos & character noise (insert/delete/replace)
- Random casing / uppercase / lowercase
- Character elongation (“goooood”)
- Slang substitutions (“you → u”, “are → r”)
- Emoji additions

**Core experiments:**
1. Robustness matrix
    - Evaluate each model under each perturbation type.
2. Perturbation intensity study
    - Level 0 = clean
    - Level 1 = mild noise
    - Level 2 = combined noise

**Improved system:**
A simple text normalization layer

**Ablation:**
Compare:
- Only lowercasing
- Lowercasing + repetition collapse
- Full normalization pipeline

## 5. Compute & resources
- **Jetstream2:**
    - Used only for LLM zero-shot evaluation if needed.
- Encoder models run easily on CPU / small GPU.
- No heavy training required — experiments are inference-based
- Expected runtime is manageable on course-provided compute.

## 6. Risks & scope
**Risks:**
- Too many perturbation combinations could create an overly large experiment grid.
- Some perturbations may unintentionally alter semantics.
- Zero-shot LLM evaluation may depend on compute access.

**Plan B:**
If scope is too large, reduce to:
- 2 models
- 3 core perturbation types
- Only 1 normalization ablation

## 7. Milestones & timeline

End of Week 1:
- Implement data loading + baseline models
- Evaluate models on clean data
- Implement 2–3 perturbation functions
- Write short progress note

End of Week 2:

- Full robustness experiments (all perturbations × models)
- Initial plots/tables
- Implement normalization + first improved results
- Draft experimental section

End of Week 3:
- Complete ablations
- Error analysis
- Final report + figures
- Final presentation & demo