# ANLY-5800 Final Project Proposal

## Team
- **Student Names:** Zihao Huang (zh291), Jiacheng Gao  
- **Project Title:** *Evaluating LLM Robustness Under Social Media Text Perturbations*  
- **Course:** Georgetown ANLY-5800 (Fall 2025)  
- **Preferred Track:** **Track D — Analysis / Evaluation Study**

---

## 1. Problem statement & motivation
Large language models (LLMs) are increasingly used to analyze social-media text, yet such content contains substantial noise—misspellings, slang, emojis, character repetitions, and inconsistent casing. These informal writing patterns differ from the clean text typically used in model evaluation.

**Goal:** Evaluate the robustness of sentiment classification models—including a zero-shot LLM—under realistic social-media perturbations.

**Success outcome:**  
A clear measurement of performance degradation across perturbation types, plus a simple normalization method that partially restores robustness.

---

## 2. Dataset

**Primary dataset: TweetEval — Sentiment**
- Source: HuggingFace TweetEval benchmark  
- Size: ~45k tweets (train/val/test combined)  
- Labels: positive / negative / neutral  
- Domain: real Twitter/X content with natural noise  

**Preprocessing needed:**
- Remove URLs and user handles  
- Tokenization handled by each model’s tokenizer  
- For LLM evaluation: wrap tweet text in a short classification prompt  

---

## 3. Baseline

**Baseline models:**  
1. A strong social-media-trained encoder model  
   - `cardiffnlp/twitter-roberta-base-sentiment-latest`  
2. A general-domain classifier trained on SST-2  

**Baseline metrics:**  
- Accuracy  
- Macro F1  

---

## 4. Approach (beyond baseline)

We will design controlled social-media-style perturbations, including:
- Typos & character noise (insert/delete/replace)  
- Random casing (uppercase/lowercase/mixed)  
- Character elongation (“goooood”)  
- Slang substitutions (“you → u”, “are → r”)  
- Emoji additions  

### Core experiments:
1. **Robustness matrix**  
   - Evaluate each model under each perturbation type.  
2. **Perturbation intensity study**  
   - Level 0 = clean  
   - Level 1 = mild noise  
   - Level 2 = combined noise  

### Improved system:
A simple **text normalization layer** applied before model inference.

### Ablation:
Compare:
- Lowercasing only  
- Lowercasing + repetition collapse  
- Full normalization pipeline  

---

## 5. Compute & resources
- **Jetstream2:** Used only for zero-shot LLM evaluation if needed  
- Encoder models run easily on CPU or small GPU  
- Experiments are inference-based (no heavy training)  
- Expected runtime is manageable on course-provided compute  

---

## 6. Risks & scope

**Risks:**
- Too many perturbation combinations could expand experiment scope  
- Some perturbations may unintentionally alter semantics  
- Zero-shot LLM inference may depend on compute availability  

**Plan B:**  
Reduce scope to:  
- 2 models  
- 3 core perturbation types  
- 1 normalization ablation  

---

## 7. Milestones & timeline

**End of Week 1:**
- Implement data loading + baseline models  
- Evaluate models on clean data  
- Implement 2–3 perturbation functions  
- Write short progress note  

**End of Week 2:**
- Full robustness experiments (perturbation × model grid)  
- Generate initial plots/tables  
- Implement normalization + collect improved results  
- Draft experimental section  

**End of Week 3:**
- Complete ablations  
- Conduct error analysis  
- Finalize report & figures  
- Final presentation & demo  

