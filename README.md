# Mental Health Post Classification: RoBERTa vs. MentalRoBERTa
**A Comparative Study of Transformers and Text Preprocessing for Analytics**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
https://colab.research.google.com/github/jahanara-m/Comparison__Transformers/blob/main/Comparison_Transformers.ipynb
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **TL;DR:** A controlled experiment comparing transformer models and text preprocessing for classifying mental health posts. Key finding: Domain-specific pretraining (`MentalRoBERTa`) is more impactful than advanced text cleaning, providing an efficient, high-accuracy pipeline.

---

##  Authentication Note
**Important:** The `mental/mental-roberta-base` model is a gated model on Hugging Face Hub.
- You will need a free Hugging Face account.
- You must agree to the model's terms of use on its [model card](https://huggingface.co/mental/mental-roberta-base).
- When prompted in the notebook, you will need to log in with a User Access Token from your [Hugging Face settings](https://huggingface.co/settings/tokens). A token with "read" permissions is sufficient.
- This is only required for Experiments 3 & 4. The standard `roberta-base` model does not require authentication.

---

##  Quick Results

| Experiment  |     Model     | Preprocessing | Accuracy | Precision | Recall | F1 Score |
|-------------|---------------|---------------|----------|-----------|--------|----------|
|      1      | RoBERTa-base  |     Basic     | 83.65%   |   83.70%  | 83.65% |  83.62%  |
|      2      | RoBERTa-base  |    Advanced   | 82.26%   |   82.99%  | 82.26% |  82.40%  | 
|      3      | MentalRoBERTa |     Basic     | 84.90%   |   84.94%  | 84.90% |  84.87%  |
|      4      | MentalRoBERTa |    Advanced   | 81.82%   |   82.30%  | 81.82% |  81.80%  |

---

##  Key Insight & Recommendation

**Finding:** While the domain-specific model (MentalRoBERTa) performed best overall, advanced preprocessing (stopword removal & stemming) consistently degraded performance for both models. This suggests that for nuanced mental health language, standard text normalization can remove critical signal.
**Recommendation:** The optimal and most efficient pipeline is MentalRoBERTa with only basic preprocessing, achieving an F1-score of **84.87%**. This insight could save significant feature engineering effort in a production system with no loss in accuracy.

---

##  Analytical Workflow & Notebook Structure

The analysis follows a structured, reproducible pipeline in `Comparison_Transformers.ipynb`:

| Block |               Content             |                                Purpose                                |
|-------|-----------------------------------|-----------------------------------------------------------------------|
|   1   |         Setup & Installation      |       Installs libraries, imports dependencies, sets random seeds     |
|   2   |   Configuration & Preprocessing   |      Defines `Config` class, text cleaning functions, data loading    |
|   3   | Core Classes (`ExperimentRunner`) |   Contains the main training/evaluation logic (modular OOP design)    |
|  4-7  |          Experiments 1-4          |  Executes the 2x2 comparison (RoBERTa/MentalRoBERTa × Basic/Advanced) |
|   8   |         Results Comparison        | Aggregates metrics, generates comparison plots, identifies best model |

---

##  How to Run & Reproduce

### **Option A: One-Click Live Demo (Recommended)**
The easiest way to review the full analysis is in Google Colab:
1.  Click the **[Open In Colab](https://colab.research.google.com/github/jahanara-m/Comparison__Transformers/blob/main/Comparison_Transformers.ipynb)** badge above.
2.  In Colab, go to **Runtime > Run all** (`Ctrl+F9` / `Cmd+F9`).
3.  *For MentalRoBERTa experiments:* **You will be prompted to log in** with a [Hugging Face token](https://huggingface.co/settings/tokens) (free, read-access is enough). See the [Authentication Note](#authentication-note) above.

### **Option B: Run Locally**
1.  **Clone** the repository:
    ```bash
    git clone https://github.com/jahanara-m/Comparison__Transformers.git
    cd Comparison__Transformers
    ```
2.  **Install** dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *The `Dataset.csv` file is included in this repository.*
3.  **Launch** Jupyter and open the notebook:
    ```bash
    jupyter notebook Comparison_Transformers.ipynb
    ```
4.  *For MentalRoBERTa experiments:* You will be prompted to log in with a Hugging Face token when the relevant code cell runs.

---

##  Project Contents

├── Comparison_Transformers.ipynb # Main analysis notebook
├── README.md # This documentation
├── requirements.txt # Python package dependencies
├── Dataset.csv # The labeled dataset for classification
└── results/ # Auto-generated outputs (not in version control)


---

##  Dependencies

Core Python packages are listed in `requirements.txt`. Main libraries include:
- `torch`, `transformers` (for model loading and training)
- `pandas`, `scikit-learn` (for data handling and metrics)
- `nltk` (for text preprocessing)
- `seaborn`, `matplotlib` (for visualization)

---


##  License

This project is shared under the [MIT License](LICENSE). The `Dataset.csv` is provided for reproducibility of this specific analysis.
