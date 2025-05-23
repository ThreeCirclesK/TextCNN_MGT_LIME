
```markdown
# Bug Localization with DualTextCNN

This project implements a deep learning-based bug localization system using a Dual TextCNN model. It takes natural language bug reports and ranks source files based on their likelihood of being related to the bug.

---

## 🔍 Project Overview

The system is designed to support tasks such as:

- Training a dual-path CNN classifier
- Evaluating ranked file predictions using IR metrics (Acc@1, Acc@5, MAP, MRR)

It is built using PyTorch and supports fully configurable training and evaluation via a command-line interface.

---

## 📁 Project Structure

```
TEXTCNN_BUGLOCALIZATION/
│
├── artifacts/
│   ├── glove6B/                            # Pretrained GloVe embeddings
│   ├── test_bugs/                          # Bug report pickle files for testing
│   │   └── sphinx-doc+sphinx_bugs.pkl
│   └── trained_model/                      # Saved model checkpoints
│       └── best_model2.pt
│
├── data/
│   ├── data_builder.py
│   ├── glove_loader.py
│   ├── paired_text_dataset.py
│   └── datasets/
│       └── past_brid2commit/
│           ├── sphinx-doc+sphinx_bugs_brid2commit.pkl
│           └── ...                         # other repos' mappings
│
├── evaluation/
│   └── ranker.py                           # Ranking logic and IR metric 
│
├── models/
│   └── dual_text_cnn.py                    # DualTextCNN model
│
├── training/
│   └── trainer.py                          # Trainer class for training and validation
│
├── main.py                                 # CLI entry point
└── README.md

````

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
````

### 2. Prepare Artifacts and Dataset

Place the following in the `artifact/` folder:

* `glove6B/` — Pretrained GloVe embeddings (link: https://nlp.stanford.edu/projects/glove/)
* `lime_results/` — Collected lime results
* `test_bugs/` — Test bug pickle files
* `trained_model/` — Trained bug localization model

Place the following in the `datasets/` folder:

* `SWEBench_Custom/` — Indexed files and pickle files for bug localization. Unzip to use them

---

## 🏋️ Training and Evaluation

### 1. Train and Evaluate:

```bash
python train.py \
  --repodir /path/to/SWEBench_Custom \
  --reponame sphinx-doc+sphinx \
  --model_name artifacts/trained_model/best_model.pt
```
ex) python train.py --repodir datasets/SWEBench_Custom --reponame sphinx-doc+sphinx --model_name artifacts/trained_model/best_model.pt

### 2. Collect Impactful Words:

```bash
python collect_impactful_words.py --repodir datasets/SWEBench_Custom --reponame sphinx-doc+sphinx --model_dir artifacts/trained_model/best_model.pt
```

### 3. Apply Lime:
```bash
python remove_mgt.py --reponame sphinx-doc+sphinx
```

---

## 📊 Metrics Reported

The evaluation script computes:

* **Accuracy\@1**
* **Accuracy\@5**
* **Mean Reciprocal Rank (MRR)**
* **Mean Average Precision (MAP)**

These metrics reflect the model’s ability to rank fixed files higher among candidates.

---

## 🧠 Model

The `DualTextCNN` processes bug reports and source files in parallel using separate convolutional pathways. It applies max-pooling and concatenates both outputs pass to FCN layers for classification.

---


## 📄 License

This project is distributed under the MIT License.

```

