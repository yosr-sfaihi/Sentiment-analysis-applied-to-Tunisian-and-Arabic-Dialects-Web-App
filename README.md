# Sentiment Analysis of Tunisian Dialect Comments

Three-class sentiment classification (**negative / neutre / positive**) of user comments written in Tunisian Arabic
about the Tunisian economic crisis. The project compares classical machine learning, hybrid CNN + recurrent
networks and a fine-tuned transformer ([AraBERT](https://huggingface.co/aubmindlab/bert-base-arabertv02)), all
evaluated on the same held-out test set.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/REPO/blob/main/Sentiment_Analysis_Tunisian_Dialect.ipynb)

## Demo

<!-- Replace VIDEO_ID with your YouTube video id (the part after v= in the URL). -->
[![Watch the demo](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)

_Click the image to watch the demo video._

## Table of contents

- [Motivation](#motivation)
- [Approach](#approach)
- [Models compared](#models-compared)
- [Dataset](#dataset)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Configuration](#configuration)
- [Results](#results)
- [Limitations](#limitations)
- [Ethics and data protection](#ethics-and-data-protection)
- [Roadmap](#roadmap)
- [License](#license)

## Motivation

Tunisian dialect (Darija) is a low-resource, non-standardised variety of Arabic: spelling varies from one writer to
the next, and comments often mix Arabic script, French and Latin-letter Arabic ("Arabizi"). Standard Arabic NLP
tools and models are trained mostly on Modern Standard Arabic, so it is not obvious how well they transfer. This
project measures that on a real set of comments about the economic crisis in Tunisia.

## Approach

| Step | What happens |
|---|---|
| 1. Load and clean labels | Read the Excel file, drop missing rows, normalise labels (lower-case, trimmed), drop rows labelled `lang` or `vide` |
| 2. Clean text | Keep Arabic-script characters only, drop comments that become empty and duplicate texts |
| 3. Split once | Stratified 80/20 train/test split with a fixed seed |
| 4. Features | TF-IDF (1000 features) and Word2Vec (100-d mean-pooled, standardised) |
| 5. Train | Classical models, hybrid CNN + RNN models, fine-tuned AraBERT |
| 6. Compare | One table ranked by macro-F1, next to a majority-class baseline, plus the confusion matrix of the best model |

**No data leakage:** the split is done before any feature extraction. The TF-IDF vectoriser, the Word2Vec model and
the feature scaler are fitted on the training set only, and the test set is used only for the final evaluation.
Early stopping in the Keras models uses a validation split taken from the training set.

## Models compared

| Family | Models | Features |
|---|---|---|
| Baseline | Majority class | none |
| Classical ML | Naive Bayes, SVM (linear, RBF, sigmoid, polynomial), MLP, Random Forest, Logistic Regression | TF-IDF, Word2Vec |
| Deep learning | CNN + LSTM, CNN + Bi-LSTM, CNN + GRU | TF-IDF, Word2Vec |
| Transformer | AraBERT (`aubmindlab/bert-base-arabertv02`), fine-tuned for 3 epochs | raw cleaned text |

## Dataset

The dataset is **not included** in this repository (see [Ethics and data protection](#ethics-and-data-protection)).

The notebook expects an Excel file with two columns:

| Column | Description |
|---|---|
| `comment` | Raw comment text |
| `label` | `positive`, `negative` or `neutre`. Rows labelled `lang` (comment in another language) or `vide` (empty) are discarded |

- **Source:** _TODO: describe where the comments were collected, when, and under which terms._
- **Annotation:** _TODO: who labelled the comments and with which guidelines._
- **Size and balance:** about 4,600 labelled comments after filtering, with a strong majority of negative comments
  (roughly 60 %). The exact counts are printed by the notebook.

Place your file at `data/crise economique (labeled).xlsx`, or point to it with the `DATA_PATH` environment variable.

## Project structure

```
.
├── Sentiment_Analysis_Tunisian_Dialect.ipynb   # full pipeline, from data loading to model comparison
├── requirements.txt                            # Python dependencies
├── .gitignore                                  # keeps data and trained models out of the repository
├── README.md
└── data/                                       # not tracked: put the dataset here
```

## Getting started

### Requirements

- Python 3.10 or later
- A GPU is recommended for the AraBERT section (Google Colab's free GPU is enough)

### Installation

```bash
git clone https://github.com/USERNAME/REPO.git
cd REPO

python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (Git Bash)
source .venv/Scripts/activate

pip install -r requirements.txt
```

### Run

```bash
# optional: custom dataset location
export DATA_PATH="path/to/your/file.xlsx"

jupyter lab
```

Open `Sentiment_Analysis_Tunisian_Dialect.ipynb` and run all cells. Everything except AraBERT runs on CPU.

### Run on Google Colab

1. Click the **Open in Colab** badge above and select a GPU runtime (Runtime > Change runtime type).
2. Upload the dataset to the Colab file browser (for example to `data/`).
3. In the first code cell, set `DATA_PATH` to the uploaded file and run all cells. Colab already provides most
   of the dependencies; install any missing one with `pip install gensim`.

## Configuration

All options are in the **Setup** cell of the notebook:

| Option | Default | Effect |
|---|---|---|
| `SEED` | `42` | Random seed used for the split, Word2Vec, Keras and PyTorch |
| `DATA_PATH` | `data/crise economique (labeled).xlsx` | Location of the dataset (can also be set as an environment variable) |
| `TEST_SIZE` | `0.2` | Fraction of the data used for testing |
| `RUN_KERAS` | `True` | Train the CNN + LSTM / Bi-LSTM / GRU models |
| `RUN_ARABERT` | `True` | Fine-tune AraBERT (set to `False` without a GPU) |

## Results

> **TODO:** after running the notebook on the real dataset, paste the top rows of the final comparison table here.
> Keep the majority-class baseline in the table: a model that does not clearly beat it has not learned much beyond
> the class imbalance.

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Majority-class baseline | | |
| _best classical model_ | | |
| _best CNN + RNN model_ | | |
| AraBERT (fine-tuned) | | |

Macro-F1 is the main metric because accuracy is misleading on imbalanced classes.

## Limitations

- **Class imbalance.** Negative comments dominate, so accuracy alone overstates quality.
- **Arabic-only cleaning.** Everything outside the Arabic script (Arabizi, French, digits, emojis) is removed, which
  discards real signal in code-switched comments.
- **Single split and no hyper-parameter search.** Differences of a point or two between models are within noise.
- **CNN + RNN on bag-of-words vectors.** The TF-IDF and Word2Vec vectors are not true sequences, so these hybrids are
  an experimental baseline rather than a conventional sequence model.
- **Domain specificity.** The comments are about one topic (the economic crisis) and one period. The models are
  unlikely to transfer unchanged to other domains.

## Ethics and data protection

- The data consists of comments written by third parties on a public platform. Do not redistribute it unless the
  platform's terms allow it and the text has been anonymised.
- The dataset and any trained model files are excluded from the repository through `.gitignore`.
- Sentiment predictions on individual comments can be wrong and can reflect annotation bias. This project is for
  research and education; it should not be used to make decisions about individuals.

## Roadmap

- [ ] Class weights or resampling to handle the imbalance
- [ ] Keep Latin-script and French text instead of removing it
- [ ] Cross-validation and a small hyper-parameter search
- [ ] Arabic-specific preprocessing for AraBERT
- [ ] Models pre-trained on dialectal Arabic and a comparison with them
- [ ] A small demo app (Streamlit or Gradio) to classify a new comment

## License

_TODO: choose a license (for example MIT for the code) and add a `LICENSE` file. The license of the code does not
cover the dataset._

## Acknowledgements

- [AraBERT](https://github.com/aub-mind/arabert) by the AUB MIND Lab
- [scikit-learn](https://scikit-learn.org), [gensim](https://radimrehurek.com/gensim/), [Keras](https://keras.io),
  [PyTorch](https://pytorch.org) and [Hugging Face Transformers](https://huggingface.co/docs/transformers)
