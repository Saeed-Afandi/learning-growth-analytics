# Explainable AI STEM Topic Classifier

A natural-language processing project that trains and compares machine-learning models to classify technical text into STEM topics.

## Why this project

This project explores a core AI task: **text classification**. The goal is to train a model that can read an unseen piece of text and predict which STEM topic it belongs to.

The project uses four categories from the public 20 Newsgroups dataset:

- Computer Graphics
- Space
- Medicine
- Electronics

To make the system more transparent, the project also extracts the words that most strongly influence each category.

## AI concepts used

- Natural Language Processing (NLP)
- TF-IDF text representation
- Supervised machine learning
- Multiclass classification
- Model comparison
- Accuracy and classification metrics
- Confusion matrix
- Explainable AI through feature weights

## Models compared

1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear Support Vector Classifier (LinearSVC)

The training script evaluates all three models on a held-out test set and saves the best-performing model.

## Project structure

```text
ai-stem-topic-classifier/
├── src/
│   ├── train.py
│   └── predict.py
├── sample_inputs.txt
├── requirements.txt
└── README.md
```

## How it works

The text is converted into numerical features using TF-IDF. Each model learns patterns that distinguish the four STEM categories. The best model is saved and can later classify new text entered by the user.

The training script also reports the most influential terms for each category when the selected model supports linear feature weights.

## Run the project

Create a virtual environment and install the requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train and evaluate the models:

```bash
python3 src/train.py
```

Then classify new text:

```bash
python3 src/predict.py "NASA plans a new telescope mission to study distant galaxies."
```

## Dataset

The project uses the 20 Newsgroups dataset available through scikit-learn. The training script downloads the dataset automatically the first time it is run.

Headers, footers, and quoted replies are removed to reduce easy source-specific clues and make the classification task focus more on text content.

## Outputs

After training, the project creates:

- `artifacts/best_model.joblib`
- `artifacts/model_results.csv`
- `artifacts/confusion_matrix.png`
- `artifacts/top_terms.csv`

## Limitations

The classifier learns from historical discussion-group text, so it may not generalize equally well to every kind of modern STEM writing. Accuracy alone does not prove that a model understands the underlying scientific concepts.

## Next ideas

Possible extensions include adding more STEM categories, testing transformer embeddings, building a small web interface, or comparing classical machine learning with a neural-language model approach.
