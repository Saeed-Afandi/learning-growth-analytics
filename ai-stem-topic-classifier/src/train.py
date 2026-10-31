from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

CATEGORIES = [
    "comp.graphics",
    "sci.space",
    "sci.med",
    "sci.electronics",
]

DISPLAY_NAMES = {
    "comp.graphics": "Computer Graphics",
    "sci.space": "Space",
    "sci.med": "Medicine",
    "sci.electronics": "Electronics",
}


def load_data():
    train_data = fetch_20newsgroups(
        subset="train",
        categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )
    test_data = fetch_20newsgroups(
        subset="test",
        categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )
    return train_data, test_data


def build_pipeline(model):
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
            ),
        ),
        ("model", model),
    ])


def save_top_terms(pipeline, target_names, top_n=15):
    model = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        return

    vectorizer = pipeline.named_steps["tfidf"]
    feature_names = vectorizer.get_feature_names_out()

    rows = []
    for class_index, class_name in enumerate(target_names):
        weights = model.coef_[class_index]
        top_indices = weights.argsort()[-top_n:][::-1]
        for rank, idx in enumerate(top_indices, start=1):
            rows.append({
                "category": DISPLAY_NAMES.get(class_name, class_name),
                "rank": rank,
                "term": feature_names[idx],
                "weight": float(weights[idx]),
            })

    pd.DataFrame(rows).to_csv(ARTIFACTS / "top_terms.csv", index=False)


def main():
    train_data, test_data = load_data()

    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.2),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Linear SVC": LinearSVC(random_state=42),
    }

    results = []
    trained = {}

    for name, model in models.items():
        pipeline = build_pipeline(model)
        pipeline.fit(train_data.data, train_data.target)
        predictions = pipeline.predict(test_data.data)
        accuracy = accuracy_score(test_data.target, predictions)

        results.append({"model": name, "accuracy": accuracy})
        trained[name] = (pipeline, predictions)

        print("\n" + "=" * 70)
        print(name)
        print(f"Accuracy: {accuracy:.4f}")
        print(
            classification_report(
                test_data.target,
                predictions,
                target_names=[DISPLAY_NAMES[name] for name in test_data.target_names],
                zero_division=0,
            )
        )

    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    results_df.to_csv(ARTIFACTS / "model_results.csv", index=False)

    best_name = results_df.iloc[0]["model"]
    best_pipeline, best_predictions = trained[best_name]

    joblib.dump(
        {
            "pipeline": best_pipeline,
            "target_names": test_data.target_names,
            "display_names": DISPLAY_NAMES,
            "model_name": best_name,
        },
        ARTIFACTS / "best_model.joblib",
    )

    ConfusionMatrixDisplay.from_predictions(
        test_data.target,
        best_predictions,
        display_labels=[DISPLAY_NAMES[name] for name in test_data.target_names],
        xticks_rotation=30,
        cmap="Blues",
    )
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "confusion_matrix.png", dpi=180)
    plt.close()

    save_top_terms(best_pipeline, test_data.target_names)

    print("\nBEST MODEL")
    print("=" * 70)
    print(best_name)
    print(f"Accuracy: {results_df.iloc[0]['accuracy']:.4f}")
    print("\nSaved artifacts to:", ARTIFACTS)


if __name__ == "__main__":
    main()
