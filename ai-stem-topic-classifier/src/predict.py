from pathlib import Path
import sys
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "best_model.joblib"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run 'python3 src/train.py' first."
        )

    if len(sys.argv) < 2:
        text = input("Enter STEM text to classify: ").strip()
    else:
        text = " ".join(sys.argv[1:]).strip()

    if not text:
        raise ValueError("Please provide some text.")

    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    target_names = bundle["target_names"]
    display_names = bundle["display_names"]

    prediction = int(pipeline.predict([text])[0])
    category_key = target_names[prediction]
    category = display_names.get(category_key, category_key)

    print("\nAI STEM TOPIC CLASSIFIER")
    print("=" * 48)
    print("Input:", text)
    print("Predicted topic:", category)
    print("Model:", bundle["model_name"])

    model = pipeline.named_steps["model"]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(
            pipeline.named_steps["tfidf"].transform([text])
        )[0]
        ranked = sorted(
            zip(target_names, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        print("\nDecision scores:")
        for name, score in ranked:
            print(f"  {display_names.get(name, name):20s} {score: .3f}")


if __name__ == "__main__":
    main()
