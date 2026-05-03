from transformers import pipeline
import torch

device = 0 if torch.backends.mps.is_available() else -1

CATEGORIES = ["engineering", "regulatory", "qa", "general"]

print("Loading model...")
classifier_pipeline = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli",
    device=device
)
print("Model loaded.")

def classify(text: str) -> str:
    result = classifier_pipeline(
        text,
        candidate_labels=CATEGORIES,
        hypothesis_template="This support ticket is about {}."
    )
    return result["labels"][0]


if __name__ == "__main__":
    tests = [
        ("The build pipeline is failing on the docker step", "engineering"),
        ("We need to update the 510k submission", "regulatory"),
        ("Test case 47 is failing on the cardiac monitor", "qa"),
        ("When is the team offsite?", "general")
    ]
    
    for text, expected in tests:
        predicted = classify(text)
        status = "✓" if predicted == expected else "✗"
        print(f"{status} predicted={predicted:<12} expected={expected:<12} ← {text}")