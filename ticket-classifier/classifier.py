from transformers import pipeline
import torch

# use MPS (Apple Silicon GPU) if available, otherwise CPU
device = 0 if torch.backends.mps.is_available() else -1

print("Loading model...")
classifier_pipeline = pipeline(
    "zero-shot-classification",
    model="cross-encoder/nli-MiniLM2-L6-H768",  # tiny 66MB model
    device=device
)
print("Model loaded.")

CATEGORIES = ["engineering", "regulatory", "qa", "general"]

def classify(text: str) -> str:
    result = classifier_pipeline(
        text,
        candidate_labels=CATEGORIES,
        hypothesis_template="This ticket is about {}."
    )
    # returns labels sorted by confidence — take the top one
    return result["labels"][0]


if __name__ == "__main__":
    # quick sanity check
    tests = [
        "The build pipeline is failing on the docker step",
        "We need to update the 510k submission",
        "Test case 47 is failing on the cardiac monitor",
        "When is the team offsite?"
    ]
    
    for test in tests:
        label = classify(test)
        print(f"{label:<12} ← {test}")