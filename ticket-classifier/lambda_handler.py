import json
from transformers import pipeline

CATEGORIES = ["engineering", "regulatory", "qa", "general"]

# Load model at module level (runs once per cold start, not per invocation)
print("Loading model from ./model_cache ...")
classifier_pipeline = pipeline(
    "zero-shot-classification",
    model="./model_cache",
)
print("Model loaded.")


def handler(event, context):
    """Lambda handler — receives an API Gateway event."""
    
    # API Gateway sends body as a JSON string
    try:
        body = json.loads(event.get("body", "{}"))
    except json.JSONDecodeError:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Invalid JSON body"})
        }
    
    text = body.get("text", "")
    if not text:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Missing 'text' field"})
        }
    
    # Run classification
    result = classifier_pipeline(
        text,
        candidate_labels=CATEGORIES,
        hypothesis_template="This support ticket is about {}."
    )
    
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "label": result["labels"][0],
            "confidence": round(result["scores"][0], 3),
            "all_scores": dict(zip(result["labels"], [round(s, 3) for s in result["scores"]]))
        })
    }