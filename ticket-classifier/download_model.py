from transformers import pipeline

print("Downloading model to ./model_cache ...")

pipe = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli",
)

# save it to a local folder
pipe.save_pretrained("./model_cache")

print("✓ Model saved to ./model_cache")