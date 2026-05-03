import json
import os
from dotenv import load_dotenv
from anthropic import Anthropic
from collections import defaultdict

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

CATEGORIES = ["engineering", "regulatory", "qa", "general"]

# ─────────────────────────────────────────
# THE CLASSIFIER
# This is what we're evaluating.
# Right now: zero-shot Claude Haiku (our baseline)
# Later: we swap this out for our fine-tuned model WER HERE NOW
# ─────────────────────────────────────────
from classifier import classify

# ─────────────────────────────────────────
# THE EVAL RUNNER
# ─────────────────────────────────────────
def run_eval(eval_path: str, run_label: str = "baseline"):
    # load eval set
    examples = []
    with open(eval_path, "r") as f:
        for line in f:
            examples.append(json.loads(line.strip()))

    print(f"\n{'━'*50}")
    print(f"EVAL RESULTS — {run_label}")
    print(f"{'━'*50}")
    print(f"Running {len(examples)} examples...\n")

    # track results
    correct = 0
    total = len(examples)
    
    # per category tracking
    category_correct = defaultdict(int)
    category_total = defaultdict(int)
    
    # confusion matrix: actual → predicted → count
    confusion = defaultdict(lambda: defaultdict(int))
    
    # track failures for inspection
    failures = []

    for i, example in enumerate(examples):
        text = example["text"]
        actual = example["label"]
        predicted = classify(text)
        
        category_total[actual] += 1
        confusion[actual][predicted] += 1
        
        if predicted == actual:
            correct += 1
            category_correct[actual] += 1
        else:
            failures.append({
                "text": text,
                "actual": actual,
                "predicted": predicted
            })
        
        # progress indicator
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{total} done...")

    # ── OVERALL ACCURACY ──
    accuracy = correct / total * 100
    print(f"\nOverall accuracy: {correct}/{total} = {accuracy:.1f}%")

    # ── PER CATEGORY ACCURACY ──
    print(f"\nPer category:")
    for cat in CATEGORIES:
        cat_correct = category_correct[cat]
        cat_total = category_total[cat]
        cat_acc = cat_correct / cat_total * 100 if cat_total > 0 else 0
        bar = "█" * int(cat_acc / 5) + "░" * (20 - int(cat_acc / 5))
        print(f"  {cat:<12} {cat_correct:>2}/{cat_total} = {cat_acc:>5.1f}%  {bar}")

    # ── CONFUSION MATRIX ──
    print(f"\nConfusion matrix (actual → predicted):")
    header = f"{'':>14}" + "".join(f"{c:>14}" for c in CATEGORIES)
    print(header)
    for actual_cat in CATEGORIES:
        row = f"  {actual_cat:<12}"
        for predicted_cat in CATEGORIES:
            count = confusion[actual_cat][predicted_cat]
            # highlight diagonal (correct predictions)
            if actual_cat == predicted_cat:
                row += f"  [{count:>2}]        "
            else:
                row += f"    {count:>2}        "
        print(row)

    # ── WORST FAILURES ──
    print(f"\nSample failures (first 5):")
    for f in failures[:5]:
        print(f"  actual={f['actual']:<12} predicted={f['predicted']:<12}")
        print(f"  text: {f['text'][:80]}...")
        print()

    # ── SAVE RESULTS ──
    result = {
        "run_label": run_label,
        "overall_accuracy": round(accuracy, 2),
        "per_category": {
            cat: round(category_correct[cat] / category_total[cat] * 100, 2)
            for cat in CATEGORIES
        },
        "total_correct": correct,
        "total_examples": total,
        "failure_count": len(failures)
    }
    
    results_file = f"eval_results_{run_label.replace(' ', '_')}.json"
    with open(results_file, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"{'━'*50}")
    print(f"Results saved to {results_file}")
    
    return result


if __name__ == "__main__":
    run_eval("eval_set.jsonl", run_label="baseline")