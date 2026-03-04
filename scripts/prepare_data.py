"""Download and prepare the text-to-SQL dataset for MLX LoRA fine-tuning.

Uses gretelai/synthetic_text_to_sql from HuggingFace.
Filters to basic SQL and single-join complexity for the 0.8B model.
Outputs completions-format JSONL (prompt/completion) to mask the prompt from loss.
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).parent.parent / "data"
SEED = 42
TRAIN_SIZE = 5000
VALID_SIZE = 500
TEST_SIZE = 1000

# Keep only clean, learnable examples
ALLOWED_COMPLEXITY = {"basic SQL", "single join"}


def format_example(row: dict) -> dict:
    """Format a row into text format for mlx-lm training.

    Single "text" field with schema, question, and SQL answer.
    Works with any model — no chat template dependency.
    """
    schema = row["sql_context"].strip()
    question = row["sql_prompt"].strip()
    sql = row["sql"].strip()

    text = f"{schema}\nQ: {question}\nA: {sql}"
    return {"text": text}


def write_jsonl(examples: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  Wrote {len(examples):,} examples to {path}")


def main():
    print("Downloading gretelai/synthetic_text_to_sql...")
    ds = load_dataset("gretelai/synthetic_text_to_sql", split="train")
    print(f"  Total rows: {len(ds):,}")

    # Filter by complexity
    filtered = ds.filter(
        lambda row: row["sql_complexity"] in ALLOWED_COMPLEXITY,
        desc="Filtering by complexity",
    )
    print(f"  After complexity filter: {len(filtered):,}")

    # Filter out empty/malformed rows
    filtered = filtered.filter(
        lambda row: bool(row["sql_context"]) and bool(row["sql_prompt"]) and bool(row["sql"]),
        desc="Removing empty rows",
    )
    print(f"  After cleaning: {len(filtered):,}")

    # Shuffle and sample
    total_needed = TRAIN_SIZE + VALID_SIZE + TEST_SIZE
    random.seed(SEED)
    indices = list(range(len(filtered)))
    random.shuffle(indices)
    indices = indices[:total_needed]

    selected = filtered.select(indices)
    examples = [format_example(row) for row in selected]

    # Split
    train = examples[:TRAIN_SIZE]
    valid = examples[TRAIN_SIZE : TRAIN_SIZE + VALID_SIZE]
    test = examples[TRAIN_SIZE + VALID_SIZE :]

    print(f"\nSplit sizes: train={len(train)}, valid={len(valid)}, test={len(test)}")

    write_jsonl(train, DATA_DIR / "train.jsonl")
    write_jsonl(valid, DATA_DIR / "valid.jsonl")
    write_jsonl(test, DATA_DIR / "test.jsonl")

    # Show a sample
    print("\nSample example:")
    sample = train[0]
    print(f"  {sample['text'][:200]}...")


if __name__ == "__main__":
    main()
