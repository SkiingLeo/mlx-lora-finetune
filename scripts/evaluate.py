"""Evaluate text-to-SQL model: base model vs fine-tuned (with LoRA adapter).

Metrics:
  - Exact match %: normalized SQL string comparison
  - SQL rate %: output starts with a SQL keyword (SELECT, INSERT, etc.)
  - Per-example verdicts: EXACT / SQL / WRONG with explanation

Usage:
  uv run python scripts/evaluate.py                   # evaluate fine-tuned model
  uv run python scripts/evaluate.py --baseline         # evaluate base model only
  uv run python scripts/evaluate.py --num-samples 200  # custom sample size
  uv run python scripts/evaluate.py --adapter-path adapters-run1  # custom adapter
"""

import argparse
import json
import re
import time
from pathlib import Path

from mlx_lm import generate, load

DEFAULT_MODEL = "mlx-community/Qwen3.5-0.8B-4bit-OptiQ"
ADAPTER_PATH = "adapters"
TEST_DATA = Path(__file__).parent.parent / "data" / "test.jsonl"
RESULTS_DIR = Path(__file__).parent.parent / "results"
DEFAULT_SAMPLES = 200

SQL_KEYWORDS = {"select", "insert", "update", "delete", "create", "alter", "drop", "with"}


def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison: lowercase, collapse whitespace, strip."""
    sql = sql.strip().rstrip(";").strip()
    sql = sql.lower()
    sql = re.sub(r"\s+", " ", sql)
    return sql


def is_valid_sql(text: str) -> bool:
    """Check if output looks like SQL (starts with a SQL keyword)."""
    first_word = text.strip().split()[0].lower() if text.strip() else ""
    return first_word in SQL_KEYWORDS


def load_test_data(path: Path, n: int) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
            if len(examples) >= n:
                break
    return examples


def split_text_example(text: str) -> tuple[str, str]:
    """Split a text-format example into (prompt, expected_sql)."""
    parts = text.rsplit("\nA: ", 1)
    prompt = parts[0] + "\nA: "
    sql = parts[1] if len(parts) > 1 else ""
    return prompt, sql


def evaluate_model(model, tokenizer, examples: list[dict], label: str) -> dict:
    """Run generation on examples and compute metrics."""
    exact_count = 0
    sql_count = 0
    total = len(examples)
    all_outputs = []

    print(f"\n{'='*60}")
    print(f"Evaluating: {label} ({total} examples)")
    print(f"{'='*60}")

    start = time.time()

    for i, ex in enumerate(examples):
        prompt, ref_sql = split_text_example(ex["text"])

        generated = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=256,
        )

        gen_sql = generated.strip().split("\n")[0].strip()

        exact = normalize_sql(gen_sql) == normalize_sql(ref_sql)
        valid_sql = is_valid_sql(gen_sql)

        if exact:
            exact_count += 1
            verdict = "EXACT"
        elif valid_sql:
            sql_count += 1
            verdict = "SQL"
        else:
            verdict = "WRONG"

        all_outputs.append({
            "expected": ref_sql,
            "generated": gen_sql,
            "verdict": verdict,
        })

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{total} | exact={exact_count}/{i+1} | sql={exact_count+sql_count}/{i+1}")

    elapsed = time.time() - start

    total_sql = exact_count + sql_count
    exact_pct = exact_count / total * 100
    sql_pct = total_sql / total * 100
    wrong_count = total - total_sql

    print(f"\n  EXACT: {exact_count:>4} ({exact_pct:.1f}%)  — character-identical SQL")
    print(f"  SQL:   {sql_count:>4} ({sql_count/total*100:.1f}%)  — valid SQL, not identical")
    print(f"  WRONG: {wrong_count:>4} ({wrong_count/total*100:.1f}%)  — not SQL")
    print(f"\n  Exact match:   {exact_pct:.1f}%")
    print(f"  Generates SQL: {sql_pct:.1f}%")
    print(f"  Time: {elapsed:.1f}s ({elapsed/total:.2f}s per example)")

    return {
        "label": label,
        "exact_match_pct": round(exact_pct, 2),
        "sql_rate_pct": round(sql_pct, 2),
        "exact_count": exact_count,
        "sql_count": total_sql,
        "wrong_count": wrong_count,
        "total": total,
        "time_seconds": round(elapsed, 1),
        "outputs": all_outputs,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate text-to-SQL models")
    parser.add_argument("--baseline", action="store_true", help="Evaluate base model only (no adapter)")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_SAMPLES, help=f"Number of test examples (default: {DEFAULT_SAMPLES})")
    parser.add_argument("--adapter-path", type=str, default=ADAPTER_PATH, help=f"Path to adapter (default: {ADAPTER_PATH})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Base model path (default: {DEFAULT_MODEL})")
    parser.add_argument("--no-baseline", action="store_true", help="Skip baseline evaluation (faster)")
    args = parser.parse_args()

    model_path = args.model
    examples = load_test_data(TEST_DATA, args.num_samples)
    print(f"Loaded {len(examples)} test examples")

    results = {}

    if args.baseline:
        print(f"\nLoading base model: {model_path}")
        model, tokenizer = load(model_path)
        results["baseline"] = evaluate_model(model, tokenizer, examples, "Base Model (no fine-tune)")
    else:
        adapter = Path(args.adapter_path)
        if not adapter.exists():
            print(f"ERROR: Adapter not found at {args.adapter_path}")
            print("Run training first, or use --baseline to evaluate the base model.")
            return

        print(f"\nLoading fine-tuned model: {model_path} + {args.adapter_path}")
        model, tokenizer = load(model_path, adapter_path=args.adapter_path)
        results["finetuned"] = evaluate_model(model, tokenizer, examples, f"Fine-tuned ({args.adapter_path})")

        if not args.no_baseline:
            del model
            print(f"\nLoading base model for comparison: {model_path}")
            model, tokenizer = load(model_path)
            results["baseline"] = evaluate_model(model, tokenizer, examples, "Base Model (no fine-tune)")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<35} {'Exact':>7} {'SQL Rate':>10}")
    print(f"  {'-'*35} {'-'*7} {'-'*10}")
    for key, res in results.items():
        print(f"  {res['label']:<35} {res['exact_match_pct']:>6}% {res['sql_rate_pct']:>9}%")

    if "baseline" in results and "finetuned" in results:
        f, b = results["finetuned"], results["baseline"]
        print(f"\n  Exact match:  +{f['exact_match_pct'] - b['exact_match_pct']:.1f}pp")
        print(f"  SQL rate:     +{f['sql_rate_pct'] - b['sql_rate_pct']:.1f}pp")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save summary (without full outputs for readability)
    summary = {}
    for key, res in results.items():
        summary[key] = {k: v for k, v in res.items() if k != "outputs"}
        summary[key]["sample_outputs"] = res["outputs"][:10]

    out_path = RESULTS_DIR / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Save ALL outputs for detailed review
    for key, res in results.items():
        all_path = RESULTS_DIR / f"outputs_{key}.jsonl"
        with open(all_path, "w") as f:
            for o in res["outputs"]:
                f.write(json.dumps(o) + "\n")
        print(f"All outputs saved to {all_path}")

    # Print samples
    for key, res in results.items():
        print(f"\n--- {res['label']} ---")
        for o in res["outputs"][:5]:
            print(f"  [{o['verdict']:>5}] Expected: {o['expected'][:70]}")
            print(f"          Got:      {o['generated'][:70]}")
            print()


if __name__ == "__main__":
    main()
