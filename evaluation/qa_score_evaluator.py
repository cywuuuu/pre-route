#!/usr/bin/env python3
"""
QA Score Evaluator using Judge Model

Evaluates the quality of predictions by comparing them with ground truth
using a judge model (GPT-4o or similar).
"""

import argparse
import os
import json
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_utils import load_jsonl, call_model


JUDGE_PROMPT = """You are an expert judge for evaluating the quality of answers to questions.

Given a question, a prediction (model's answer), and a ground truth (reference answer), your task is to score the prediction on a scale of 0-4 based on how well it matches the ground truth in terms of:

1. **Factual Accuracy**: Does the prediction contain the correct information?
2. **Completeness**: Does the prediction cover all the key points from the ground truth?
3. **Conciseness**: Is the prediction appropriately concise without unnecessary fluff?

Scoring Guide:
- **4 points**: Excellent - Fully accurate, complete, and concise (matches ground truth perfectly)
- **3 points**: Good - Accurate and mostly complete, with minor omissions or extra details
- **2 points**: Fair - Partially accurate but missing key information or contains some errors
- **1 point**: Poor - Contains relevant information but is mostly incorrect or incomplete
- **0 points**: Very Poor - Completely incorrect, irrelevant, or nonsensical

Output Format:
Respond with a single integer score (0-4) followed by a brief justification.

Example:
Score: 3
Justification: The answer correctly identifies the main advantage but misses the scalability aspect mentioned in the ground truth.

---

Question: {question}

Prediction: {prediction}

Ground Truth: {ground_truth}

Provide your evaluation:"""


def evaluate_qa_score(question: str, prediction: str, ground_truth: str,
                     judge_model: str = "gpt-4o") -> Tuple[int, str]:
    """
    Evaluate a single prediction using a judge model.

    Returns:
        (score, justification)
    """
    prompt = JUDGE_PROMPT.format(
        question=question,
        prediction=prediction,
        ground_truth=ground_truth
    )

    messages = [
        {"role": "system", "content": "You are an expert judge for evaluating answer quality."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = call_model(messages, judge_model, thinking=False)
        response_text = response[1] if isinstance(response, tuple) and len(response) >= 2 else str(response)

        # Parse score from response
        score = 0
        justification = response_text

        for line in response_text.split('\n'):
            line_lower = line.lower()
            if 'score:' in line_lower or '评分:' in line_lower:
                # Extract the number after "Score:"
                for char in line.split(':')[1]:
                    if char.isdigit():
                        score = int(char)
                        break
                break

        return score, justification

    except Exception as e:
        print(f"Error evaluating: {e}")
        return 0, f"Error: {e}"


def evaluate_results(result_path: str, judge_model: str = "gpt-4o",
                   sample_size: int = None) -> Dict[str, Any]:
    """
    Evaluate all predictions in a result file.

    Args:
        result_path: Path to the result JSONL file
        judge_model: Model to use for judging
        sample_size: If set, only evaluate this many samples (for testing)

    Returns:
        Statistics dictionary
    """
    results = load_jsonl(result_path)

    if sample_size:
        results = results[:sample_size]

    print(f"Evaluating {len(results)} predictions with {judge_model}...")

    scores = []
    route_scores = {"no_retrieval": [], "single_step": [], "multi_step": []}

    for i, r in enumerate(tqdm(results)):
        score, justification = evaluate_qa_score(
            r["question"],
            r["answer"],
            r.get("ground_truth", ""),
            judge_model
        )

        scores.append(score)
        route = r.get("selected_route", "unknown")
        if route in route_scores:
            route_scores[route].append(score)

        # Print sample evaluations
        if i < 3:
            print(f"\n--- Sample {i+1} (Route: {route}, Score: {score}) ---")
            print(f"Question: {r['question'][:100]}...")
            print(f"Prediction: {r['answer'][:150]}...")
            print(f"Justification: {justification[:200]}...")

    # Compute statistics
    stats = {
        "total_evaluated": len(scores),
        "mean_score": sum(scores) / len(scores) if scores else 0,
        "route_means": {},
        "score_distribution": {i: scores.count(i) for i in range(5)}
    }

    for route, route_score_list in route_scores.items():
        if route_score_list:
            stats["route_means"][route] = sum(route_score_list) / len(route_score_list)

    return stats


def print_statistics(stats: Dict[str, Any]):
    """Print evaluation statistics"""
    print("\n" + "=" * 60)
    print("QA Score Evaluation Results")
    print("=" * 60)

    print(f"\nTotal Evaluated: {stats['total_evaluated']}")
    print(f"Overall Mean Score: {stats['mean_score']:.2f} / 4.0")

    print(f"\n--- Score Distribution ---")
    for score, count in stats['score_distribution'].items():
        pct = (count / stats['total_evaluated'] * 100) if stats['total_evaluated'] > 0 else 0
        print(f"  {score}/4: {count:3d} ({pct:5.1f}%)")

    if stats['route_means']:
        print(f"\n--- Mean Score by Route ---")
        for route, mean_score in stats['route_means'].items():
            print(f"  {route}: {mean_score:.2f} / 4.0")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate QA Scores")
    parser.add_argument('--result_file', type=str, required=True,
                        help='Path to result JSONL file')
    parser.add_argument('--judge_model', type=str, default="gpt-4o",
                        help='Judge model to use')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Number of samples to evaluate (for testing)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for evaluation results')

    args = parser.parse_args()

    if not os.path.exists(args.result_file):
        print(f"Error: Result file not found: {args.result_file}")
        return

    # Run evaluation
    stats = evaluate_results(args.result_file, args.judge_model, args.sample_size)

    # Print statistics
    print_statistics(stats)

    # Save results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
