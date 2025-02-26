import json
import jsonlines
from typing import Dict, List, Any
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Field definitions with their types and requirements
FIELD_DEFINITIONS = {
    "Contract Address": {
        "type": "rich_address",
        "is_identifier": True,
        "evaluation": "exact",
    },
    "Public Name Tag": {"type": "text", "is_identifier": True, "evaluation": "ner"},
    "Project Name": {"type": "text", "is_identifier": True, "evaluation": "ner"},
    "UI/Website Link": {"type": "link", "is_identifier": True, "evaluation": "exact"},
    "Public Note": {"type": "text", "is_identifier": False, "evaluation": "semantic"},
}


def load_jsonl(file_path: str) -> List[Dict[str, str]]:
    """Load and parse JSONL file."""
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate string similarity using SequenceMatcher."""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def calculate_cosine_similarity(text1: str, text2: str) -> float:
    """Calculate cosine similarity between two texts using word-level vectorization."""
    # Convert texts to lowercase to make comparison case-insensitive
    text1 = text1.lower()
    text2 = text2.lower()
    
    # Create word-level vectors (instead of character-level)
    vectorizer = CountVectorizer(lowercase=True, token_pattern=r'(?u)\b\w+\b')
    try:
        vectors = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(vectors)[0][1]
        return float(similarity)
    except:
        return 0.0


def evaluate_exact_field(gt: str, pred: str) -> Dict[str, Any]:
    """Evaluate fields requiring exact matches."""
    is_match = gt.lower() == pred.lower()
    return {"exact_match": 1 if is_match else 0, "ground_truth": gt, "prediction": pred}


def evaluate_ner_field(gt: str, pred: str) -> Dict[str, Any]:
    """Evaluate NER fields with some flexibility."""
    similarity = calculate_similarity(gt, pred)
    return {
        "exact_match": 1 if gt.lower() == pred.lower() else 0,
        "near_match": 1 if similarity > 0.9 else 0,
        "similarity": similarity,
        "ground_truth": gt,
        "prediction": pred,
    }


def evaluate_semantic_field(gt: str, pred: str) -> Dict[str, Any]:
    """Evaluate semantic fields using cosine similarity."""
    similarity = calculate_cosine_similarity(gt, pred)
    # Lower the threshold for "meets_threshold" as cosine similarity
    # can be stricter than sequence matching
    return {
        "similarity": similarity,
        "meets_threshold": 1 if similarity >= 0.5 else 0,  # Lowered from 0.85 to 0.5
        "ground_truth": gt,
        "prediction": pred,
    }


def calculate_f1_score(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def evaluate_dataset(
    ground_truth: List[Dict], predictions: List[Dict]
) -> Dict[str, Any]:
    """Evaluate the entire dataset."""
    logging.info(f"Starting evaluation of {len(ground_truth)} entries...")
    results = []
    field_metrics = {
        field: {"exact": 0, "near": 0, "total": 0} for field in FIELD_DEFINITIONS
    }

    for idx, (gt_entry, pred_entry) in enumerate(zip(ground_truth, predictions)):
        if idx % 100 == 0:  # Log progress every 100 entries
            logging.info(f"Processing entry {idx}/{len(ground_truth)}")
        entry_results = {}

        for field, definition in FIELD_DEFINITIONS.items():
            gt_value = gt_entry.get(field, "")
            pred_value = pred_entry.get(field, "")

            if definition["evaluation"] == "exact":
                result = evaluate_exact_field(gt_value, pred_value)
                field_metrics[field]["exact"] += result["exact_match"]
                field_metrics[field]["near"] += result["exact_match"]
            elif definition["evaluation"] == "ner":
                result = evaluate_ner_field(gt_value, pred_value)
                field_metrics[field]["exact"] += result["exact_match"]
                field_metrics[field]["near"] += result["near_match"]
            else:  # semantic
                result = evaluate_semantic_field(gt_value, pred_value)
                field_metrics[field]["near"] += result["meets_threshold"]

            field_metrics[field]["total"] += 1
            entry_results[field] = result

        results.append(entry_results)

    logging.info("Calculating aggregate metrics...")
    # Calculate aggregate metrics
    aggregate_results = {}
    for field, metrics in field_metrics.items():
        definition = FIELD_DEFINITIONS[field]
        if definition["evaluation"] in ["exact", "ner"]:
            precision = (
                metrics["exact"] / metrics["total"] if metrics["total"] > 0 else 0
            )
            recall = metrics["near"] / metrics["total"] if metrics["total"] > 0 else 0
            f1 = calculate_f1_score(precision, recall)

            aggregate_results[field] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "exact_match_rate": metrics["exact"] / metrics["total"],
            }
        else:  # semantic
            aggregate_results[field] = {
                "semantic_similarity_rate": metrics["near"] / metrics["total"]
            }

    return {"individual_results": results, "aggregate_results": aggregate_results}


if __name__ == "__main__":
    logging.info("Starting evaluation script...")

    logging.info("Loading ground truth data...")
    ground_truth = load_jsonl("data/ground-truth/data-set1.jsonl")
    logging.info("Loading prediction data...")
    predictions = load_jsonl("data/predictions/data-set1.jsonl")

    logging.info(f"Evaluating {len(ground_truth)} entries...")
    results = evaluate_dataset(ground_truth, predictions)

    logging.info("Saving results to evaluation_results.json...")
    # Save detailed results
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\nEvaluation Summary:")
    for field, metrics in results["aggregate_results"].items():
        print(f"\n{field}:")
        if "f1_score" in metrics:
            print(f"  F1 Score: {metrics['f1_score']:.2f}")
            print(f"  Precision: {metrics['precision']:.2f}")
            print(f"  Recall: {metrics['recall']:.2f}")
        else:
            print(
                f"  Semantic Similarity Rate: {metrics['semantic_similarity_rate']:.2f}"
            )

    print("\nDetailed results saved to 'evaluation_results.json'")
