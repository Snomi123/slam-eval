from __future__ import annotations

import json
from typing import Any

from slam_eval.scorer import Score, Scorer


def normalize_value(val: Any) -> str:
    if val is None:
        return ""
    return str(val).strip().lower()


def extract_leaf_values(obj: Any, path: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            if isinstance(value, (dict, list)):
                result.update(extract_leaf_values(value, new_path))
            else:
                result[new_path] = value
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            new_path = f"{path}[{idx}]"
            result.update(extract_leaf_values(item, new_path))

    return result


def get_structure(obj: Any, path: str = "") -> dict[str, str]:
    structure: dict[str, str] = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key

            if isinstance(value, dict):
                structure[new_path] = "object"
                structure.update(get_structure(value, new_path))
            elif isinstance(value, list):
                structure[new_path] = "array"
                for idx, item in enumerate(value):
                    item_path = f"{new_path}[{idx}]"
                    structure.update(get_structure(item, item_path))
            else:
                structure[new_path] = type(value).__name__

    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            item_path = f"{path}[{idx}]"
            structure.update(get_structure(item, item_path))

    return structure


def safe_parse_prediction(y_pred: Any) -> dict[str, Any]:
    if isinstance(y_pred, dict):
        return y_pred

    if isinstance(y_pred, str):
        try:
            parsed = json.loads(y_pred)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}

    return {}


def calculate_metrics(
    ground_truth: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, float]:
    gt_fields = extract_leaf_values(ground_truth)
    pred_fields = extract_leaf_values(prediction)

    matching_fields = 0
    for key in gt_fields:
        if key in pred_fields and normalize_value(gt_fields[key]) == normalize_value(
            pred_fields[key]
        ):
            matching_fields += 1

    precision = matching_fields / len(pred_fields) if pred_fields else 1.0
    recall = matching_fields / len(gt_fields) if gt_fields else 1.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    gt_struct = get_structure(ground_truth)
    pred_struct = get_structure(prediction)

    gt_struct_set = set(gt_struct.items())
    pred_struct_set = set(pred_struct.items())

    intersection = len(gt_struct_set & pred_struct_set)
    union = len(gt_struct_set | pred_struct_set)
    structure_similarity = intersection / union if union > 0 else 1.0

    gt_keys = set(gt_fields.keys())
    pred_keys = set(pred_fields.keys())

    completeness = len(gt_keys & pred_keys) / len(gt_keys) if gt_keys else 1.0
    hallucination = len(pred_keys - gt_keys) / len(pred_keys) if pred_keys else 0.0

    hallucination_penalty = max(0.0, 1.0 - hallucination)
    total_score = (
        0.4 * f1
        + 0.2 * structure_similarity
        + 0.25 * completeness
        + 0.15 * hallucination_penalty
    )

    return {
        "total_score": total_score,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "completeness": completeness,
        "hallucination": hallucination,
        "structure_similarity": structure_similarity,
    }


class MergeQualityScorer(Scorer):
    def __call__(self, y_true: Any, y_pred: Any) -> Score:
        if not isinstance(y_true, dict):
            raise TypeError("MergeQualityScorer expects y_true to be a dict.")

        parsed_prediction = safe_parse_prediction(y_pred)
        metrics = calculate_metrics(y_true, parsed_prediction)

        return Score(
            primary=metrics["total_score"],
            sub_scores={
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "completeness": metrics["completeness"],
                "hallucination": metrics["hallucination"],
                "structure_similarity": metrics["structure_similarity"],
            },
        )
