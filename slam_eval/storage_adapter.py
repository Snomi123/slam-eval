import json
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from slam_eval.collections.base import EvalCaseCollection
from slam_eval.model import Model
from slam_eval.scorer import Score
from slam_eval.utils.typing import HasStr


class EvalStorageAdapter(ABC):
    def __init__(self) -> None:
        pass

    def save(
        self,
        group_id: str,
        model: Model,
        eval_case_collection: EvalCaseCollection,
        scores: list[Score],
        model_answers: list[HasStr],
        **other_results,
    ) -> None:
        datetime_now = datetime.now()
        result_id = (
            f"eval:{group_id}:{datetime_now.isoformat('_')}_M_"
            f"{model.name}_C_{eval_case_collection.name}"
        )

        result_dict = {
            "group_id": group_id,
            "timestamp": datetime_now.timestamp(),
            "model": model.name,
            "eval_case_collection": eval_case_collection.name,
            "scores": [score.primary for score in scores],
            "sub_scores": [score.sub_scores for score in scores],
            "model_answers": model_answers,
        }

        for k, v in other_results.items():
            result_dict[k] = v

        self._save_result_dict(result_id, result_dict)

    @abstractmethod
    def load(self, id_regex: str) -> list[dict[str, Any]]:
        """Load evaluation results filtered by regex pattern on id field."""

    @abstractmethod
    def _save_result_dict(
        self, result_id: str, result_dict: dict[str, Any]
    ) -> None: ...


class LocalJsonlAdapter(EvalStorageAdapter):
    def __init__(self, path_to_jsonl: str) -> None:
        super().__init__()
        self.path_to_jsonl = path_to_jsonl

    def load(self, id_regex: str) -> list[dict[str, Any]]:
        """Load evaluation results filtered by regex pattern on id field."""
        pattern = re.compile(id_regex)
        results = []

        try:
            with open(self.path_to_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            result_dict = json.loads(line)
                            if "id" in result_dict and pattern.search(
                                result_dict["id"]
                            ):
                                results.append(result_dict)
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            pass

        return results

    def _save_result_dict(self, result_id: str, result_dict: dict[str, Any]) -> None:
        result_dict_with_id = {"id": result_id, **result_dict}
        with open(self.path_to_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_dict_with_id) + "\n")
