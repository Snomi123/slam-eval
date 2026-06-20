from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from slam_eval.utils.typing import HasStr


@dataclass(frozen=True)
class Score:
    primary: float
    sub_scores: dict[str, float] | None = None


class Scorer(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def __call__(self, y_true: Any, y_pred: Any) -> Score: ...


class ExactMatch(Scorer):
    def __init__(
        self,
        name: str,
        preprocessing_func: Callable[[Any], Any] | None = None,
    ) -> None:
        super().__init__(name)
        self.preprocessing_func = preprocessing_func

    def _preprocess(self, value: Any) -> Any:
        if self.preprocessing_func is None:
            return value
        return self.preprocessing_func(value)

    def __call__(self, y_true: Any, y_pred: Any) -> Score:
        processed_true = self._preprocess(y_true)
        processed_pred = self._preprocess(y_pred)
        return Score(
            primary=float(int(processed_true == processed_pred)), sub_scores=None
        )


def json_string_to_dict(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        clean_value = value.strip()
        if clean_value.startswith("```json"):
            clean_value = clean_value[7:]
        elif clean_value.startswith("```"):
            clean_value = clean_value[3:]
        if clean_value.endswith("```"):
            clean_value = clean_value[:-3]
        clean_value = clean_value.strip()

        try:
            return json.loads(clean_value)
        except json.JSONDecodeError:
            # Fallback to extract first `{...}` or `[...]` block via regex
            import re

            match = re.search(r"(\{.*\}|\[.*\])", clean_value, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            return value
    return value


def build_json_string_to_dict() -> Callable[[Any], Any]:
    return json_string_to_dict


class IgnoreAllWhitespaces(Scorer):
    def __call__(self, y_true: HasStr, y_pred: HasStr) -> Score:
        y_true_str = str(y_true)
        y_pred_str = str(y_pred)

        escaped_chars = [re.escape(char) for char in y_true_str if not char.isspace()]
        if not escaped_chars:
            contains_non_whitespace = any(not char.isspace() for char in y_pred_str)
            return Score(
                primary=float(int(not contains_non_whitespace)), sub_scores=None
            )

        pattern = r"\s*".join(escaped_chars)
        pattern = rf"^\s*{pattern}\s*$"

        return Score(
            primary=float(int(bool(re.match(pattern, y_pred_str)))),
            sub_scores=None,
        )
