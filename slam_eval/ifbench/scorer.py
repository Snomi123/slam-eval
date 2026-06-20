from __future__ import annotations

import inspect
from typing import Any, Dict, Iterable, Mapping

from slam_eval.ifbench.checker_factory import IFBenchCheckerFactory
from slam_eval.scorer import Score, Scorer


class IFBenchScorer(Scorer):
    def __init__(self, name: str, checker_factory: IFBenchCheckerFactory) -> None:
        super().__init__(name)
        self._checker_factory = checker_factory
        self._build_signature_cache: dict[type[Any], tuple[set[str], bool]] = {}

    def __call__(self, y_true: Mapping[str, Any], y_pred: str) -> Score:
        instruction_ids: Iterable[str] = y_true["instruction_id_list"]
        kwargs_list: Iterable[Dict[str, Any]] = y_true["kwargs"]

        sub_scores: dict[str, float] = {}
        results: list[float] = []

        for instruction_id, raw_kwargs in zip(
            instruction_ids, kwargs_list, strict=True
        ):
            checker = self._checker_factory(instruction_id)
            build_kwargs = self._prepare_build_description_kwargs(checker, raw_kwargs)
            checker.build_description(**build_kwargs)

            checker_result = float(int(bool(checker.check_following(y_pred))))
            sub_scores[instruction_id] = checker_result
            results.append(checker_result)

        if not results:
            return Score(primary=0.0, sub_scores={})

        return Score(
            primary=sum(results) / len(results),
            sub_scores=sub_scores,
        )

    def _prepare_build_description_kwargs(
        self,
        checker: Any,
        raw_kwargs: Mapping[str, Any] | None,
    ) -> Dict[str, Any]:
        if not raw_kwargs:
            return {}

        accepted_keys, accepts_kwargs = self._get_build_signature(checker)

        filtered_kwargs = {}
        for key, value in raw_kwargs.items():
            if value is None:
                continue
            if key in accepted_keys or accepts_kwargs:
                filtered_kwargs[key] = value

        return filtered_kwargs

    def _get_build_signature(self, checker: Any) -> tuple[set[str], bool]:
        checker_type = type(checker)
        if checker_type in self._build_signature_cache:
            return self._build_signature_cache[checker_type]

        signature = inspect.signature(checker.build_description)
        accepted_keys = {
            param.name
            for param in signature.parameters.values()
            if param.name != "self"
            and param.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }

        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )

        result = (accepted_keys, accepts_kwargs)
        self._build_signature_cache[checker_type] = result
        return result
