from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, TypedDict

from slam_eval.collections.base import (CollectionInfo, EvalCaseCollection,
                                        check_if_loaded)


# NOTE: These TypedDicts intentionally mirror the ones in
# ``slam_eval.collections.text_generation`` but are redefined locally so that
# importing this module does NOT pull in ``text_generation`` (which imports the
# heavy, optional ``datasets`` package at module load time). This keeps
# ``hydra.utils.instantiate(cfg.collection)`` working even when ``datasets`` is
# not installed in the runtime environment.
class TextGenerationInput(TypedDict):
    system_prompt: Optional[str]
    user_prompt: str


class TextGenerationWithUniqueGroundTruth(TypedDict):
    x: TextGenerationInput
    y_true: str | dict[str, Any]


DEFAULT_DATASET_PATH = "/Users/igormajorov/Desktop/nauka/merge_quality_dataset.jsonl"

DEFAULT_PROMPT_TEMPLATE = (
    "You are given the unique identifiers of a single target person and "
    "several data chunks. Some chunks describe the target person, while "
    "others are distractors or contain mixed data that must be ignored.\n"
    "Using only the information that belongs to the target person, merge the "
    "relevant data chunks into a single JSON object with the attributes of "
    "that person. Your answer must contain only the resulting JSON, nothing "
    "else, and must exclude the unique identifiers themselves.\n\n"
    "### Unique identifiers of the target person\n\n"
    "{provided_identifiers}\n\n"
    "### Data chunks\n\n"
    "{chunks}"
)


@dataclass
class _MergeQualityExample:
    provided_identifiers: dict[str, Any]
    chunks: list[dict[str, Any]]
    attributes: dict[str, Any]


class MergeQualityCollection(EvalCaseCollection):
    """Eval-case collection for the merge-quality JSONL dataset.

    Each JSONL line has the shape::

        {
          "ground_truth": {"attributes": {...}, ...},
          "provided_identifiers": {...},
          "chunks": [{"format": "...", "owner_id": "...", "content": "..."}, ...]
        }

    For every record this collection yields an :class:`EvalCase` where:

    * ``x`` is a :class:`TextGenerationInput` (``system_prompt`` / ``user_prompt``)
      built from ``provided_identifiers`` and all ``chunks``.  The dict shape is
      required because :class:`slam_eval.model.LlmViaOpenAiApi` indexes ``x`` by
      ``"system_prompt"`` and ``"user_prompt"``.
    * ``y_true`` is ``item["ground_truth"]["attributes"]``.
    """

    def __init__(
        self,
        name: str = "merge_quality_dataset",
        jsonl_path: str = DEFAULT_DATASET_PATH,
        user_prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        system_prompt: str | None = None,
    ) -> None:
        super().__init__(name)
        self.jsonl_path = Path(jsonl_path).expanduser()
        self.user_prompt_template = user_prompt_template
        self.system_prompt = system_prompt

    # ------------------------------------------------------------------
    # Prompt construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _format_provided_identifiers(provided_identifiers: dict[str, Any]) -> str:
        return json.dumps(provided_identifiers, ensure_ascii=False, indent=2)

    @staticmethod
    def _format_chunks(chunks: list[dict[str, Any]]) -> str:
        formatted: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            fmt = chunk.get("format", "unknown")
            owner_id = chunk.get("owner_id", "unknown")
            content = chunk.get("content", "")
            formatted.append(
                f"Chunk #{index}\n"
                f"Format: {fmt}\n"
                f"Owner: {owner_id}\n"
                f"Content:\n{content}"
            )
        return "\n\n".join(formatted)

    def _build_user_prompt(self, example: _MergeQualityExample) -> str:
        return self.user_prompt_template.format(
            provided_identifiers=self._format_provided_identifiers(
                example.provided_identifiers
            ),
            chunks=self._format_chunks(example.chunks),
        )

    # ------------------------------------------------------------------
    # EvalCaseCollection contract
    # ------------------------------------------------------------------
    def _load(self) -> CollectionInfo:
        raw_lines = self.jsonl_path.read_text(encoding="utf-8").splitlines()
        non_empty_lines = [line for line in raw_lines if line.strip()]

        def _iterator() -> Iterator[_MergeQualityExample]:
            for line in non_empty_lines:
                payload = json.loads(line)
                ground_truth = payload["ground_truth"]
                yield _MergeQualityExample(
                    provided_identifiers=payload["provided_identifiers"],
                    chunks=payload.get("chunks", []),
                    attributes=ground_truth["attributes"],
                )

        return CollectionInfo(
            collection=_iterator(),
            collection_len=len(non_empty_lines),
        )

    @check_if_loaded
    def __next__(self) -> TextGenerationWithUniqueGroundTruth:
        example: _MergeQualityExample = next(self.collection)  # type: ignore[arg-type]
        return TextGenerationWithUniqueGroundTruth(  # type: ignore[misc]
            x=TextGenerationInput(
                system_prompt=self.system_prompt,
                user_prompt=self._build_user_prompt(example),
            ),
            y_true=example.attributes,
        )
