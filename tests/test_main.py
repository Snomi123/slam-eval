import datetime
from typing import Any, Optional

import hydra
import pytest
from freezegun import freeze_time
from omegaconf import DictConfig, OmegaConf

from slam_eval.collections.base import CollectionInfo, EvalCase, EvalCaseCollection
from slam_eval.collections.text_generation import TextGenerationInput
from slam_eval.model import Model
from slam_eval.scripts.main import main
from slam_eval.storage_adapter import EvalStorageAdapter
from slam_eval.utils.common import get_config_path
from slam_eval.utils.typing import HasStr

DICT_STORAGE = []


class SimpleEvalCaseCollection(EvalCaseCollection):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.i = 0

    def _load(self) -> CollectionInfo:
        self.collection_data = [
            ("Test question 1", "Test answer 1"),
            ("Test question 2", "Test answer 2"),
            ("Test question 3", "Test answer 3"),
        ]
        return CollectionInfo(
            collection=iter(self.collection_data),
            collection_len=len(self.collection_data),
        )

    def __next__(self) -> EvalCase:
        if self.i >= len(self.collection_data):
            raise StopIteration

        res = self.collection_data[self.i]
        self.i += 1
        return {
            "x": TextGenerationInput(system_prompt=None, user_prompt=res[0]),
            "y_true": res[1],
        }


class SimpleEvalStorageAdapter(EvalStorageAdapter):
    def __init__(self) -> None:
        global DICT_STORAGE
        self.dict_storage: list[dict[str, Any]] = DICT_STORAGE

    def load(self, id_regex: str) -> list[dict[str, Any]]:
        """Load evaluation results filtered by regex pattern on id field."""
        import re

        pattern = re.compile(id_regex)
        results = []

        for result_dict in self.dict_storage:
            if "id" in result_dict and pattern.search(result_dict["id"]):
                results.append(result_dict)

        return results

    def _save_result_dict(self, result_id: str, result_dict: dict[str, Any]) -> None:
        result_dict_with_id = {"id": result_id, **result_dict}
        self.dict_storage.append(result_dict_with_id)


@pytest.fixture
def cfg():
    with hydra.initialize(
        version_base="1.3", config_path="../config", job_name="test_app"
    ):
        default_cfg = hydra.compose(config_name="config_main")

    return default_cfg


@pytest.fixture
def eval_case_collection_cfg():
    return {
        "_target_": "tests.test_main.SimpleEvalCaseCollection",
        "name": "simple_eval_case_collection",
    }


@pytest.fixture
def storage_adapter_cfg():
    return {
        "_target_": "tests.test_main.SimpleEvalStorageAdapter",
    }


@pytest.fixture(autouse=True)
def reset_dict_storage():
    """Reset the global DICT_STORAGE before each test."""
    global DICT_STORAGE
    DICT_STORAGE.clear()
    yield
    DICT_STORAGE.clear()


@freeze_time("2000-01-01")
def test_main(
    cfg: DictConfig, eval_case_collection_cfg, storage_adapter_cfg, monkeypatch
):
    # Mock requests to LLMs
    monkeypatch.setattr(
        "slam_eval.model.request_based_on_message_history",
        lambda *args, **kwargs: {"role": "assistant", "content": "Test answer 1"},
    )

    # Mock eval case collection
    cfg.collection = eval_case_collection_cfg

    # Mock storage adapter
    cfg.storage_adapter = storage_adapter_cfg

    # Run the function being tested
    main(cfg)

    # Check the storage
    datetime_now = datetime.datetime.now()

    global DICT_STORAGE
    assert DICT_STORAGE == [
        {
            "id": "eval:{group_id}:{datetime}_M_{model}_C_{eval_case_collection}".format(
                group_id=cfg.group_id,
                datetime=datetime_now.isoformat("_"),
                model=cfg.model.name,
                eval_case_collection=cfg.collection.name,
            ),
            "group_id": cfg.group_id,
            "timestamp": datetime_now.timestamp(),
            "model": cfg.model.name,
            "eval_case_collection": cfg.collection.name,
            "scores": [1, 0, 0],
            "sub_scores": [None, None, None],
            "model_answers": ["Test answer 1"] * 3,
        }
    ]
