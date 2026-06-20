import json
import os
import tempfile
from unittest.mock import Mock

from slam_eval.collections.base import EvalCaseCollection
from slam_eval.model import Model
from slam_eval.scorer import Score
from slam_eval.storage_adapter import LocalJsonlAdapter


class TestLocalJsonlAdapter:
    def test_init(self):
        adapter = LocalJsonlAdapter("/path/to/file.jsonl")
        assert adapter.path_to_jsonl == "/path/to/file.jsonl"

    def test_save_result_dict_creates_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = os.path.join(temp_dir, "test_results.jsonl")
            adapter = LocalJsonlAdapter(jsonl_path)

            test_dict = {
                "model": "test_model",
                "scores": [1, 0, 1],
                "sub_scores": [None, None, None],
            }

            adapter._save_result_dict("test_eval_123", test_dict)

            assert os.path.exists(jsonl_path)

            with open(jsonl_path, "r") as f:
                content = f.read().strip()
            loaded_dict = json.loads(content)
            expected_dict = {"id": "test_eval_123", **test_dict}
            assert loaded_dict == expected_dict

    def test_save_result_dict_appends_to_existing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = os.path.join(temp_dir, "test_results.jsonl")
            adapter = LocalJsonlAdapter(jsonl_path)

            first_dict = {"score": 0.8}
            adapter._save_result_dict("test_1", first_dict)

            second_dict = {"score": 0.9}
            adapter._save_result_dict("test_2", second_dict)

            with open(jsonl_path, "r") as f:
                lines = f.read().strip().split("\n")
            assert len(lines) == 2
            assert json.loads(lines[0]) == {"id": "test_1", **first_dict}
            assert json.loads(lines[1]) == {"id": "test_2", **second_dict}

    def test_save_integration_with_base_class(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = os.path.join(temp_dir, "test_results.jsonl")
            adapter = LocalJsonlAdapter(jsonl_path)

            mock_model = Mock(spec=Model)
            mock_model.name = "test_model"

            mock_collection = Mock(spec=EvalCaseCollection)
            mock_collection.name = "test_collection"

            scores = [
                Score(primary=1.0, sub_scores=None),
                Score(primary=0.0, sub_scores=None),
                Score(primary=1.0, sub_scores=None),
            ]
            model_answers = ["answer1", "answer2", "answer3"]

            adapter.save(
                group_id="test_group",
                model=mock_model,
                eval_case_collection=mock_collection,
                scores=scores,
                model_answers=model_answers,
                custom_field="custom_value",
            )

            with open(jsonl_path, "r") as f:
                content = f.read().strip()
            result_dict = json.loads(content)

            assert result_dict["model"] == "test_model"
            assert result_dict["eval_case_collection"] == "test_collection"
            assert result_dict["group_id"] == "test_group"
            assert result_dict["scores"] == [1.0, 0.0, 1.0]
            assert result_dict["sub_scores"] == [None, None, None]
            assert result_dict["model_answers"] == model_answers
            assert result_dict["custom_field"] == "custom_value"
            assert "id" in result_dict
            assert result_dict["id"].startswith("eval:test_group:")
            assert "timestamp" in result_dict

    def test_save_multiple_calls_append_correctly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = os.path.join(temp_dir, "test_results.jsonl")
            adapter = LocalJsonlAdapter(jsonl_path)

            mock_model = Mock(spec=Model)
            mock_model.name = "test_model"
            mock_collection = Mock(spec=EvalCaseCollection)
            mock_collection.name = "test_collection"

            for i in range(3):
                adapter.save(
                    group_id=f"group_{i}",
                    model=mock_model,
                    eval_case_collection=mock_collection,
                    scores=[Score(primary=float(i), sub_scores=None)],
                    model_answers=[f"answer_{i}"],
                )

            with open(jsonl_path, "r") as f:
                lines = f.read().strip().split("\n")
            assert len(lines) == 3

            for i, line in enumerate(lines):
                result_dict = json.loads(line)
                assert result_dict["scores"] == [float(i)]
                assert result_dict["sub_scores"] == [None]
                assert result_dict["model_answers"] == [f"answer_{i}"]
                assert result_dict["group_id"] == f"group_{i}"
                assert result_dict["id"].startswith(f"eval:group_{i}:")

    def test_storage_format_with_sub_scores(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = os.path.join(temp_dir, "test_results.jsonl")
            adapter = LocalJsonlAdapter(jsonl_path)

            mock_model = Mock(spec=Model)
            mock_model.name = "test_model"
            mock_collection = Mock(spec=EvalCaseCollection)
            mock_collection.name = "test_collection"

            scores = [
                Score(
                    primary=0.5,
                    sub_scores={"checker_one": 1.0, "checker_two": 0.0},
                )
            ]

            adapter.save(
                group_id="test_group",
                model=mock_model,
                eval_case_collection=mock_collection,
                scores=scores,
                model_answers=["answer1"],
            )

            with open(jsonl_path, "r") as f:
                content = f.read().strip()
            result_dict = json.loads(content)

            assert result_dict["scores"] == [0.5]
            assert result_dict["sub_scores"] == [
                {"checker_one": 1.0, "checker_two": 0.0}
            ]

    def test_storage_format_without_sub_scores(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = os.path.join(temp_dir, "test_results.jsonl")
            adapter = LocalJsonlAdapter(jsonl_path)

            mock_model = Mock(spec=Model)
            mock_model.name = "test_model"
            mock_collection = Mock(spec=EvalCaseCollection)
            mock_collection.name = "test_collection"

            scores = [Score(primary=1.0, sub_scores=None)]

            adapter.save(
                group_id="test_group",
                model=mock_model,
                eval_case_collection=mock_collection,
                scores=scores,
                model_answers=["answer1"],
            )

            with open(jsonl_path, "r") as f:
                content = f.read().strip()
            result_dict = json.loads(content)

            assert result_dict["scores"] == [1.0]
            assert result_dict["sub_scores"] == [None]

    def test_load_returns_empty_list_when_file_not_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = os.path.join(temp_dir, "nonexistent.jsonl")
            adapter = LocalJsonlAdapter(jsonl_path)

            results = adapter.load(".*")
            assert results == []

    def test_load_filters_by_regex(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = os.path.join(temp_dir, "test_results.jsonl")
            adapter = LocalJsonlAdapter(jsonl_path)

            test_data = [
                (
                    "eval:group1:test_1",
                    {"model": "model1", "scores": [1], "sub_scores": [None]},
                ),
                (
                    "eval:group2:test_2",
                    {"model": "model2", "scores": [2], "sub_scores": [None]},
                ),
                (
                    "eval:group1:test_3",
                    {"model": "model1", "scores": [3], "sub_scores": [None]},
                ),
                (
                    "other:group1:test_4",
                    {"model": "model3", "scores": [4], "sub_scores": [None]},
                ),
            ]

            for result_id, data in test_data:
                adapter._save_result_dict(result_id, data)

            results = adapter.load(r"eval:group1:.*")
            assert len(results) == 2
            assert results[0]["id"] == "eval:group1:test_1"
            assert results[1]["id"] == "eval:group1:test_3"

    def test_load_with_different_regex_patterns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = os.path.join(temp_dir, "test_results.jsonl")
            adapter = LocalJsonlAdapter(jsonl_path)

            test_data = [
                (
                    "eval:exp1:2023-01-01_M_model1_C_collection1",
                    {"model": "model1", "scores": [1], "sub_scores": [None]},
                ),
                (
                    "eval:exp2:2023-01-02_M_model2_C_collection1",
                    {"model": "model2", "scores": [2], "sub_scores": [None]},
                ),
                (
                    "eval:exp1:2023-01-03_M_model1_C_collection2",
                    {"model": "model1", "scores": [3], "sub_scores": [None]},
                ),
            ]

            for result_id, data in test_data:
                adapter._save_result_dict(result_id, data)

            results_exp1 = adapter.load(r"eval:exp1:.*")
            assert len(results_exp1) == 2

            results_model1 = adapter.load(r".*_M_model1_.*")
            assert len(results_model1) == 2

            results_collection1 = adapter.load(r".*_C_collection1")
            assert len(results_collection1) == 2

            results_all = adapter.load(r"eval:.*")
            assert len(results_all) == 3

    def test_load_handles_malformed_json_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = os.path.join(temp_dir, "test_results.jsonl")
            adapter = LocalJsonlAdapter(jsonl_path)

            with open(jsonl_path, "w") as f:
                f.write('{"id": "eval:test:1", "model": "model1"}\n')
                f.write("invalid json line\n")
                f.write('{"id": "eval:test:2", "model": "model2"}\n')
                f.write("\n")

            results = adapter.load(r"eval:test:.*")
            assert len(results) == 2
            assert results[0]["id"] == "eval:test:1"
            assert results[1]["id"] == "eval:test:2"

    def test_load_returns_empty_list_for_no_matches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = os.path.join(temp_dir, "test_results.jsonl")
            adapter = LocalJsonlAdapter(jsonl_path)

            test_data = [
                ("eval:group1:test_1", {"model": "model1"}),
                ("eval:group2:test_2", {"model": "model2"}),
            ]

            for result_id, data in test_data:
                adapter._save_result_dict(result_id, data)

            results = adapter.load(r"eval:group3:.*")
            assert results == []

    def test_load_handles_entries_without_id_field(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = os.path.join(temp_dir, "test_results.jsonl")
            adapter = LocalJsonlAdapter(jsonl_path)

            with open(jsonl_path, "w") as f:
                f.write('{"id": "eval:test:1", "model": "model1"}\n')
                f.write('{"model": "model2", "scores": [1, 2]}\n')
                f.write('{"id": "eval:test:2", "model": "model3"}\n')

            results = adapter.load(r"eval:test:.*")
            assert len(results) == 2
            assert results[0]["id"] == "eval:test:1"
            assert results[1]["id"] == "eval:test:2"
