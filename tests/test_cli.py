import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.benchmark import benchmark_verdict, score_with_model
from src.model import LangChainModel


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def run_cli(self, module, *args):
        completed = subprocess.run(
            [sys.executable, "-m", module, *args], cwd=ROOT,
            capture_output=True, text=True, encoding="utf-8", check=True,
        )
        return json.loads(completed.stdout)

    def test_start_and_confirm_across_cli_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "workflow.db")
            result = self.run_cli("src.main", "--db", db, "start", "examples/normal.input.json")
            self.assertEqual(result["status"], "awaiting_confirmation")
            self.assertIn("response", result)
            saved = self.run_cli("src.main", "--db", db, "confirm", result["request_id"],
                                 result["proposal_id"], "1")
            self.assertEqual(saved["status"], "draft_saved")
            self.assertEqual(saved["saved_slot"], result["candidates"][0])

    def test_chat_routes_text_before_database_or_model_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "unused.db"
            for message, expected, reply_contains in (
                ("你好！", "greeting", "素材审核"),
                ("谢谢", "thanks", "不客气"),
                ("你能做什么？", "scope_help", "start"),
                ("你好，帮我写个排序算法", "out_of_scope", "发布排期"),
                ("帮我安排旅游排期", "out_of_scope", "发布排期"),
                ("帮我写一段短视频发布文案", "out_of_scope", "发布排期"),
                ("帮我排个期", "scope_clarification", "内容素材"),
                ("你好，帮我给视频排个期", "needs_input", "允许的时间窗口"),
                ("帮我发布这条短视频", "needs_input", "允许的时间窗口"),
                ("确认第 1 项", "confirmation_required", "proposal_id"),
                ("   ", "empty_input", "告诉我"),
            ):
                with self.subTest(message=message):
                    result = self.run_cli("src.main", "--mode", "langchain", "--db", str(db),
                                          "--data", str(Path(tmp) / "missing.json"), "chat", message)
                    self.assertEqual(result["status"], expected)
                    self.assertIn(reply_contains, result["response"])
            self.assertFalse(db.exists())

    def test_benchmark_covers_the_core_paths_and_records_rubric(self):
        report = self.run_cli("src.benchmark", "--mode", "offline")
        self.assertEqual(report["summary"]["passed"], 15)
        self.assertEqual(report["summary"]["failed"], 0)
        self.assertEqual(len(report["cases"]), 15)
        self.assertTrue(report["verdict"]["passed"])
        self.assertEqual(report["verdict"]["score"], 100)
        self.assertEqual(report["verdict"]["threshold"], 80)
        for case in report["cases"]:
            self.assertIn("rubric", case)
            self.assertIn("trace", case["initial"])

    def test_benchmark_threshold_and_hard_gates_are_independent(self):
        case = {"passed": True, "rubric": {"score": 79}}
        self.assertFalse(benchmark_verdict([case], {"passed": True}, 0)["passed"])
        case = {"passed": False, "rubric": {"score": 100}}
        self.assertFalse(benchmark_verdict([case], {"passed": True}, 0)["passed"])
        case = {"passed": True, "rubric": {"score": 80}}
        self.assertTrue(benchmark_verdict([case], {"passed": True}, 0)["passed"])

    def test_remember_preference_cli_is_read_by_a_later_assessment(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "workflow.db")
            path = Path(tmp) / "busy.json"
            data = json.loads((ROOT / "data/demo.json").read_text(encoding="utf-8"))
            data["bookings"] = [{"resource": "DEMO-CHANNEL", "start": "2030-05-15T14:30:00+08:00",
                                 "end": "2030-05-15T15:00:00+08:00"}]
            path.write_text(json.dumps(data), encoding="utf-8")
            result = self.run_cli("src.main", "--db", db, "remember-preference", "DEMO-USER", "latest")
            self.assertEqual(result["status"], "preference_saved")
            proposal = self.run_cli("src.main", "--db", db, "--data", str(path), "start",
                                    "examples/normal.input.json")
            self.assertEqual(proposal["candidates"][0]["start"], "2030-05-15T15:30:00+08:00")

    def test_model_adapter_routes_two_endpoint_configurations_and_exposes_usage(self):
        for name, endpoint, mode, method in (
            ("local-model", "http://localhost:8080/v1", "schema", "json_schema"),
            ("compatible-model", "https://provider.example/v1", "object", "json_mode"),
        ):
            with self.subTest(mode=mode):
                chat = Mock()
                chat.with_structured_output.return_value.invoke.return_value = {
                    "parsed": {"action": "get_material", "basis": "Needs material facts."},
                    "raw": SimpleNamespace(usage_metadata={"input_tokens": 12, "output_tokens": 5}),
                }
                factory = Mock(return_value=chat)
                settings = {"SCHEDULE_MODEL": name, "SCHEDULE_BASE_URL": endpoint,
                            "SCHEDULE_JSON_MODE": mode, "SCHEDULE_API_KEY": "fixture"}
                with patch.dict(os.environ, settings), patch.dict(
                    sys.modules, {"langchain_openai": SimpleNamespace(ChatOpenAI=factory)}
                ):
                    decision = LangChainModel().decide({"allowed": ["get_material"], "observed": []})
                self.assertEqual(factory.call_args.kwargs["model"], name)
                self.assertEqual(factory.call_args.kwargs["base_url"], endpoint)
                self.assertEqual(chat.with_structured_output.call_args.kwargs["method"], method)
                self.assertEqual(decision["usage"], {"input_tokens": 12, "output_tokens": 5})

    def test_semantic_judge_must_cite_an_existing_step(self):
        class BadJudge:
            label = "test"

            def structured(self, schema, instruction, payload):
                return {"grounded": 2, "actionable": 2, "confirmation": 2, "constraints": 2,
                        "evidence_step": 999, "evidence": "Unsupported citation"}

        result = {"status": "needs_input", "response": "Supply rights information.",
                  "trace": [{"step": 1, "action": "get_material", "observation": {"status": "missing"}}]}
        self.assertEqual(score_with_model(BadJudge(), result, result)["status"], "eval_error")
