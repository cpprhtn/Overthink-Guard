import json
from pathlib import Path

import pytest

from overthink_guard.cli import main
from overthink_guard.templates import get_template, template_for_model, template_ids

FIXTURES = Path(__file__).parent / "fixtures"


def test_templates_load_and_inherit_from_generic():
    assert {"generic", "qwen3", "deepseek_r1"} <= set(template_ids())
    qwen = get_template("qwen3")
    assert qwen.think_end == "</think>"
    assert qwen.answer_patterns == get_template("generic").answer_patterns
    assert get_template("deepseek_r1").starts_in_thinking


@pytest.mark.parametrize(
    ("model", "template_id"),
    [("qwen3:8b", "qwen3"), ("QwQ-32B", "qwen3"), ("deepseek-r1:7b", "deepseek_r1"), ("llama3.1", "generic")],
)
def test_template_for_model(model, template_id):
    assert template_for_model(model).id == template_id


def test_unknown_template_lists_available():
    with pytest.raises(KeyError, match="available"):
        get_template("nope")


def test_analyze_json_summary(capsys):
    files = [str(FIXTURES / f) for f in ("synthetic_overthink.txt", "synthetic_essay.txt")]
    assert main(["analyze", "--json", *files]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["summary"]["files"] == 2
    assert out["summary"]["stopped"] == 1
    assert out["results"][0]["stop"]["answer"] == "9"
    assert out["results"][1]["stop"] is None


def test_analyze_thinking_only_input(tmp_path, capsys):
    trace = tmp_path / "t.txt"
    body = (FIXTURES / "synthetic_overthink.txt").read_text(encoding="utf-8")
    trace.write_text(body.split("<think>")[1].split("</think>")[0], encoding="utf-8")
    assert main(["analyze", "--json", "--thinking-only", str(trace)]) == 0
    result = json.loads(capsys.readouterr().out)["results"][0]
    assert result["stop"]["answer"] == "9"
    assert result["final_answer"] is None
