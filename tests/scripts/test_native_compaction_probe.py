"""The live probe must not turn missing or invalid provider evidence into PASS."""

import hashlib
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from evals.compaction.native_probe import create_stream_response, run_probe


def _response(text, *, checkpoint=False, status="completed", missing_terminal=False, model="gpt-6-astra"):
    output = [SimpleNamespace(
        type="message", status="completed", phase="final_answer", id="msg_1",
        content=[SimpleNamespace(type="output_text", text=text)],
    )]
    if checkpoint:
        output.insert(0, SimpleNamespace(type="compaction", encrypted_content="secret-opaque-state"))
    events = [SimpleNamespace(type="response.output_item.done", item=item) for item in output]
    if not missing_terminal:
        events.append(SimpleNamespace(type="response.completed" if status == "completed" else "response.incomplete",
                                      response=SimpleNamespace(model=model, status=status, output=None, usage=None)))
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: nullcontext(iter(events))))
    return create_stream_response(client, {"model": "gpt-6-astra"})


@pytest.mark.parametrize("condition,expected_status", [
    ("valid", "PASS"), ("no_checkpoint", "NO_COMPACTION"),
    ("incomplete", "INCOMPLETE"), ("wrong_fact", "FAIL"),
    ("wrong_boolean", "FAIL"), ("wrong_integer", "FAIL"),
    ("missing_terminal", "INCOMPLETE"), ("wrong_model", "MODEL_MISMATCH"),
])
def test_probe_requires_checkpoint_replay_and_exact_constraint_types(condition, expected_status):
    requests = []
    tag = hashlib.sha256(b"continuity-0").hexdigest()[:16]

    def create(request):
        requests.append(request)
        if len(requests) == 1:
            return _response("Ready", checkpoint=condition != "no_checkpoint",
                             status="incomplete" if condition == "incomplete" else "completed",
                             missing_terminal=condition == "missing_terminal",
                             model="gpt-5.6-sol" if condition == "wrong_model" else "gpt-6-astra")
        assert any(item.get("type") == "compaction" for item in request["input"])
        assert tag not in json.dumps(request["input"])
        result = {"region": "north", "max_items": 1, "publish": False, "archive_tag": tag}
        if condition == "wrong_fact":
            result["archive_tag"] = "wrong"
        elif condition == "wrong_boolean":
            result["publish"] = 0
        elif condition == "wrong_integer":
            result["max_items"] = True
        return _response(json.dumps(result))

    report = run_probe(create, model="gpt-6-astra", provider="openai")
    assert report["status"] == expected_status
    assert len(requests) == (1 if condition in {"no_checkpoint", "incomplete", "missing_terminal", "wrong_model"} else 2)
    assert "secret-opaque-state" not in json.dumps(report)
    assert all(request["store"] is False and "previous_response_id" not in request for request in requests)


@pytest.mark.parametrize("model,fail_turn", [("gpt-5.2", 1), ("gpt-6-astra", 1), ("gpt-6-astra", 2)])
def test_probe_reports_errors_without_secrets_or_retrying(model, fail_turn):
    calls = []

    def create(request):
        calls.append(request)
        if len(calls) == fail_turn:
            raise RuntimeError("private credential and request body")
        return _response("Ready", checkpoint=True)

    report = run_probe(create, model=model, provider="openai")
    assert report["status"] == ("INELIGIBLE" if model == "gpt-5.2" else "ERROR")
    assert len(calls) == (0 if model == "gpt-5.2" else fail_turn)
    assert "private credential" not in json.dumps(report)
    assert "secret-opaque-state" not in json.dumps(report)
