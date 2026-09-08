"""Native compaction must reach Astra requests without weakening route gates."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent.codex_responses_adapter import (
    _normalize_codex_response,
    _classify_responses_issuer,
)
from agent.native_compaction import resolve_native_compaction_capabilities
from agent.message_content import flatten_message_text


def _agent(tmp_path, monkeypatch, base_url, provider):
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "compression:\n  codex_responses_native: true\n"
        "  codex_responses_compact_threshold: 4096\n",
        encoding="utf-8",
    )
    return AIAgent(
        api_key="test-key", base_url=base_url, api_mode="codex_responses",
        model="gpt-6-astra", provider=provider, quiet_mode=True,
        skip_context_files=True, skip_memory=True, enabled_toolsets=[],
    )


@pytest.mark.parametrize("base_url,provider", [
    ("https://api.openai.com/v1", "openai"),
    ("https://chatgpt.com/backend-api/codex", "openai-codex"),
])
def test_astra_config_checkpoint_and_constraints_reach_next_request(
    tmp_path, monkeypatch, base_url, provider
):
    agent = _agent(tmp_path, monkeypatch, base_url, provider)
    original = [{"role": "user", "content": "Keep all outputs local. Do not publish."}]
    before = deepcopy(original)
    first = agent._build_api_kwargs(original)
    assert first["context_management"] == [{"type": "compaction", "compact_threshold": 4096}]
    issuer = _classify_responses_issuer(base_url=base_url, is_codex_backend=provider == "openai-codex")
    response = SimpleNamespace(status="completed", output=[
        SimpleNamespace(type="compaction", encrypted_content="opaque-checkpoint"),
        SimpleNamespace(type="message", status="completed", phase="final_answer",
                        id="msg_1", content=[SimpleNamespace(type="output_text", text="Ready")]),
    ])
    message, finish = _normalize_codex_response(response, issuer_kind=issuer)
    assert finish == "stop"
    history = [*original, {"role": "assistant", "content": message.content,
                           "codex_reasoning_items": message.codex_reasoning_items},
               {"role": "user", "content": "Change of plan: produce only a draft."}]
    saved = deepcopy(history)
    next_request = agent._build_api_kwargs(history)
    assert next_request["context_management"] == first["context_management"]
    assert next_request["store"] is False
    assert "previous_response_id" not in next_request
    checkpoints = [item for item in next_request["input"] if item.get("type") == "compaction"]
    assert checkpoints == [{"type": "compaction", "encrypted_content": "opaque-checkpoint"}]
    user_text = [flatten_message_text(item["content"]) for item in next_request["input"]
                 if item.get("role") == "user"]
    assert original[0]["content"] in user_text
    assert history[-1]["content"] in user_text
    assert original == before and history == saved


@pytest.mark.parametrize("disabled_by", ["opt_out", "compression", "checkpoint", "model", "route"])
def test_astra_disabling_native_keeps_full_history_available(
    tmp_path, monkeypatch, disabled_by
):
    agent = _agent(tmp_path, monkeypatch, "https://api.openai.com/v1", "openai")
    changes = {
        "opt_out": {"codex_responses_native_compaction": False},
        "compression": {"compression_enabled": False},
        "checkpoint": {"compression_checkpoint_required": True},
        "model": {"model": "gpt-5.2"},
        "route": {"base_url": "https://proxy.example/v1", "provider": "custom"},
    }
    for key, value in changes[disabled_by].items():
        setattr(agent, key, value)
    agent.runtime_capabilities = resolve_native_compaction_capabilities(
        model=agent.model, base_url=agent.base_url, provider=agent.provider,
    )
    history = [
        {"role": "user", "content": "Do not publish."},
        {"role": "assistant", "content": "Earlier evidence must remain accessible.",
         "codex_reasoning_items": [{"type": "compaction", "encrypted_content": "opaque-checkpoint",
                                    "_issuer_kind": "other:https://api.openai.com/v1"}]},
        {"role": "user", "content": "Continue locally."},
    ]
    before = deepcopy(history)
    request = agent._build_api_kwargs(history)
    assert "context_management" not in request
    assert not any(item.get("type") == "compaction" for item in request["input"])
    text = [flatten_message_text(item.get("content", "")) for item in request["input"]]
    assert all(message["content"] in text for message in history)
    assert history == before
