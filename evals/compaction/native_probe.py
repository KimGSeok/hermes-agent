"""Opt-in, two-request native-compaction/continuity probe using synthetic data.

Exercises Hermes' real request converter and response sidecar, not the full
agent loop. Never persists credentials, raw history, or opaque reasoning blobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from agent.codex_responses_adapter import _classify_responses_issuer, _normalize_codex_response
from agent.native_compaction import native_compaction_context_management, resolve_native_compaction_capabilities
from agent.transports.codex import ResponsesApiTransport


ROUTES = {
    "openai": "https://api.openai.com/v1",
    "openai-codex": "https://chatgpt.com/backend-api/codex",
}


def run_probe(create, *, model, provider, seed=0):
    """Return a falsifiable receipt; a missing checkpoint is never a recall pass."""
    base_url = ROUTES[provider]
    codex = provider == "openai-codex"
    agent = SimpleNamespace(
        model=model, base_url=base_url, codex_responses_native_compaction=True,
        compression_enabled=True, codex_responses_compact_threshold=1024,
        runtime_capabilities=resolve_native_compaction_capabilities(
            model=model, base_url=base_url, provider=provider, is_codex_backend=codex,
        ),
    )
    directive = native_compaction_context_management(agent, is_codex_backend=codex)
    report = {
        "model": model, "provider": provider, "seed": seed,
        "scope": "synthetic Responses adapter integration; not full-agent quality",
        "native_eligible": directive is not None, "requests": [],
        "checkpoint_observed": False, "checkpoint_replayed": False,
        "archived_fact_absent_from_plaintext": False, "constraints_retained": False,
        "status": "INELIGIBLE",
    }
    root = Path(__file__).resolve().parents[2]
    report["source_digest"] = hashlib.sha256(b"".join(
        (root / name).read_bytes() for name in (
            "evals/compaction/native_probe.py", "agent/native_compaction.py",
            "agent/transports/codex.py", "agent/codex_responses_adapter.py", "agent/codex_runtime.py",
        )
    )).hexdigest()
    if directive is None:
        return report
    tag = hashlib.sha256(f"continuity-{seed}".encode()).hexdigest()[:16]
    expected = {"region": "north", "max_items": 1, "publish": False, "archive_tag": tag}
    history = [
        {"role": "user", "content": "Use region north and at most 3 items. Never publish. Retain these constraints."},
        {"role": "assistant", "content": f"The computed archive_tag is {tag}. This fact is needed later."},
        {"role": "user", "content": "Read the progress log, keep the archive tag and constraints, and acknowledge readiness."},
        {"role": "assistant", "content": "\n".join(f"Checked batch {i}: unchanged, no action needed." for i in range(800))},
        {"role": "user", "content": "Acknowledge readiness briefly without repeating the archive tag."},
    ]
    report["fixture_digest"] = hashlib.sha256(json.dumps(history, sort_keys=True).encode()).hexdigest()
    transport = ResponsesApiTransport()
    issuer = _classify_responses_issuer(base_url=base_url, is_codex_backend=codex)
    started = time.monotonic()
    try:
        for turn in range(2):
            request = transport.build_kwargs(
                model=model, messages=history, tools=[], base_url=base_url, provider=provider,
                is_codex_backend=codex, context_management=directive,
                reasoning_config={"effort": "medium", "enabled": True},
                instructions="Follow the user's current constraints. Return only the requested answer.",
            )
            if turn:
                report["checkpoint_replayed"] = any(i.get("type") == "compaction" for i in request["input"])
                visible = [{k: v for k, v in item.items() if k != "encrypted_content"} for item in request["input"]]
                report["archived_fact_absent_from_plaintext"] = tag not in json.dumps(visible)
            request_record = {"turn": turn, "context_management_sent": "context_management" in request,
                              "reasoning": request.get("reasoning")}
            report["requests"].append(request_record)
            tick = time.monotonic()
            response = create(request)
            request_record.update(
                elapsed_seconds=round(time.monotonic() - tick, 3),
                response_model=getattr(response, "model", None),
                response_status=getattr(response, "status", None),
                usage=response.usage.model_dump() if getattr(response, "usage", None) else None,
            )
            message, finish = _normalize_codex_response(response, issuer_kind=issuer)
            request_record["finish_reason"] = finish
            if getattr(response, "status", None) != "completed" or finish != "stop":
                report["status"] = "INCOMPLETE"
                break
            if getattr(response, "model", None) != model:
                report["status"] = "MODEL_MISMATCH"
                break
            sidecar = message.codex_reasoning_items or []
            if not turn:
                report["checkpoint_observed"] = any(i.get("type") == "compaction" for i in sidecar)
                if not report["checkpoint_observed"]:
                    report["status"] = "NO_COMPACTION"
                    break
                history.extend([
                    {"role": "assistant", "content": message.content, "codex_reasoning_items": sidecar},
                    {"role": "user", "content": "Change max_items to 1. Return JSON with region, max_items, publish (boolean), and the earlier archive_tag."},
                ])
            else:
                try:
                    actual = json.loads(message.content)
                except (TypeError, ValueError):
                    actual = None
                report["constraints_retained"] = actual == expected and all(
                    type(actual[key]) is type(value) for key, value in expected.items()
                )
                report["status"] = "PASS" if all(report[key] for key in (
                    "checkpoint_observed", "checkpoint_replayed",
                    "archived_fact_absent_from_plaintext", "constraints_retained",
                )) else "FAIL"
    except Exception as exc:
        # Provider exceptions can include request bodies; never log str(exc).
        report.update(status="ERROR", error_type=type(exc).__name__,
                      http_status=getattr(exc, "status_code", None))
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return report


def create_stream_response(client, request):
    """Use Hermes' stream assembler, but require a real terminal frame for evals."""
    from agent.codex_runtime import _consume_codex_event_stream

    terminal = {}

    def observe(event):
        if event.type in {"response.completed", "response.failed", "response.incomplete"}:
            response = event.response
            terminal.update(status=response.status, model=getattr(response, "model", None))

    with client.responses.create(**request, stream=True) as stream:
        response = _consume_codex_event_stream(stream, model=request["model"], on_event=observe)
    # The production assembler intentionally salvages useful text without a terminal
    # frame; an evaluator cannot promote that salvage into completed provider proof.
    response.status = terminal.get("status", "incomplete")
    response.model = terminal.get("model")
    return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", choices=ROUTES, default="openai")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--live", action="store_true", help="Authorize at most two provider requests; requires OPENAI_API_KEY")
    args = parser.parse_args()
    if not args.live:
        parser.error("pass --live explicitly; this probe consumes provider tokens")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        parser.error("OPENAI_API_KEY is required; no profile credentials are loaded")
    from openai import OpenAI
    from agent.codex_headers import codex_cloudflare_headers

    headers = codex_cloudflare_headers(key) if args.provider == "openai-codex" else {}
    # Reserve the destination before spending tokens; never overwrite an earlier attempt.
    with args.out.open("x", encoding="utf-8") as output, OpenAI(
        api_key=key, base_url=ROUTES[args.provider], default_headers=headers,
        timeout=60, max_retries=0,
    ) as client:
        def create(request):
            return create_stream_response(client, request)

        report = run_probe(create, model=args.model, provider=args.provider, seed=args.seed)
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({"status": report["status"], "report": str(args.out)}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
