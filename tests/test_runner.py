import json
import re
from pathlib import Path

import pytest

from app.llm import LLM
from app.messages import Claim, new_id
from app.runner import run_check
from app.store import Store
from app.tools import LABELS, SpeakerHistory

NO_SPEAKERS = SpeakerHistory(Path("/nonexistent.tsv"))


class FakeLLM:
    """Deterministic analyst JSON built from the prompt; reuses the real JSON/repair logic."""

    provider, model, domain = "fake", "fake-model", "us_politics"
    structured = LLM.structured

    def __init__(self, model: str | None = None) -> None:
        pass

    def chat(self, messages, *, name, metadata):
        user = next(m["content"] for m in messages if m["role"] == "user")
        statement = re.search(r"^Statement: (.*)$", user, re.M).group(1)
        speaker = re.search(r"^Speaker.*?: (.*)$", user, re.M).group(1)
        output = {
            "thesis": statement,
            "checkable": True,
            "domain": self.domain,
            "entities": [],
            "subject": "politics",
            "speaker": None if speaker == "unknown" else speaker,
            "search_queries": [statement[:60]],
        }
        return json.dumps(output), {"model": self.model, "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


class BrokenLLM(FakeLLM):
    def __init__(self, *args) -> None:
        self.replies = iter(["not json at all", "still { broken"])

    def chat(self, messages, *, name, metadata):
        return next(self.replies), {}


class FakeModel:
    version = "fake@test"

    def predict(self, text, metadata):
        return dict(zip(LABELS, [0.05, 0.1, 0.15, 0.4, 0.2, 0.1], strict=True))


def make_claim(statement="Each year, 18,000 people die in America because they don't have health care."):
    return Claim(
        run_id=new_id("run"), sender="user", recipient="runner", statement=statement, speaker="Hillary Clinton"
    )


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "test.db")


def start(store, claim, llm):
    store.create_run(claim.run_id, claim.model_dump(), llm.provider, llm.model)


def test_two_agents_exchange_messages(store):
    llm, claim = FakeLLM(), make_claim()
    start(store, claim, llm)
    status, result = run_check(store, claim, llm, NO_SPEAKERS, FakeModel)
    assert status == "completed"
    analysis, prediction = result["analysis"], result["prediction"]
    assert (analysis["sender"], analysis["recipient"], analysis["in_reply_to"]) == (
        "analyst",
        "classifier",
        claim.message_id,
    )
    assert prediction["in_reply_to"] == analysis["message_id"] and prediction["input_text"] == analysis["thesis"]
    assert prediction["label"] == "half-true" and abs(sum(prediction["probabilities"].values()) - 1) < 1e-6
    assert prediction["metadata"]["speaker"] == "hillary-clinton" and prediction["warnings"]
    assert prediction["applicable"] is True
    events = store.events(claim.run_id)
    calls = [(e["agent"], e["kind"], e["name"]) for e in events if e["kind"] in ("llm_call", "tool_call")]
    assert calls == [
        ("analyst", "llm_call", "analyze_claim"),
        ("classifier", "tool_call", "lookup_speaker_history"),
        ("classifier", "tool_call", "classify_statement"),
    ]
    assert next(e for e in events if e["kind"] == "llm_call")["payload"]["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
    }
    workload = store.workload(claim.run_id)
    assert workload["total_calls"] == 3 and workload["calls_by_agent"] == {"analyst": 1, "classifier": 2}


def test_out_of_domain_statement_is_marked_as_not_a_verdict(store):
    class OtherDomainLLM(FakeLLM):
        domain = "other"

    llm, claim = OtherDomainLLM(), make_claim("The Eiffel Tower was built in 2015.")
    start(store, claim, llm)
    status, result = run_check(store, claim, llm, NO_SPEAKERS, FakeModel)
    assert status == "completed" and result["prediction"]["applicable"] is False
    assert any("outside the model's training domain" in w for w in result["prediction"]["warnings"])


def test_invalid_json_gets_one_repair_then_a_clean_failure(store):
    llm, claim = BrokenLLM(), make_claim()
    start(store, claim, llm)
    status, result = run_check(store, claim, llm, NO_SPEAKERS, FakeModel)
    assert status == "failed" and result == {}
    assert store.get_run(claim.run_id)["error"].startswith("analyst: LLM returned invalid JSON after 2 attempts")
    assert sum(e["kind"] == "llm_call" for e in store.events(claim.run_id)) == 2


def test_missing_checkpoint_fails_cleanly_with_partial_result(store):
    def get_model():
        raise FileNotFoundError("Hybrid checkpoint not found: /nope.pt")

    llm, claim = FakeLLM(), make_claim()
    start(store, claim, llm)
    status, result = run_check(store, claim, llm, NO_SPEAKERS, get_model)
    assert status == "failed" and "analysis" in result and "prediction" not in result
    assert "checkpoint not found" in store.get_run(claim.run_id)["error"]


def test_deadline_stops_the_run_before_the_next_call(store, monkeypatch):
    monkeypatch.setattr("app.runner.config.RUN_TIMEOUT_S", -1.0)
    llm, claim = FakeLLM(), make_claim()
    start(store, claim, llm)
    status, _ = run_check(store, claim, llm, NO_SPEAKERS, FakeModel)
    assert status == "timed_out" and "deadline" in store.get_run(claim.run_id)["error"]


def test_real_llm_requires_an_openrouter_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(Exception, match="OPENROUTER_API_KEY"):
        LLM()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    llm = LLM("openai/gpt-5.4-mini")
    assert (llm.provider, llm.model) == ("openrouter", "openai/gpt-5.4-mini")


def test_llm_chat_passes_langfuse_name_and_openrouter_usage_accounting(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    llm, calls = LLM("openai/gpt-5.4-mini"), []

    class Usage:
        prompt_tokens, completion_tokens = 7, 3

    class Response:
        model, usage = "openai/gpt-5.4-mini", Usage()
        choices = [type("Choice", (), {"message": type("Message", (), {"content": '{"ok": true}'})()})()]

    def fake_create(**kwargs):
        calls.append(kwargs)
        return Response()

    monkeypatch.setattr(llm.client.chat.completions, "create", fake_create)
    text, info = llm.chat([{"role": "user", "content": "hi"}], name="analyze_claim", metadata={"agent": "x"})
    assert text == '{"ok": true}' and info["usage"] == {"prompt_tokens": 7, "completion_tokens": 3}
    assert calls[0]["name"] == "analyze_claim" and calls[0]["metadata"] == {"agent": "x"}
    assert calls[0]["extra_body"] == {"usage": {"include": True}}


class FakeObservation:
    def __init__(self, tracer, as_type, name):
        self.tracer, self.as_type, self.name, self.updates = tracer, as_type, name, []

    def __enter__(self):
        self.tracer.records.append((len(self.tracer.stack), self.as_type, self.name))
        self.tracer.stack.append(self)
        return self

    def __exit__(self, *exc):
        self.tracer.stack.pop()

    def update(self, **kwargs):
        self.updates.append(kwargs)


class FakeTracer:
    def __init__(self):
        self.records, self.stack, self.flushed = [], [], False

    def start_as_current_observation(self, *, as_type="span", name, **kwargs):
        return FakeObservation(self, as_type, name)

    def flush(self):
        self.flushed = True


def test_langfuse_observation_tree_mirrors_the_run(store, monkeypatch):
    tracer = FakeTracer()
    monkeypatch.setattr("app.runner.tracer", tracer)
    llm, claim = FakeLLM(), make_claim()
    start(store, claim, llm)
    run_check(store, claim, llm, NO_SPEAKERS, FakeModel)
    assert tracer.records == [  # the LLM generation itself is recorded by the Langfuse OpenAI integration
        (0, "span", "fact_check"),
        (1, "agent", "analyst"),
        (1, "agent", "classifier"),
        (2, "retriever", "lookup_speaker_history"),
        (2, "tool", "classify_statement"),
    ]
    assert tracer.flushed
    started = next(e for e in store.events(claim.run_id) if e["kind"] == "run" and e["name"] == "started")
    assert len(started["payload"]["trace_id"]) == 32 and started["payload"]["trace_url"] is None


def test_api_validates_input_and_runs_end_to_end(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import app.server as server

    monkeypatch.setattr(server, "store", Store(tmp_path / "api.db"))
    monkeypatch.setattr(server, "load_model", FakeModel)
    client = TestClient(server.app)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert client.post("/api/runs", json={"statement": "x"}).status_code == 400
    assert client.get("/api/config").json()["configured"] is False
    monkeypatch.setattr(server, "LLM", FakeLLM)
    assert client.post("/api/runs", json={"statement": "   "}).status_code == 422
    assert client.post("/api/runs", json={"statement": "x" * 2001}).status_code == 422
    created = client.post("/api/runs", json={"statement": "The unemployment rate is 3 percent."})
    assert created.status_code == 202
    run_id = created.json()["run_id"]
    detail = client.get(f"/api/runs/{run_id}").json()
    assert detail["run"]["status"] == "completed" and detail["workload"]["total_calls"] == 3
    assert client.get(f"/api/runs/{run_id}/events.jsonl").text.count("\n") == len(detail["events"])
