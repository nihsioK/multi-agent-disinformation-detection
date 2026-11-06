import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from contextvars import copy_context

from langfuse import Langfuse, propagate_attributes

from app import config
from app.agents import Analyst, Assessor, Classifier, Retriever, Synthesizer, Verifier
from app.messages import Claim, RunError
from app.store import Store

STEPS = (Analyst.id, Retriever.id, Classifier.id, Assessor.id, Synthesizer.id, Verifier.id)
TRACING = bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))
tracer = Langfuse()  # disables itself (no-op observations) when the Langfuse keys are not set


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


class Run:
    """One check: the time and step budget plus the single wrapper every LLM and tool call goes through."""

    def __init__(self, run_id: str, store: Store, timeout_s: float | None = None, max_steps: int | None = None) -> None:
        self.run_id, self.store = run_id, store
        self.timeout_s = config.RUN_TIMEOUT_S if timeout_s is None else timeout_s
        self.max_steps = config.MAX_STEPS if max_steps is None else max_steps
        self.deadline = time.monotonic() + self.timeout_s
        self.steps = 0
        self._lock = threading.Lock()  # retriever and classifier run in parallel

    def log(
        self, agent: str, kind: str, name: str, status: str = "ok", duration_ms: int | None = None, **payload
    ) -> None:
        self.store.add_event(self.run_id, agent, kind, name, status, duration_ms, payload)

    def call(self, agent: str, kind: str, name: str, fn, *, as_type: str | None = None, **payload):
        """Run `fn() -> (value, info)` with deadline and step checks, timing, logging and tracing.

        `as_type` (tool, retriever, ...) opens a Langfuse observation around the call; LLM calls pass None
        because the Langfuse OpenAI integration already records them as generations.
        """
        if time.monotonic() > self.deadline:
            raise RunError("timed_out", f"run deadline of {self.timeout_s:.0f}s reached before {agent}.{name}")
        with self._lock:
            self.steps += 1
            if self.steps > self.max_steps:
                raise RunError("failed", f"step budget of {self.max_steps} exceeded at {agent}.{name}")
        started = time.monotonic()
        observation = nullcontext()
        if as_type:
            observation = tracer.start_as_current_observation(
                as_type=as_type, name=name, input=payload or None, metadata={"agent": agent}
            )
        with observation as span:
            try:
                value, info = fn()
            except Exception as error:
                reason = error.reason if isinstance(error, RunError) else f"{type(error).__name__}: {error}"[:500]
                if span is not None:
                    span.update(level="ERROR", status_message=reason)
                self.log(agent, kind, name, "error", _ms(started), error=reason, **payload)
                raise
            if span is not None:
                span.update(output=info)
        self.log(agent, kind, name, "ok", _ms(started), **{**payload, **info})
        return value


def run_check(
    store: Store, claim: Claim, llm, speakers, get_model, search, source: str = "api", link_checker=None
) -> tuple[str, dict]:
    """Check one claim with all agents and save the final status. Never raises."""
    run = Run(claim.run_id, store)
    store.set_status(claim.run_id, "running")
    trace_id = Langfuse.create_trace_id(seed=claim.run_id)
    result: dict = {}
    status, error = "completed", None
    with (
        propagate_attributes(
            trace_name="fact_check",
            tags=[source],
            metadata={"run_id": claim.run_id, "model": llm.model, "source": source, "search": search.backend},
        ),
        tracer.start_as_current_observation(
            as_type="span",
            name="fact_check",
            trace_context={"trace_id": trace_id},
            input={"statement": claim.statement, "speaker": claim.speaker, "context": claim.context},
            metadata={"message_id": claim.message_id},
        ) as root,
    ):
        run.log(
            "runner",
            "run",
            "started",
            plan=list(STEPS),
            provider=llm.provider,
            model=llm.model,
            search=search.backend,
            source=source,
            limits={
                "run_timeout_s": run.timeout_s,
                "max_steps": run.max_steps,
                "llm_timeout_s": config.LLM_TIMEOUT_S,
                "llm_max_retries": config.LLM_MAX_RETRIES,
                "revise_rounds": 1,
            },
            trace_id=trace_id,
            trace_url=_trace_url(trace_id),
        )
        try:
            _steps(run, result, claim, llm, speakers, get_model, search, link_checker)
        except RunError as run_error:
            status, error = run_error.status, run_error.reason
        except Exception as unexpected:  # provider errors, bugs: still a final status, never a hung run
            status, error = "failed", f"{type(unexpected).__name__}: {unexpected}"[:500]
        draft = result.get("draft", {})
        root.update(
            output={
                "status": status,
                "verdict": draft.get("verdict"),
                "confidence": draft.get("confidence"),
                "error": error,
            },
            level="ERROR" if status in ("failed", "timed_out") else "DEFAULT",
            status_message=error,
        )
    store.finish_run(claim.run_id, status, result, error)
    run.log(
        "runner",
        "run",
        "finished",
        "ok" if status == "completed" else "error",
        final_status=status,
        error=error,
        steps_used=run.steps,
    )
    tracer.flush()
    return status, result


def _steps(run: Run, result: dict, claim: Claim, llm, speakers, get_model, search, link_checker) -> None:
    analysis = _run_agent(run, Analyst(llm), claim)
    result["analysis"] = analysis.model_dump()

    with ThreadPoolExecutor(max_workers=2) as pool:  # search and the local model do not depend on each other
        evidence_future = pool.submit(copy_context().run, _run_agent, run, Retriever(search), analysis)
        prediction_future = pool.submit(copy_context().run, _run_agent, run, Classifier(get_model, speakers), analysis)
        prediction = prediction_future.result()
        evidence = evidence_future.result()
    result["prediction"] = prediction.model_dump()
    result["evidence"] = evidence.model_dump()
    if not evidence.items:
        raise RunError("insufficient_evidence", evidence.note or "no sources found")

    assessment = _run_agent(run, Assessor(llm), analysis, evidence)
    result["assessment"] = assessment.model_dump()
    verifier = Verifier(link_checker) if link_checker else Verifier()

    draft = _run_agent(run, Synthesizer(llm), analysis, prediction, evidence, assessment)
    result["draft"] = draft.model_dump()
    verification = _run_agent(run, verifier, draft, evidence, assessment)
    result["verification"] = verification.model_dump()
    if verification.status == "revise":  # exactly one round of fixes, as the plan allows
        draft = _run_agent(
            run,
            Synthesizer(llm, fixes=verification.required_fixes, revision=1),
            analysis,
            prediction,
            evidence,
            assessment,
        )
        result["draft"] = draft.model_dump()
        verification = _run_agent(run, verifier, draft, evidence, assessment)
        result["verification"] = verification.model_dump()
        if verification.status == "revise":
            raise RunError("failed", "verifier rejected the verdict twice: " + "; ".join(verification.required_fixes))


def _run_agent(run: Run, agent, *messages):
    run.log("runner", "step", agent.id, "started", input=[type(m).__name__ for m in messages])
    with tracer.start_as_current_observation(
        as_type="agent", name=agent.id, input=[m.model_dump() for m in messages]
    ) as span:
        try:
            output = agent.run(run, *messages)
        except Exception as error:
            span.update(level="ERROR", status_message=str(error)[:500])
            run.log("runner", "step", agent.id, "error")
            raise
        span.update(output=output.model_dump())
    run.log("runner", "step", agent.id, "finished", output=type(output).__name__)
    return output


def _trace_url(trace_id: str) -> str | None:
    if not TRACING:
        return None
    try:
        return tracer.get_trace_url(trace_id=trace_id)
    except Exception:  # the UI link is a convenience; never fail a run over it
        return None
