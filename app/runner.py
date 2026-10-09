import os
import time
from contextlib import nullcontext

from langfuse import Langfuse, propagate_attributes

from app import config
from app.agents import Analyst, Classifier
from app.messages import Claim, RunError
from app.store import Store

STEPS = (Analyst.id, Classifier.id)
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


def run_check(store: Store, claim: Claim, llm, speakers, get_model, source: str = "api") -> tuple[str, dict]:
    """Check one claim with both agents and save the final status. Never raises."""
    run = Run(claim.run_id, store)
    store.set_status(claim.run_id, "running")
    trace_id = Langfuse.create_trace_id(seed=claim.run_id)
    result: dict = {}
    status, error = "completed", None
    with (
        propagate_attributes(
            trace_name="fact_check",
            tags=[source],
            metadata={"run_id": claim.run_id, "model": llm.model, "source": source},
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
            source=source,
            limits={
                "run_timeout_s": run.timeout_s,
                "max_steps": run.max_steps,
                "llm_timeout_s": config.LLM_TIMEOUT_S,
                "llm_max_retries": config.LLM_MAX_RETRIES,
            },
            trace_id=trace_id,
            trace_url=_trace_url(trace_id),
        )
        try:
            analysis = _run_agent(run, Analyst(llm), claim)
            result["analysis"] = analysis.model_dump()
            prediction = _run_agent(run, Classifier(get_model, speakers), analysis)
            result["prediction"] = prediction.model_dump()
        except RunError as run_error:
            status, error = run_error.status, run_error.reason
        except Exception as unexpected:  # provider errors, bugs: still a final status, never a hung run
            status, error = "failed", f"{type(unexpected).__name__}: {unexpected}"[:500]
        prediction_dump = result.get("prediction", {})
        root.update(
            output={
                "status": status,
                "label": prediction_dump.get("label"),
                "confidence": prediction_dump.get("confidence"),
                "error": error,
            },
            level="ERROR" if error else "DEFAULT",
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


def _run_agent(run: Run, agent, message):
    run.log("runner", "step", agent.id, "started", input=type(message).__name__)
    with tracer.start_as_current_observation(as_type="agent", name=agent.id, input=message.model_dump()) as span:
        try:
            output = agent.run(run, message)
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
