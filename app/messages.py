import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app import config

SCHEMA_VERSION = "1.1"
FINAL_STATUSES = ("completed", "failed", "timed_out", "insufficient_evidence")
Verdict = Literal["true", "mostly-true", "half-true", "mostly-false", "false", "unverifiable"]
Stance = Literal["supports", "refutes", "mixed", "unrelated"]


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class RunError(Exception):
    """Stops the run with a final `status` and a readable `reason`."""

    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status, self.reason = status, reason


class Message(BaseModel):
    """Common header of every message the agents exchange."""

    schema_version: str = SCHEMA_VERSION
    message_id: str = Field(default_factory=lambda: new_id("msg"))
    run_id: str = ""
    created_at: str = Field(default_factory=now_iso)
    sender: str = ""
    recipient: str = ""
    in_reply_to: str | None = None


class ClaimInput(BaseModel):
    statement: str = Field(max_length=config.MAX_STATEMENT_CHARS)
    speaker: str | None = None
    context: str | None = None

    @field_validator("statement")
    @classmethod
    def _statement_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("statement must not be empty")
        return value

    @field_validator("speaker", "context")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        return value.strip() or None if value else None


class Claim(Message, ClaimInput):
    """User -> runner."""


class AnalystOutput(BaseModel):
    """What the analyst's LLM must return (also the JSON schema shown in the prompt)."""

    thesis: str = Field(description="One self-contained checkable claim, close to the original wording, max 40 words")
    checkable: bool = Field(description="false for opinions, predictions, questions or slogans")
    domain: Literal["us_politics", "other"] = Field(
        default="other",
        description="us_politics for claims of the kind PolitiFact rates (US politics, policy, economy, elections, "
        "public figures), otherwise other",
    )
    entities: list[str] = Field(
        default_factory=list, description="People, organizations, places, laws, figures to verify"
    )
    date: str | None = Field(default=None, description="Time reference if stated or clearly implied, else null")
    location: str | None = Field(default=None, description="Geography the claim is about, else null")
    subject: str | None = Field(
        default=None, description="1-3 lowercase PolitiFact-style topics, comma separated, e.g. 'immigration,economy'"
    )
    speaker: str | None = Field(default=None, description="Who made the statement if known or named, else null")
    context: str | None = Field(default=None, description="Where or how the statement was made if known, else null")
    search_queries: list[str] = Field(default_factory=list, description="2-3 short web search queries to find evidence")
    language: str = Field(default="en", description="ISO 639-1 code of the statement language")
    notes: str | None = Field(default=None, description="Ambiguities the next steps should know about, else null")


class Analysis(Message, AnalystOutput):
    """Analyst -> retriever and classifier."""


class Prediction(Message):
    """Classifier -> runner: the thesis model's verdict with probabilities."""

    checkpoint_version: str
    input_text: str
    label: str
    label_id: int
    confidence: float
    probabilities: dict[str, float]
    metadata: dict
    # False when the statement is outside the model's domain: the label is a signal, not a verdict
    applicable: bool = True
    warnings: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    evidence_id: str
    url: str
    domain: str
    title: str
    published: str | None = None
    snippet: str
    content_hash: str
    accessed_at: str


class Evidence(Message):
    """Retriever -> assessor: what the web search found, without any judgement."""

    queries: list[str]
    backend: str
    items: list[EvidenceItem]
    sufficient: bool  # at least two independent domains
    note: str | None = None  # e.g. why the search came back empty


class SourceAssessment(BaseModel):
    evidence_id: str
    relevance: float = Field(ge=0, le=1, description="How much the source says about the thesis itself")
    reliability: float = Field(ge=0, le=1, description="How trustworthy the publisher is")
    stance: Stance = Field(description="Does the source support or refute the thesis")
    reasons: str = Field(description="One sentence")


class AssessorOutput(BaseModel):
    items: list[SourceAssessment]


class Assessment(Message, AssessorOutput):
    """Assessor -> synthesizer: one assessment per source, reliability blended with domain rules."""

    independent_domains: int


class SynthesizerOutput(BaseModel):
    verdict: Verdict = Field(description="One of true, mostly-true, half-true, mostly-false, false, unverifiable")
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(description="3-6 sentences, cite sources by id like [E1]")
    evidence_ids: list[str] = Field(description="Exactly the source ids the rationale relies on")
    conflicts: str | None = Field(default=None, description="Disagreements between sources or with the model signal")
    limitations: str | None = Field(default=None, description="What could not be checked")


class Draft(Message, SynthesizerOutput):
    """Synthesizer -> verifier."""

    revision: int = 0


class Check(BaseModel):
    name: str
    passed: bool
    detail: str
    blocking: bool = True


class Verification(Message):
    """Verifier -> runner: pass, or revise with the list of required fixes."""

    status: Literal["pass", "revise"]
    checks: list[Check]
    required_fixes: list[str] = Field(default_factory=list)
