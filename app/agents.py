import json

from app.llm import LLM
from app.messages import Analysis, AnalystOutput, Claim, Prediction, RunError
from app.tools import LABELS, SpeakerHistory, slugify

ANALYST_PROMPT = """You are the analyst, the first step of a fact-checking pipeline for political statements.
Your only job: turn the input into exactly ONE checkable thesis plus hints for the next steps.
You never judge whether the statement is true or false.

Guidelines:
- thesis: one self-contained factual claim, as close to the original wording as possible, max 40 words; resolve pronouns using the speaker and context when possible.
- checkable: false for pure opinions, predictions, questions or slogans.
- domain: "us_politics" when the claim is about US politics, policy, economy, elections or public figures (what PolitiFact rates), otherwise "other".
- entities: people, organizations, places, laws and figures worth verifying.
- date / location: only if stated or clearly implied, otherwise null.
- subject: 1-3 lowercase PolitiFact-style topics separated by commas (e.g. "immigration,economy").
- speaker / context: copy from the input when given; otherwise infer only if the statement names them, else null.
- search_queries: 2-3 short web search queries that would find evidence.

Return ONLY a JSON object (no markdown, no commentary) that matches this JSON schema:
{schema}"""  # noqa: E501


class Analyst:
    """Claim -> Analysis. One LLM call, plus one retry if the JSON comes back broken."""

    id = "analyst"

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def run(self, run, claim: Claim) -> Analysis:
        prompt = ANALYST_PROMPT.format(schema=json.dumps(AnalystOutput.model_json_schema()))
        user = (
            f"Statement: {claim.statement}\n"
            f"Speaker (if known): {claim.speaker or 'unknown'}\n"
            f"Context (if known): {claim.context or 'unknown'}"
        )
        output = self.llm.structured(run, self.id, "analyze_claim", prompt, user, AnalystOutput)
        analysis = Analysis(
            **output.model_dump(),
            run_id=run.run_id,
            sender=self.id,
            recipient=Classifier.id,
            in_reply_to=claim.message_id,
        )
        analysis.speaker = claim.speaker or analysis.speaker  # what the user typed wins over a guess
        analysis.context = claim.context or analysis.context
        if not analysis.thesis.strip():
            raise RunError("failed", f"{self.id}: empty thesis, nothing to check")
        run.log(self.id, "message", "Analysis", recipient=analysis.recipient, message=analysis.model_dump())
        return analysis


class Classifier:
    """Analysis -> Prediction. Looks the speaker up in LIAR, then runs the thesis model."""

    id = "classifier"

    def __init__(self, get_model, speakers: SpeakerHistory) -> None:
        self.get_model = get_model
        self.speakers = speakers

    def run(self, run, analysis: Analysis) -> Prediction:
        history = run.call(
            self.id,
            "tool_call",
            "lookup_speaker_history",
            lambda: self._history(analysis.speaker),
            as_type="retriever",
            speaker=analysis.speaker,
        )
        metadata = {
            "speaker": slugify(analysis.speaker) if analysis.speaker else None,
            "party": None,
            "job": None,
            "state": None,
            "subject": analysis.subject,
            "context": analysis.context,
            "counts": {},
        }
        warnings = []
        if history:
            metadata.update(history)
        elif analysis.speaker:
            warnings.append(f"speaker '{analysis.speaker}' has no LIAR history; the model sees an unknown speaker")
        else:
            warnings.append("no speaker given; credibility features fall back to the neutral prior")
        if not analysis.checkable:
            warnings.append("analyst marked the statement as not checkable; the model signal is indicative only")
        if analysis.domain != "us_politics":
            warnings.append(
                "statement is outside the model's training domain (US political claims); the label is not a verdict"
            )

        probabilities, version = run.call(
            self.id,
            "tool_call",
            "classify_statement",
            lambda: self._classify(analysis.thesis, metadata),
            as_type="tool",
            text=analysis.thesis,
            metadata=metadata,
        )
        if abs(sum(probabilities.values()) - 1.0) > 1e-3 or set(probabilities) != set(LABELS):
            raise RunError("failed", f"{self.id}: model returned an invalid distribution")
        label = max(probabilities, key=probabilities.get)
        prediction = Prediction(
            run_id=run.run_id,
            sender=self.id,
            recipient="runner",
            in_reply_to=analysis.message_id,
            checkpoint_version=version,
            input_text=analysis.thesis,
            label=label,
            label_id=LABELS.index(label),
            confidence=probabilities[label],
            probabilities=probabilities,
            metadata=metadata,
            applicable=analysis.checkable and analysis.domain == "us_politics",
            warnings=warnings,
        )
        run.log(self.id, "message", "Prediction", recipient=prediction.recipient, message=prediction.model_dump())
        return prediction

    def _history(self, speaker: str | None):
        record = self.speakers.lookup(speaker)
        return record, {"found": record is not None, "source": self.speakers.source}

    def _classify(self, text: str, metadata: dict):
        model = self.get_model()  # loads the checkpoint on first use
        probabilities = model.predict(text, metadata)
        label = max(probabilities, key=probabilities.get)
        info = {
            "label": label,
            "confidence": round(probabilities[label], 4),
            "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
            "checkpoint": model.version,
        }
        return (probabilities, model.version), info
