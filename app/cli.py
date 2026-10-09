import argparse
import json
import sys
from pathlib import Path

from app import config
from app.llm import LLM, ConfigError
from app.messages import Claim, new_id
from app.runner import run_check
from app.store import Store
from app.tools import SpeakerHistory, get_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check one political statement with the two agents.")
    parser.add_argument("statement")
    parser.add_argument("--speaker")
    parser.add_argument("--context")
    parser.add_argument("--model", help="OpenRouter model id, default from LLM_MODEL")
    parser.add_argument("--db", default=str(config.DB_PATH))
    args = parser.parse_args(argv)
    try:
        llm = LLM(args.model)
        claim = Claim(
            run_id=new_id("run"),
            sender="user",
            recipient="runner",
            statement=args.statement,
            speaker=args.speaker,
            context=args.context,
        )
    except (ConfigError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    store = Store(Path(args.db))
    store.create_run(claim.run_id, claim.model_dump(), llm.provider, llm.model)
    status, result = run_check(
        store,
        claim,
        llm,
        SpeakerHistory(config.LIAR_TRAIN_TSV),
        lambda: get_model(config.HYBRID_CHECKPOINT),
        source="cli",
    )
    run = store.get_run(claim.run_id)
    output = {
        "run_id": claim.run_id,
        "status": status,
        "error": run["error"],
        "result": result,
        "workload": store.workload(claim.run_id),
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
