import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_env()

OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-5.5")
HYBRID_CHECKPOINT = Path(os.environ.get("HYBRID_CHECKPOINT", ROOT / "models/hybrid_liar/best_model.pt"))
BASE_MODEL_DIR = Path(os.environ.get("BASE_MODEL_DIR", ROOT / "models/roberta-base"))
LIAR_TRAIN_TSV = Path(os.environ.get("LIAR_TRAIN_TSV", ROOT / "data/liar/train.tsv"))
DB_PATH = Path(os.environ.get("DB_PATH", ROOT / "data/app.db"))

RUN_TIMEOUT_S = float(os.environ.get("RUN_TIMEOUT_S", "120"))
LLM_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "45"))
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "2"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "8"))
MAX_STATEMENT_CHARS = 2000
