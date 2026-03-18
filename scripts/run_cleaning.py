import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.multiple_testing import simulate_mixed_pvalues, simulate_null_pvalues


def main() -> None:
    config = json.loads((ROOT / "config" / "assignment.json").read_text(encoding="utf-8"))
    cleaned_dir = ROOT / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    null_pvalues = simulate_null_pvalues(config=config)
    mixed_pvalues = simulate_mixed_pvalues(config=config)

    null_pvalues.to_csv(cleaned_dir / "null_pvalues.csv", index=False)
    mixed_pvalues.to_csv(cleaned_dir / "mixed_pvalues.csv", index=False)


if __name__ == "__main__":
    main()
