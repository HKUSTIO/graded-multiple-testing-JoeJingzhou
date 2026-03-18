import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.multiple_testing import summarize_multiple_testing


def main() -> None:
    config = json.loads((ROOT / "config" / "assignment.json").read_text(encoding="utf-8"))
    cleaned_dir = ROOT / "cleaned"
    output_dir = ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    null_pvalues = pd.read_csv(cleaned_dir / "null_pvalues.csv")
    mixed_pvalues = pd.read_csv(cleaned_dir / "mixed_pvalues.csv")

    results = summarize_multiple_testing(
        null_pvalues=null_pvalues,
        mixed_pvalues=mixed_pvalues,
        alpha=float(config["alpha"]),
    )
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
