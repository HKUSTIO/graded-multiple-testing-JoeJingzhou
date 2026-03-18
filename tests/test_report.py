import json
from pathlib import Path


def test_output_files_exist_after_run():
    root = Path(__file__).resolve().parents[1]
    results = root / "output" / "results.json"
    null_pvalues = root / "cleaned" / "null_pvalues.csv"
    mixed_pvalues = root / "cleaned" / "mixed_pvalues.csv"

    assert null_pvalues.exists(), "Missing cleaned/null_pvalues.csv. Run scripts/run_pipeline.py."
    assert mixed_pvalues.exists(), "Missing cleaned/mixed_pvalues.csv. Run scripts/run_pipeline.py."
    assert results.exists(), "Missing output/results.json. Run scripts/run_pipeline.py."


def test_results_json_has_required_keys():
    root = Path(__file__).resolve().parents[1]
    results = json.loads((root / "output" / "results.json").read_text(encoding="utf-8"))
    required = {
        "fwer_uncorrected",
        "fwer_bonferroni",
        "fwer_holm",
        "fdr_uncorrected",
        "fdr_bh",
        "fdr_by",
        "power_uncorrected",
        "power_bh",
        "power_by",
    }
    assert required.issubset(results.keys()), "results.json is missing required outputs."


def test_rendered_html_exists_and_contains_required_sections():
    root = Path(__file__).resolve().parents[1]
    html_path = root / "report" / "solution.html"
    assert html_path.exists(), "Missing report/solution.html. Render report/solution.qmd."

    html = html_path.read_text(encoding="utf-8")
    required_strings = [
        "FWER control",
        "FDR and power",
        "uncorrected",
        "bonferroni",
        "holm",
        "benjamini_hochberg",
        "benjamini_yekutieli",
    ]
    for token in required_strings:
        assert token in html, f"report/solution.html does not contain `{token}`."
