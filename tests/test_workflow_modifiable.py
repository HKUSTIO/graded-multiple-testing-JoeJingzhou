from pathlib import Path


def test_required_project_files_exist():
    root = Path(__file__).resolve().parents[1]
    required = [
        "config/assignment.json",
        "src/multiple_testing.py",
        "scripts/run_cleaning.py",
        "scripts/run_analysis.py",
        "scripts/run_pipeline.py",
        "cleaned/.gitkeep",
        "report/solution.qmd",
        ".github/workflows/classroom.yml",
    ]
    for rel in required:
        assert (root / rel).exists(), f"Required file missing: {rel}"


def test_report_references_run_script():
    root = Path(__file__).resolve().parents[1]
    qmd = (root / "report" / "solution.qmd").read_text(encoding="utf-8")
    assert "scripts/run_pipeline.py" in qmd
