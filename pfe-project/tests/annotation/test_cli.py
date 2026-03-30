"""Tests for the annotation bootstrap CLI."""

import json
from pathlib import Path
from typing import Any, Dict, List

from app.annotation import cli


def test_cli_generate_assets_writes_output(tmp_path: Path, capsys) -> None:
    """Ensure the CLI generates bootstrap assets and prints their paths."""
    exit_code = cli.main(["generate-assets", "--output-dir", str(tmp_path)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert Path(payload["schema"]).exists()
    assert Path(payload["guidelines"]).exists()


def test_cli_agreement_report_outputs_summary(tmp_path: Path, capsys) -> None:
    """Ensure the CLI prints a field-level agreement report."""
    annotator_a_path = tmp_path / "annotator_a.json"
    annotator_b_path = tmp_path / "annotator_b.json"

    annotator_a_path.write_text(
        json.dumps({"doc-001": ["INVOICE_ID"], "doc-002": ["TOTAL_AMOUNT"]}),
        encoding="utf-8",
    )
    annotator_b_path.write_text(
        json.dumps({"doc-001": ["INVOICE_ID"], "doc-002": []}),
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "agreement-report",
            "--annotator-a",
            str(annotator_a_path),
            "--annotator-b",
            str(annotator_b_path),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert "INVOICE_ID" in payload
    assert payload["INVOICE_ID"]["cohen_kappa"] == 1.0


def test_cli_import_tasks_uses_label_studio_client(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    """Ensure import-tasks loads documents and sends them to the client."""
    (tmp_path / "invoice_001.txt").write_text(
        "Invoice INV-2026-001",
        encoding="utf-8",
    )
    captured_tasks: Dict[str, Any] = {}

    def _fake_import(self: Any, project_id: int, tasks: List[Dict[str, Any]]) -> Dict[str, int]:
        captured_tasks["project_id"] = project_id
        captured_tasks["count"] = len(tasks)
        return {"task_count": len(tasks)}

    monkeypatch.setattr(cli.LabelStudioClient, "import_tasks", _fake_import)

    exit_code = cli.main(
        ["import-tasks", "--project-id", "12", "--input-dir", str(tmp_path)]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert captured_tasks["project_id"] == 12
    assert captured_tasks["count"] == 1
    assert payload["import_response"]["task_count"] == 1


def test_cli_export_annotations_saves_files(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    """Ensure export-annotations writes raw and training-ready outputs."""
    raw_payload = [
        {
            "id": 21,
            "data": {
                "document_id": "invoice-021",
                "text": "Invoice INV-2026-021 total is $42.00",
            },
            "annotations": [
                {
                    "result": [
                        {
                            "type": "labels",
                            "value": {
                                "start": 8,
                                "end": 20,
                                "text": "INV-2026-021",
                                "labels": ["INVOICE_ID"],
                            },
                        }
                    ]
                }
            ],
        }
    ]

    def _fake_export(self: Any, project_id: int, export_type: str = "JSON") -> List[Dict[str, Any]]:
        return raw_payload

    monkeypatch.setattr(cli.LabelStudioClient, "export_annotations", _fake_export)

    output_file = tmp_path / "training.json"
    raw_output_file = tmp_path / "raw.json"
    exit_code = cli.main(
        [
            "export-annotations",
            "--project-id",
            "21",
            "--output-file",
            str(output_file),
            "--raw-output-file",
            str(raw_output_file),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output_file.exists()
    assert raw_output_file.exists()
    assert payload["documents"] == 1
    assert payload["saved_paths"]["training"] == str(output_file)


def test_cli_export_spacy_saves_jsonl(tmp_path: Path, capsys: Any) -> None:
    """Ensure export-spacy converts normalized training JSON into JSONL."""
    training_file = tmp_path / "training.json"
    training_file.write_text(
        json.dumps(
            [
                {
                    "document_id": "doc-100",
                    "text": "Invoice INV-100",
                    "annotations": [
                        {
                            "start": 8,
                            "end": 15,
                            "text": "INV-100",
                            "label": "INVOICE_ID",
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    output_file = tmp_path / "spacy.jsonl"

    exit_code = cli.main(
        [
            "export-spacy",
            "--input-file",
            str(training_file),
            "--output-file",
            str(output_file),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output_file.exists()
    assert payload["documents"] == 1
    assert payload["saved_paths"]["spacy_jsonl"] == str(output_file)
