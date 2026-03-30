# PFE Reporting App

Bootstrap implementation for the AI-based document extraction and reporting system.

## Current Scope

This scaffold now covers:
- project structure aligned with the target architecture
- centralized configuration and logging
- Label Studio project configuration generation
- annotation guideline generation and export helpers
- Label Studio local Docker bootstrap
- inter-annotator agreement reporting
- regex + spaCy ensemble extraction
- sequential confidence and decision routing
- KPI summaries
- FastAPI production-style endpoints
- database-backed async batch job persistence
- automated test coverage across the end-to-end scaffold

## Quick Start

1. Create a virtual environment.
2. Install dependencies: `pip install -e .[dev]`
3. Copy `.env.example` to `.env` and update the values.
4. Run tests: `pytest`
5. Run the API: `uvicorn app.main:app --reload`

## API Endpoints

- `GET /health`
- `POST /api/v1/extract`
- `POST /api/v1/pipeline/run`
- `POST /api/v1/pipeline/batch`
- `POST /api/v1/pipeline/batch/submit`
- `GET /api/v1/pipeline/batch/jobs/{job_id}`
- `POST /api/v1/kpi/report`
- `GET /api/v1/admin/status`
- `GET /api/v1/admin/model`
- `GET /api/v1/admin/metrics`

API errors return a shared JSON payload:
`{"detail": "...", "error_code": "..."}`.

Successful API responses now include traceability metadata with:
`processed_at`, `app_version`, `pipeline_version`, `extraction_version`, and `model_version`.

Async batch flow:
1. Submit work with `POST /api/v1/pipeline/batch/submit`
2. Poll status and results with `GET /api/v1/pipeline/batch/jobs/{job_id}`

Async batch jobs are persisted through SQLAlchemy ORM, so they can survive API process restarts as long as the configured database remains available.

## Annotation Workflow

1. Generate annotation assets:
   `python3 -m app.annotation.cli generate-assets --output-dir docs/annotation`
2. Start Label Studio locally:
   `./scripts/bootstrap_label_studio.sh`
3. Import local source documents into a project:
   `python3 -m app.annotation.cli import-tasks --project-id 1 --input-dir docs/source_documents`
4. Export completed annotations into training JSON:
   `python3 -m app.annotation.cli export-annotations --project-id 1 --output-file docs/annotation/training_export.json --raw-output-file docs/annotation/label_studio_export.json`
5. Convert normalized training JSON into spaCy JSONL:
   `python3 -m app.annotation.cli export-spacy --input-file docs/annotation/training_export.json --output-file docs/annotation/spacy_train.jsonl`
6. Compute annotator agreement from JSON summaries:
   `python3 -m app.annotation.cli agreement-report --annotator-a docs/annotation/sample_annotator_a.json --annotator-b docs/annotation/sample_annotator_b.json`

## Supported Source Files

- `.txt`: direct text import
- `.json`: object containing `text`, optional `document_id`, optional `metadata`
- `.csv`: flattened into row-labeled text for annotation
- `.pdf`: supported when `pypdf` is installed
- `.xlsx`: supported when `openpyxl` is installed

## Label Studio Deployment

- Docker Compose files live in `deploy/label_studio/`
- Copy `deploy/label_studio/.env.example` to `deploy/label_studio/.env` if you want custom local credentials and ports
- Example input files for task import live in `docs/source_documents/`

## NER Training

1. Validate spaCy JSONL training data:
   `python3 -m app.ml.cli validate-ner-data --input-file docs/annotation/spacy_train.jsonl`
2. Split examples into train and validation sets:
   `python3 -m app.ml.cli split-ner-data --input-file docs/annotation/spacy_train_demo_multi.jsonl --train-output-file docs/annotation/spacy_train_split.jsonl --validation-output-file docs/annotation/spacy_validation_split.jsonl --validation-ratio 0.2`
3. Evaluate predictions against gold annotations:
   `python3 -m app.ml.cli evaluate-ner-data --gold-file docs/annotation/spacy_train.jsonl --predicted-file docs/annotation/spacy_predicted_sample.jsonl`
4. Train the spaCy NER model:
   `python3 -m app.ml.cli train-ner --input-file docs/annotation/spacy_train.jsonl --output-dir artifacts/models/ner --iterations 20`
5. Run regex + spaCy ensemble extraction on raw text:
   `python3 -m app.ml.cli extract-entities --text $'Invoice INV-2026-001\nVendor: Acme Supplies LLC\nDate: 2026-03-29\nTotal: $12.00'`
6. Run the threshold-based sequential pipeline:
   `python3 -m app.ml.cli run-pipeline --text $'Invoice INV-2026-001\nVendor: Acme Supplies LLC\nDate: 2026-03-29\nTotal: $12.00'`
7. Run the batch pipeline with aggregated metrics:
   `python3 -m app.ml.cli run-pipeline-batch --input-dir docs/source_documents`
8. Build a reusable KPI summary from the batch pipeline:
   `python3 -m app.ml.cli build-kpi-report --input-dir docs/source_documents`

If spaCy is not installed yet, the training command will fail with a clear message while validation still works.
If the trained spaCy model directory at `NER_MODEL_PATH` is missing, ensemble extraction falls back to regex-only mode.
With the current default regex-only scores, extracted fields route to `review` because they sit in the `0.70-0.89` threshold band.
Per-field routing is configurable through `FIELD_THRESHOLDS_JSON`, which lets fields like `VENDOR_NAME` auto-approve sooner than stricter fields like `TOTAL_AMOUNT`.
The evaluation command now reports both aggregate metrics and a `per_label` breakdown so weak fields can be targeted directly.
The KPI command summarizes document-level outcomes, per-field counts, and average confidence for reuse in future APIs and dashboards.
