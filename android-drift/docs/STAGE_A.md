# Stage A: offline data preparation

Run commands from `android-drift`. These commands read APK archives statically; they never install, execute or upload APKs. Output directories must be new.

```powershell
.\.venv\Scripts\python.exe -m driftlab extract-batch --input data/apks --output data/extraction-v1
.\.venv\Scripts\python.exe -m driftlab validate-metadata --metadata data/metadata.csv
.\.venv\Scripts\python.exe -m driftlab build-dataset --extractions data/extraction-v1/extractions.jsonl --metadata data/metadata.csv --train-through 2024-12 --output data/dataset-v1
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Metadata CSV requires `sample_id,month,label,source`. `sample_id` is the lowercase APK SHA256, month is the externally verified observation month (`YYYY-MM`), label is exactly 0 or 1, and source records label/data provenance. Optional columns (e.g. label policy and scan date) are preserved in provenance. Do not infer labels from permissions or dates from filesystem timestamps. Sources and labels are declarations, not independently verified by this validator.

Extraction writes one JSONL record per file in sorted path order, including failures and duplicate hashes, plus summary counts. A batch with failures returns CLI exit code 1 after saving its results. Invalid input or an existing output directory returns exit code 2.

Builder joins metadata to successful records by hash, rejects missing/conflicting records and duplicate metadata IDs, and sorts rows by month/hash. Vocabulary uses only metadata samples at or before `--train-through`; later unknown tokens are recorded in provenance and omitted from vectors. Set this cutoff to the actual initial training period before running experiments. Features are manifest-only, not Drebin/DEX/API features. Output `dataset.csv` is compatible with `driftlab.data.load_csv`; `provenance.json` records source hashes, vocabulary, cutoff, metadata and future-only tokens. Unreferenced extractions are counted. Default dense CSV limit: 5,000,000 elements; no sparse support is implied. Existing experiment constraints (continuous months, sufficient initial class counts) still apply.

The tests use generated archive fixtures only. They do not inspect project APKs or establish Android malware accuracy.
