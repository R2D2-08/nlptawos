# BERT-Based Word Embeddings for Within-Project Software Story Point Estimation

Run from the repository root using Python 3.10+ and MySQL 8.

## 1. Set up the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 2. Prepare TAWOS data

Download the [TAWOS v1.1 dataset](https://doi.org/10.5522/04/21308124) and import its SQL dump into MySQL. Set the connection details:

```bash
export TAWOS_HOST=127.0.0.1
export TAWOS_PORT=3306
export TAWOS_DATABASE=tawos
export TAWOS_USER=tawos
read -rs -p 'MySQL password: ' TAWOS_PASSWORD; echo
export TAWOS_PASSWORD
```

Export the issues and identify a project with sufficient labeled data:

```bash
python -m ps1.export_tawos --output data/raw/tawos_issues.csv
python -m ps1.audit_projects --csv data/raw/tawos_issues.csv
```

## 3. Run the experiment

```bash
python -m ps1.pipeline \
  --csv data/raw/tawos_issues.csv \
  --project DM \
  --config configs/default.yaml \
  --outdir runs
```


## 4. View the results

```bash
cat runs/DM/results/summary.json
cat runs/DM/results/cv_summary.csv
cat runs/DM/results/heldout_test_results.csv
ls runs/DM/results/plots/
```
