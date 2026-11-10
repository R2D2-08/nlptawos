# BERT-Based Word Embeddings for Within-Project Software Story Point Estimation

**Project Statement 1 (PS-1) | Natural Language Processing**

**Group:** Group 2  
**Members:** Aditya Sunil Menon, Athul Anil Kumar, Johan John Koshy, Kiran Nambiar, Mohammed Omar  
**Project data:** DM (TAWOS v1.1)

Run from the repository root using Python 3.10+ and MySQL 8.


## 1. The problem at hand

How effectively do BERT-based contextual token embeddings predict story point values from the title and description of an issue, when training and testing use distinct issues from the same project under a random split? This is an example of supervised numerical regression.

## 2. Methodology, Data collection and preprocessing

Dataset: Tawosi et al., TAWOS v1.1 (DOI 10.5522/04/21308124). The official database is MySQL; the extractor joins Issue and Project, using Title and Description_Text as input and Story_Point as the target. Issues explicitly changed after estimation, empty texts, invalid labels, and repeated exact descriptions are filtered. Unknown change flags remain a potential limitation.

| Audit stage | Count |
| --- | --- |
| Source issues with project key | 20065 |
| Positive numeric labels | 20065 |
| After explicit edit filtering | 14365 |
| Final distinct cleaned issues | 14061 |
| Development data (80%) | 11248 |
| Final held-out test (20%) | 2813 |

# Reproducing the experiments:

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
