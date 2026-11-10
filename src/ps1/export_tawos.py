"""Export the real TAWOS v1.1 MySQL Issue + Project schema to one CSV.

TAWOS download: https://doi.org/10.5522/04/21308124
Original schema: https://github.com/SOLAR-group/TAWOS
Requires an imported MySQL database and environment credentials.
"""
import argparse
import csv
import os
from pathlib import Path

FIELDS = [
    "issue_key", "project_key", "title", "description_text", "story_point",
    "creation_date", "title_changed_after_estimation",
    "description_changed_after_estimation", "story_point_changed_after_estimation",
]

QUERY = """
SELECT i.Issue_Key, p.Project_Key, i.Title, i.Description_Text,
       i.Story_Point, i.Creation_Date,
       i.Title_Changed_After_Estimation,
       i.Description_Changed_After_Estimation,
       i.Story_Point_Changed_After_Estimation
FROM Issue AS i
JOIN Project AS p ON p.ID = i.Project_ID
WHERE i.Story_Point IS NOT NULL AND i.Story_Point > 0
  AND (%s IS NULL OR p.Project_Key = %s)
ORDER BY p.Project_Key, i.Issue_Key
"""


def export_csv(out_path, project_key=None):
    try:
        import pymysql
    except ImportError as e:
        raise RuntimeError("Install requirements first: pip install -e .") from e
    database = os.getenv("TAWOS_DATABASE", "TAWOS")
    conn = pymysql.connect(
        host=os.getenv("TAWOS_HOST", "127.0.0.1"),
        port=int(os.getenv("TAWOS_PORT", "3306")),
        user=os.getenv("TAWOS_USER", "root"),
        password=os.getenv("TAWOS_PASSWORD", ""),
        database=database,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.SSCursor,
        read_timeout=1800,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        with conn.cursor() as cursor, out_path.open("w", newline="", encoding="utf-8") as f:
            cursor.execute(QUERY, (project_key, project_key))
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            while True:
                rows = cursor.fetchmany(2000)
                if not rows:
                    break
                for row in rows:
                    writer.writerow(dict(zip(FIELDS, row)))
                    count += 1
    finally:
        conn.close()
    print(f"Exported {count} labeled issues to {out_path}")
    if count == 0:
        raise ValueError("No labeled issues found; confirm TAWOS import and project key")


def main():
    p = argparse.ArgumentParser(description="Export real TAWOS SQL dataset as CSV")
    p.add_argument("--output", default="data/raw/tawos_issues.csv")
    p.add_argument("--project", default=None, help="Optional project key, e.g. MESOS; omitted = all 39 projects")
    args = p.parse_args()
    export_csv(args.output, args.project)


if __name__ == "__main__":
    main()
