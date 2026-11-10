"""Prepare *one project* without target leakage or cross-project mixing."""
import argparse
import re
from pathlib import Path
import pandas as pd
import numpy as np
from .common import save_json

COL_ALIASES = {
    'issue_key': ['issue_key', 'issuekey', 'key'],
    'project_key': ['project_key', 'project', 'projectkey'],
    'title': ['title', 'summary'],
    'description_text': ['description_text', 'description'],
    'story_point': ['story_point', 'storypoint', 'storypoints', 'story_points'],
    'creation_date': ['creation_date', 'created', 'created_at'],
}
FLAG_COLS = [
    'title_changed_after_estimation',
    'description_changed_after_estimation',
    'story_point_changed_after_estimation',
]


def normalise_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
    renames = {}
    for canonical, choices in COL_ALIASES.items():
        if canonical not in df.columns:
            for alias in choices:
                if alias in df.columns:
                    renames[alias] = canonical
                    break
    return df.rename(columns=renames)


def clean_text(s):
    return re.sub(r'\s+', ' ', str(s or '')).strip()


def prepare_file(input_csv, output_csv, project=None, stats_json=None):
    df = normalise_columns(pd.read_csv(input_csv, low_memory=False))
    for field in ['title', 'description_text', 'story_point']:
        if field not in df.columns:
            raise ValueError(f'Missing CSV column: {field}')
    if project:
        if 'project_key' not in df.columns:
            raise ValueError('Expected a project_key column to filter within-project')
        df = df[df['project_key'].astype(str).str.upper() == project.upper()].copy()
    elif 'project_key' in df.columns:
        projects = df['project_key'].dropna().unique()
        if len(projects) != 1:
            raise ValueError(f'Multiple projects found ({len(projects)}). Specify --project; PS-1 is WITHIN-project.')
        project = str(projects[0])
    else:
        raise ValueError('CSV must identify project_key; cannot establish within-project validity')
    start = len(df)
    if start < 10:
        raise ValueError(f'Too few issues in project {project}: {start}')
    df['story_point'] = pd.to_numeric(df['story_point'], errors='coerce')
    df = df[np.isfinite(df['story_point']) & (df['story_point'] > 0)].copy()
    after_labels = len(df)
    present_flags = [c for c in FLAG_COLS if c in df.columns]
    # Exclude explicitly edited-after-estimation examples. Unknown (NULL)
    # statuses are kept but counted and documented as a limitation.
    missing_flag_counts = {}
    for c in present_flags:
        flags = pd.to_numeric(df[c], errors='coerce')
        missing_flag_counts[c] = int(flags.isna().sum())
        df = df[flags.ne(1)].copy()
    after_flags = len(df)
    df['title'] = df['title'].fillna('').map(clean_text)
    df['description_text'] = df['description_text'].fillna('').map(clean_text)
    df['text'] = (df['title'] + '. ' + df['description_text']).map(lambda s: s.strip(' .'))
    df = df[df['text'].str.len() >= 5].copy()
    after_text = len(df)
    if 'issue_key' not in df.columns:
        raise ValueError('CSV must have issue_key to audit and prevent repeated-issue leakage')
    df['issue_key'] = df['issue_key'].astype(str)
    df = df.drop_duplicates(subset=['issue_key'], keep='first')
    after_ids = len(df)
    # If identical description has inconsistent targets, drop *all* instances.
    df['norm_text'] = df['text'].str.lower().str.replace(r'\s+', ' ', regex=True)
    conflicts = df.groupby('norm_text')['story_point'].nunique()
    conflicted = set(conflicts.index[conflicts.gt(1)])
    df = df[~df['norm_text'].isin(conflicted)].copy()
    conflicts_removed = after_ids - len(df)
    df = df.drop_duplicates(subset=['norm_text'], keep='first').copy()
    source_missing_flags = [c for c in FLAG_COLS if c not in df.columns]
    df = df[['issue_key', 'project_key', 'title', 'description_text', 'text', 'story_point']].reset_index(drop=True)
    if len(df) < 30:
        raise ValueError(f'Only {len(df)} clean issues remain for {project}; choose project with enough labeled data')
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    stats = {
        'project': str(project).upper(), 'raw_project_issues': start,
        'with_positive_numeric_story_points': after_labels,
        'after_excluding_explicit_post_estimate_edits': after_flags,
        'nonempty_text': after_text, 'unique_issue_ids': after_ids,
        'conflicting_text_rows_dropped': conflicts_removed,
        'final_unique_issues': len(df),
        'story_points_median': float(df['story_point'].median()),
        'story_points_min': float(df['story_point'].min()),
        'story_points_max': float(df['story_point'].max()),
        'missing_edit_flags': source_missing_flags,
        'unknown_flag_value_counts': missing_flag_counts,
        'warning': 'Unknown/missing edit flags are not assumed safe; discuss possible post-estimation text leakage.',
    }
    if stats_json:
        save_json(stats_json, stats)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--project', required=True)
    ap.add_argument('--out', default='data/processed/issues.csv')
    args = ap.parse_args()
    print(prepare_file(args.csv, args.out, args.project, str(Path(args.out).with_suffix('.stats.json'))))


if __name__ == '__main__':
    main()
