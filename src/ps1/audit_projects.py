"""Count usable story-point labels by source project to choose a study target."""
import argparse
import pandas as pd
from .prepare import normalise_columns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    args = ap.parse_args()
    df = normalise_columns(pd.read_csv(args.csv, low_memory=False))
    if not {'project_key', 'story_point', 'title', 'description_text'}.issubset(df.columns):
        raise ValueError('Expected project_key, story_point, title, description_text fields')
    df['story_point'] = pd.to_numeric(df['story_point'], errors='coerce')
    df = df[df['story_point'].gt(0)]
    print('Labeled issues by project, BEFORE dedup/post-estimate-edit filtering:')
    print(df.groupby('project_key').size().sort_values(ascending=False).to_string())
    print('\nChoose ONE project with ample labeled issues, then run pipeline --project THAT_KEY.')


if __name__ == '__main__':
    main()
