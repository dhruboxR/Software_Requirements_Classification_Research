"""Project/source distribution and project-class counts."""
import pandas as pd
from common import load_main, save_csv, save_report, PROJECT_COL, SOURCE_COL, LABEL_COL


def main():
    df = load_main()
    for col, name in ((PROJECT_COL, 'project'), (SOURCE_COL, 'source')):
        counts = df[col].fillna('').str.strip().replace('', '[BLANK]').value_counts()
        tab = counts.rename_axis(name).reset_index(name='count')
        tab['percentage'] = (100*tab['count']/max(1, len(df))).round(3)
        save_csv(tab, f'04_{name}_distribution.csv')
    pairs = df.groupby([SOURCE_COL, PROJECT_COL], dropna=False).size().reset_index(name='count')
    save_csv(pairs, '04_source_project_counts.csv')
    project_classes = df.groupby([PROJECT_COL, LABEL_COL], dropna=False).size().reset_index(name='count')
    save_csv(project_classes, '04_project_class_counts.csv')
    project_source = df.groupby(PROJECT_COL, dropna=False)[SOURCE_COL].nunique().reset_index(name='distinct_sources')
    save_csv(project_source[project_source.distinct_sources > 1], '04_projects_with_multiple_sources.csv')
    save_report('04_project_summary.txt', [f'Records: {len(df)}',
                f'Unique project IDs: {df[PROJECT_COL].nunique()}',
                f'Unique source identifiers: {df[SOURCE_COL].nunique()}',
                f'Project IDs associated with multiple sources: {int((project_source.distinct_sources > 1).sum())}',
                'Project IDs are not guaranteed to represent independent real-world projects.'])
    print('Project and source counts saved.')

if __name__ == '__main__':
    main()
