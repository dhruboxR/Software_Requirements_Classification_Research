"""Compare main CSV schemas and counts; do not modify input files."""
import pandas as pd
from common import find_main_csv, load_main, save_csv, save_report, TEXT_COL, LABEL_COL, PROJECT_COL, SOURCE_COL, ID_COL


def main():
    summaries = []
    for stage in ('preprocessing', 'processing'):
        df = load_main(stage)
        path = find_main_csv(stage)
        summary = {'stage': stage, 'filename': path.name, 'rows': len(df),
                   'columns': len(df.columns),
                   'unique_requirements_ids': df[ID_COL].nunique(),
                   'unique_project_ids': df[PROJECT_COL].nunique(),
                   'unique_sources': df[SOURCE_COL].nunique(),
                   'unique_labels': df[LABEL_COL].nunique()}
        summaries.append(summary)
        print(f'\n{stage.upper()}: {len(df):,} rows, {len(df.columns)} columns')
        print('Columns:', list(df.columns))
        save_csv(pd.DataFrame({'position': range(1, len(df.columns)+1), 'column': df.columns,
                               'dtype': [str(d) for d in df.dtypes]}), f'01_schema_{stage}.csv')
    save_csv(pd.DataFrame(summaries), '01_dataset_overview.csv')
    a, b = (load_main(s) for s in ('preprocessing', 'processing'))
    common_cols = list(set(a.columns) & set(b.columns))
    left = a[common_cols].fillna('').astype(str)
    right = b[common_cols].fillna('').astype(str)
    same_ids = set(a[ID_COL]) == set(b[ID_COL])
    # Compare by requirement ID only when IDs are unique in each file.
    same_common_values = None
    if a[ID_COL].is_unique and b[ID_COL].is_unique and same_ids:
        aa = left.set_index(ID_COL).sort_index()
        bb = right.set_index(ID_COL).sort_index()
        same_common_values = aa.equals(bb[aa.columns])
    save_report('01_comparison.txt', [
        f'Rows: preprocessing={len(a)}, processing={len(b)}',
        f'Requirement ID sets equal: {same_ids}',
        f'IDs unique in both: {a[ID_COL].is_unique and b[ID_COL].is_unique}',
        f'Identical shared-column contents by ID: {same_common_values} (None = cannot safely test)',
        f'Extra processing columns: {sorted(set(b.columns)-set(a.columns))}',
        'Matching IDs do not, by themselves, prove source authenticity or annotation quality.'
    ])

if __name__ == '__main__':
    main()
