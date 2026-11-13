"""Missingness, duplicates, IDs, and conflicting labels in the selected main CSV."""
import pandas as pd
from common import load_main, normalize_text, save_csv, save_report, TEXT_COL, LABEL_COL, ID_COL


def main():
    df = load_main()
    clean = df.apply(lambda col: col.astype('string').fillna('').str.strip())
    missing = pd.DataFrame({'column': df.columns,
                            'blank_or_missing_count': [(clean[c] == '').sum() for c in df.columns]})
    missing['percentage'] = (100 * missing['blank_or_missing_count'] / max(len(df), 1)).round(2)
    save_csv(missing, '02_missing_values.csv')
    norm = clean[TEXT_COL].map(normalize_text)
    labels = clean[LABEL_COL]
    dup_exact_rows = int(clean.duplicated().sum())
    dup_ids = int(clean[ID_COL].duplicated().sum())
    dup_texts = int(norm.duplicated().sum())
    counts = pd.DataFrame({'normalized_text': norm, 'label': labels})
    grouped = counts[counts.normalized_text != ''].groupby('normalized_text').agg(
        record_count=('label', 'size'), distinct_labels=('label', 'nunique'),
        labels=('label', lambda x: ' | '.join(sorted(set(x))))).reset_index()
    conflicting = grouped[grouped.distinct_labels > 1].sort_values('record_count', ascending=False)
    repeated = grouped[grouped.record_count > 1].sort_values('record_count', ascending=False)
    save_csv(repeated, '02_repeated_texts.csv')
    save_csv(conflicting, '02_conflicting_text_labels.csv')
    save_report('02_data_quality_summary.txt', [
        f'Total records: {len(df)}', f'Exact duplicate rows: {dup_exact_rows}',
        f'Duplicate Requirement IDs (beyond first instance): {dup_ids}',
        f'Duplicate normalized requirement texts (beyond first instance): {dup_texts}',
        f'Distinct normalized texts with multiple labels: {len(conflicting)}',
        'Conflicting labels are flags for manual review, not proof of annotation mistakes.',
        'No records were removed or changed.'
    ])
    print(f'Rows={len(df)}, duplicate IDs={dup_ids}, duplicate texts={dup_texts}, conflicting texts={len(conflicting)}')

if __name__ == '__main__':
    main()
