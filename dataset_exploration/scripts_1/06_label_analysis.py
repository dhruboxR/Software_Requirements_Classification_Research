"""Create systematic label review samples; no automated relabeling."""
import pandas as pd
from common import load_main, normalize_text, save_csv, save_report, LABEL_COL, TEXT_COL, ID_COL, SOURCE_COL, PROJECT_COL


def main():
    df = load_main()
    df = df.copy()
    df['_normalized'] = df[TEXT_COL].map(normalize_text)
    label_summary = df.groupby(LABEL_COL, dropna=False).agg(
        count=(ID_COL,'size'), projects=(PROJECT_COL,'nunique'), sources=(SOURCE_COL,'nunique')).reset_index()
    save_csv(label_summary.sort_values('count', ascending=False), '06_label_coverage.csv')
    # Deterministic examples drawn from each label for human annotation review.
    samples = []
    for label, group in df.groupby(LABEL_COL, dropna=False, sort=True):
        s = group.sample(n=min(8,len(group)), random_state=42)
        samples.append(s[[ID_COL, PROJECT_COL, SOURCE_COL, LABEL_COL, TEXT_COL]])
    samples_df = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
    save_csv(samples_df, '06_label_review_samples.csv')
    conflict_keys = (df[df['_normalized'] != ''].groupby('_normalized')[LABEL_COL].nunique())
    conflicting = df[df['_normalized'].isin(conflict_keys[conflict_keys > 1].index)]
    save_csv(conflicting[[ID_COL, PROJECT_COL, SOURCE_COL, LABEL_COL, TEXT_COL]], '06_conflict_review_rows.csv')
    save_report('06_label_review_instructions.txt', [
        'Review 06_label_review_samples.csv against official label definitions.',
        'Review 06_conflict_review_rows.csv for same normalized text with different class labels.',
        'A disagreement flag does not prove an incorrect label; context and annotation policy matter.',
        'No examples were corrected, filtered, or relabeled by this script.'
    ])
    print(f'Sampled {len(samples_df)} rows for manual label review; {len(conflicting)} conflict-related rows.')

if __name__ == '__main__':
    main()
