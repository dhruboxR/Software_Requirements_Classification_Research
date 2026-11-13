"""Class counts and class imbalance summary."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import load_main, save_csv, save_report, FIGURES, LABEL_COL


def main():
    df = load_main()
    labels = df[LABEL_COL].fillna('').str.strip().replace('', '[BLANK]')
    counts = labels.value_counts(dropna=False)
    out = counts.rename_axis('class').reset_index(name='count')
    out['percentage'] = (100 * out['count'] / max(len(df), 1)).round(3)
    save_csv(out, '03_class_distribution.csv')
    if not out.empty:
        fig, ax = plt.subplots(figsize=(max(8, min(17, len(out)*0.55)), 6))
        ax.bar(out['class'], out['count'])
        ax.set(title='Requirements by class', xlabel='Class', ylabel='Number of requirements')
        ax.tick_params(axis='x', labelrotation=65)
        fig.tight_layout()
        fig.savefig(FIGURES / '03_class_distribution.png', dpi=180)
        plt.close(fig)
    smallest = int(counts.min()) if len(counts) else 0
    largest = int(counts.max()) if len(counts) else 0
    save_report('03_class_summary.txt', [f'Records: {len(df)}', f'Classes including blank: {len(counts)}',
                f'Majority class size: {largest}', f'Minority class size: {smallest}',
                f'Majority/minority ratio: {largest/smallest:.2f}' if smallest else 'Majority/minority ratio: undefined'])
    print(out.to_string(index=False))

if __name__ == '__main__':
    main()
