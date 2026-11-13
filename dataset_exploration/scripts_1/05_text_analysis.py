"""Text length statistics; lightweight tokenization for descriptive analysis only."""
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import load_main, save_csv, save_report, FIGURES, TEXT_COL, LABEL_COL


def main():
    df = load_main()
    texts = df[TEXT_COL].fillna('').astype(str)
    words = texts.map(lambda t: len(re.findall(r'\b\w+\b', t, flags=re.UNICODE)))
    chars = texts.map(len)
    stats = pd.DataFrame({'metric': ['words','characters'],
                          'min':[words.min(),chars.min()],
                          'mean':[words.mean(),chars.mean()],
                          'median':[words.median(),chars.median()],
                          'p90':[words.quantile(.9),chars.quantile(.9)],
                          'max':[words.max(),chars.max()]})
    save_csv(stats, '05_text_length_summary.csv')
    per_class = pd.DataFrame({'class':df[LABEL_COL], 'words':words, 'characters':chars}).groupby('class',dropna=False).agg(
        count=('words','size'),mean_words=('words','mean'),median_words=('words','median'),
        mean_characters=('characters','mean')).reset_index().round(2)
    save_csv(per_class, '05_length_by_class.csv')
    fig, ax = plt.subplots(figsize=(9,5))
    ax.hist(words, bins=40)
    ax.set(title='Requirement word-count distribution', xlabel='Word count', ylabel='Requirements')
    fig.tight_layout()
    fig.savefig(FIGURES / '05_word_count_histogram.png', dpi=180)
    plt.close(fig)
    save_report('05_text_summary.txt', [f'Records: {len(df)}',
                f'Mean words: {words.mean():.2f}', f'Median words: {words.median():.2f}',
                f'Max words: {words.max()}',
                'Word counts use a Unicode regular expression and may differ from model tokenizer lengths.'])
    print(stats.to_string(index=False))

if __name__ == '__main__':
    main()
