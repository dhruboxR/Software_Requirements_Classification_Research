"""Examine duplicate normalized texts, same vs different projects/sources."""
from diagnostics_utils import *


def main():
    df=dataset()
    df=df[df['_norm']!=''].copy()
    groups=[]
    for norm,g in df.groupby('_norm',sort=False):
        if len(g)<2: continue
        groups.append({'normalized_text':norm,'rows':len(g),
                       'extra_occurrences':len(g)-1,
                       'unique_labels':g['_label'].nunique(),
                       'sources':g[SOURCE_COL].nunique(),
                       'projects':g[PROJECT_COL].nunique(),
                       'same_source_only':g[SOURCE_COL].nunique()==1,
                       'across_projects':g[PROJECT_COL].nunique()>1,
                       'across_sources':g[SOURCE_COL].nunique()>1,
                       'source_ids':' | '.join(sorted(g[SOURCE_COL].unique())),
                       'labels':' | '.join(sorted(g['_label'].unique()))})
    out=pd.DataFrame(groups)
    if not out.empty: out=out.sort_values(['rows','normalized_text'],ascending=[False,True])
    save_csv(out,'11_duplicate_groups.csv')
    repeated_keys=set(out['normalized_text']) if not out.empty else set()
    matches=df[df['_norm'].isin(repeated_keys)]
    save_csv(matches[[ID_COL,PROJECT_COL,SOURCE_COL,LABEL_COL,TEXT_COL,'_norm']].rename(columns={'_norm':'normalized_text'}),'11_duplicate_records.csv')
    save_report('11_duplicate_summary.txt',[
        f'Duplicate normalized-text groups: {len(out)}',
        f'Extra duplicate text occurrences: {int(out.extra_occurrences.sum()) if len(out) else 0}',
        f'Rows participating in duplicated text groups: {len(matches)}',
        f'Groups across sources: {int(out.across_sources.sum()) if len(out) else 0}',
        f'Groups across projects: {int(out.across_projects.sum()) if len(out) else 0}',
        f'Duplicate groups with label conflicts: {int((out.unique_labels>1).sum()) if len(out) else 0}',
        'Duplicate counts use Unicode NFKC, whitespace normalization, and casefold.',
        'These checks do not identify semantic paraphrases.',
        'No records were removed.'])
    print(f'Duplicate groups={len(out)}, extra occurrences={int(out.extra_occurrences.sum()) if len(out) else 0}')

if __name__=='__main__': main()
