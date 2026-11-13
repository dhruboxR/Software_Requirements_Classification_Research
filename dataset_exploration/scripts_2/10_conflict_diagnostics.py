"""Characterize different labels assigned to the same normalized text."""
from diagnostics_utils import *


def main():
    df=dataset()
    g=df[df['_norm']!=''].groupby('_norm',sort=False)
    nunique=g['_label'].nunique()
    keys=nunique[nunique>1].index
    conflict=df[df['_norm'].isin(keys)].copy()
    if conflict.empty:
        save_csv(pd.DataFrame(columns=['normalized_text','rows','labels','sources','projects','cross_source','within_source_conflict']), '10_conflict_groups.csv')
        save_report('10_conflict_summary.txt',['No conflicting label groups detected.'])
        print('No conflicts detected'); return
    rows=[]
    for norm,z in conflict.groupby('_norm',sort=False):
        within=any(q['_label'].nunique()>1 for _,q in z.groupby(SOURCE_COL))
        rows.append({'normalized_text':norm,'rows':len(z),
                     'labels':' | '.join(sorted(z['_label'].unique())),
                     'sources':' | '.join(sorted(z[SOURCE_COL].unique())),
                     'projects':z[PROJECT_COL].nunique(),
                     'cross_source':z[SOURCE_COL].nunique()>1,
                     'within_source_conflict':within})
    summary=pd.DataFrame(rows).sort_values(['rows','normalized_text'],ascending=[False,True])
    save_csv(summary,'10_conflict_groups.csv')
    save_csv(conflict[[ID_COL,PROJECT_COL,SOURCE_COL,LABEL_COL,TEXT_COL,'_norm']].rename(columns={'_norm':'normalized_text'}),'10_conflict_records.csv')
    pairs=(conflict.groupby([SOURCE_COL,'_label']).size().reset_index(name='conflict_rows'))
    save_csv(pairs,'10_conflict_source_label_counts.csv')
    save_report('10_conflict_summary.txt',[
        f'Conflicting normalized-text groups: {len(summary)}',
        f'Rows in conflicting groups: {len(conflict)}',
        f'Groups involving multiple sources: {int(summary.cross_source.sum())}',
        f'Groups with within-source label conflict: {int(summary.within_source_conflict.sum())}',
        'The same text with different labels is a review flag, not proof of mislabeling.',
        'Blank labels count as a distinct label in this audit.',
        'No labels were changed or removed.'])
    print(f'Conflicting groups={len(summary)}, rows={len(conflict)}, cross-source groups={int(summary.cross_source.sum())}')

if __name__=='__main__': main()
