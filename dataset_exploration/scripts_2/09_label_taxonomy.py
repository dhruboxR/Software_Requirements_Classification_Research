"""Inspect source-specific label vocabulary, without silently merging labels."""
from diagnostics_utils import *


def main():
    df=dataset()
    t=(df.groupby([SOURCE_COL,'_label'],dropna=False)
       .agg(rows=(ID_COL,'size'),projects=(PROJECT_COL,'nunique'))
       .reset_index().rename(columns={SOURCE_COL:'source','_label':'label'}))
    totals=t.groupby('source')['rows'].transform('sum')
    t['share_within_source_pct']=(t['rows']/totals*100).round(3)
    save_csv(t.sort_values(['source','rows'],ascending=[True,False]),'09_source_label_taxonomy.csv')
    x=t.pivot_table(index='label',columns='source',values='rows',fill_value=0,aggfunc='sum').reset_index()
    save_csv(x,'09_label_source_matrix.csv')
    notes=[]
    for label,g in df.groupby('_label'):
        notes.append({'label':label,'total_rows':len(g),'sources':';'.join(sorted(g[SOURCE_COL].unique())),
                      'definition_verified':'NO', 'definition_source':'', 'proposed_parent':'',
                      'review_note':'Manual mapping required; do not infer equivalence from abbreviation'})
    save_csv(pd.DataFrame(notes),'09_taxonomy_review_template.csv')
    save_report('09_label_taxonomy_notes.txt',[
        f'Distinct observed nonblank labels: {int(df.loc[df["_label"]!="[BLANK]","_label"].nunique())}',
        'The 09_taxonomy_review_template.csv is for HUMAN REVIEW, not automatic labels.',
        'Verify definitions in dataset documentation and source files before mapping F/FR or NFR/specialized classes.',
        'Original labels and records were not altered.'])
    print(f'Saved taxonomy tables. Nonblank labels: {df.loc[df["_label"]!="[BLANK]","_label"].nunique()}')

if __name__=='__main__': main()
