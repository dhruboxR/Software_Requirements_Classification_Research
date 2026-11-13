from validation_common import *

df=get_data(); rows=[]
# One text can appear in multiple sources/classes; keep these dimensions explicit.
bykey=df.groupby('text_key').agg(n_rows=(ID_COL,'size'),n_sources=('source','nunique'),
                                 n_labels=('label','nunique')).reset_index()
df=df.merge(bykey,on='text_key',validate='many_to_one')
for (s,l),g in df.groupby(['source','label'],sort=True):
    rows.append(dict(source=s,label=l,rows=len(g),unique_texts=g.text_key.nunique(),
        rows_repeated_within_source=int(g.duplicated('text_key',keep=False).sum()),
        rows_repeated_globally=int((g.n_rows>1).sum()),
        rows_cross_source=int((g.n_sources>1).sum()),
        rows_with_global_label_conflict=int((g.n_labels>1).sum())))
save(pd.DataFrame(rows),'20_per_class_duplicate_conflict_risk.csv')
# Leakage grouping keys must be global across source labels.
save(bykey[bykey.n_rows>1].sort_values('n_rows',ascending=False), '20_repeated_text_group_index.csv')
print('Global repeated-text groups:',int((bykey.n_rows>1).sum()))
print('Cross-source text groups:',int((bykey.n_sources>1).sum()))
print('Conflicting text groups:',int((bykey.n_labels>1).sum()))
report('20_duplicate_risk_notes.txt',['Global conflicting-label groups include mixed taxonomy comparisons that may not be annotation errors.',
    'Repeated texts must not cross training/evaluation partitions if all sources are combined.',
    'No duplicate was removed; every count reflects the raw combined file.'])
