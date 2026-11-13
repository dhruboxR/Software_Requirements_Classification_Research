from validation_common import *

df = get_data()
# Raw support vs unique text support. Neither is confirmed independent without provenance.
rows=[]
for (s, label), g in df.groupby(['source','label'], sort=True):
    n = len(g); unique = g['text_key'].nunique()
    rows.append(dict(source=s, label=label, rows=n, unique_normalized_texts=unique,
                     duplicate_extra=n-unique, raw_source_fraction_pct=round(100*n/len(df[df.source==s]),3),
                     below_30_unique=unique<30, below_100_unique=unique<100,
                     screening_status='NO_LABEL' if label=='[BLANK]' else ('LOW_SUPPORT' if unique<30 else 'REVIEW')))
out = pd.DataFrame(rows).sort_values(['source','rows'],ascending=[True,False])
save(out, '19_per_class_unique_support.csv')
summary=out.groupby('source').agg(classes=('label',lambda x:(x!='[BLANK]').sum()),
                                  classes_under_30=('below_30_unique','sum'),
                                  classes_under_100=('below_100_unique','sum'),
                                  min_unique_class_support=('unique_normalized_texts',lambda x:int(x.min()))).reset_index()
save(summary,'19_source_class_support_summary.csv')
print(summary.to_string(index=False))
report('19_class_support_notes.txt',['Unique normalized text counts are a necessary but not sufficient proxy for independent examples.',
  'Do not infer model performance or merge labels solely based on counts.','Class sizes are calculated separately for each source.'])
