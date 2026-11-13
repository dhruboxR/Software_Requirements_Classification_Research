from validation_common import *

df=get_data()
rows=[]
for s,g in df.groupby('source',sort=True):
    counts=g[g.label!='[BLANK]'].groupby('label').text_key.nunique()
    rows.append(dict(source=s,rows=len(g),nonblank_classes=len(counts),
        smallest_unique_class=int(counts.min()) if len(counts) else 0,
        minority_classes_under_30=int((counts<30).sum()),
        minority_classes_under_100=int((counts<100).sum()),
        blank_labels=int((g.label=='[BLANK]').sum()),
        within_source_extra_duplicate_rows=int(len(g)-g.text_key.nunique()),
        claimed_independent_projects='NO' if s!='D3' else 'PARTIALLY_RECOVERABLE',
        risk_note={
          'D1':'Major PROMISE overlap; mixed labels',
          'D2':'Project metadata unreliable; FR/NFR labels need source review',
          'D3':'Small minority classes; NSR taxonomy compression; original project recovery incomplete',
          'D4':'LLM-associated requirements; repeated texts; 90:10 imbalance',
          'D5':'70 rows; substantial PROMISE/D1 overlap; project label unreliable'
        }.get(s,'Unreviewed'),
        selection_status='NOT APPROVED'))
out=pd.DataFrame(rows)
save(out,'23_dataset_comparison_evidence.csv')
print(out.to_string(index=False))
report('23_selection_notes.txt',['No datasets have been approved for modeling.',
  'This is a preliminary assessment, not a ranking of expected model accuracy.',
  'Minority-class counts must be interpreted after provenance, label review, and grouping.',
  'If any task has inadequate independent minority-class examples, evaluate other datasets rather than oversample into a misleading benchmark.'])
