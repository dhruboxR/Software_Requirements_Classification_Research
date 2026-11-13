from validation_common import *

df=get_data()
# Descriptive sample of same text with different labels and provenance.
g=df.groupby('text_key').agg(n=('label','size'),labels=('label',lambda x:'|'.join(sorted(set(x)))),
                              sources=('source',lambda x:'|'.join(sorted(set(x)))),
                              n_labels=('label','nunique'),n_sources=('source','nunique'),
                              sample_text=(TEXT_COL,'first')).reset_index()
conf=g[g.n_labels>1].copy().sort_values(['n_sources','n'],ascending=False)
save(conf,'22_conflict_groups_source_context.csv')
# Sample records for human review, deliberately no automatic change to labels.
review=df[df.text_key.isin(set(conf.text_key))][[ID_COL,'source','label',PROJECT_COL,'Project Name',TEXT_COL,'text_key']]
save(review,'22_conflict_manual_review_rows.csv')
summary=pd.DataFrame([dict(global_conflict_groups=len(conf), cross_source_conflict_groups=int((conf.n_sources>1).sum()),
                       single_source_conflict_groups=int((conf.n_sources==1).sum()),
                       affected_rows=len(review))])
save(summary,'22_conflict_summary.csv')
print(summary.to_string(index=False))
report('22_taxonomy_cautions.txt',['Classes F and FR may express similar labels, but mapping is NOT automatic.',
 'NSR in D3 pools functionality, performance, usability, and portability; do not equate it to NFR.',
 'Cross-source conflicts may be taxonomy changes rather than erroneous annotation.',
 'Human review should document definition, decision, reason, and original source.'])
