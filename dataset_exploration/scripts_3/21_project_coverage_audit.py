from validation_common import *

df=get_data()
recovery=TABLES/'21_d3_original_project_recovery.csv'
if recovery.exists():
    r=pd.read_csv(recovery,dtype='string',keep_default_na=False)
    required={'Requirement ID','Original Project IDs','Mapping Status'}
    if not required.issubset(r.columns):
        raise ValueError(f'Unexpected recovery columns: {r.columns.tolist()}')
    valid=r[r['Mapping Status'].eq('unique_original_project')].copy()
    valid=valid.drop_duplicates('Requirement ID')
    valid=valid[['Requirement ID','Original Project IDs']].rename(columns={'Requirement ID':ID_COL,'Original Project IDs':'verified_original_project'})
    df=df.merge(valid,on=ID_COL,how='left',validate='one_to_one')
else:
    print('WARNING: D3 project recovery table missing; cannot evaluate verified project support')
    df['verified_original_project']=pd.NA
rows=[]
for (s,label),g in df.groupby(['source','label'],sort=True):
    verified=g.dropna(subset=['verified_original_project'])
    verified=verified[verified['verified_original_project'].astype(str).str.strip().ne('')]
    # Only source D3 has original project IDs recovered, and matching isn't absolute verification.
    group_sizes=verified.groupby('verified_original_project').size()
    rows.append(dict(source=s,label=label,rows=len(g),unique_texts=g.text_key.nunique(),
        source_matched_project_rows=len(verified) if s=='D3' else 0,
        recoverable_source_projects=verified.verified_original_project.nunique() if s=='D3' else 0,
        max_project_share_pct=round(group_sizes.max()/len(verified)*100,2) if s=='D3' and len(verified) else '',
        project_group_status='PROVISIONAL_DOSSPRE_TEXT_MATCH' if s=='D3' else 'UNVERIFIED',
        split_readiness='NOT_APPROVED'))
out=pd.DataFrame(rows)
save(out,'21_per_class_project_coverage.csv')
print(out[out.source=='D3'].to_string(index=False))
report('21_group_feasibility_notes.txt',['Recovered D3 IDs are inferred from normalized text matches to DOSSPRE.',
    'Unmatched and multiple-original-project D3 records are not assigned groups.',
    'Do not use D2 combined Project ID alone as a project-aware split key.',
    'Do not approve any split until group overlap and fold-specific class support have been validated.'])
