"""Evidence-driven D1-D5 mapping. Do not assume source IDs from filenames."""
from diagnostics_utils import *


def main():
    df=dataset()
    rows=[]
    for source,g in df.groupby(SOURCE_COL,sort=True):
        counts=g['_label'].value_counts()
        rows.append({'source_id':source,'rows':len(g),'projects':g[PROJECT_COL].nunique(),
                     'labels':g['_label'].nunique(), 'top_label':counts.index[0],
                     'top_label_count':int(counts.iloc[0]),
                     'top_label_fraction':round(float(counts.iloc[0]/len(g)),5),
                     'example_project_names':' | '.join(sorted(set(g.get('Project Name',pd.Series(dtype=str)).fillna('').astype(str)))[:5]) if 'Project Name' in g else ''})
    save_csv(pd.DataFrame(rows),'08_source_profiles.csv')
    matches=[]
    for ref in get_reference_sets():
        for source,g in df.groupby(SOURCE_COL,sort=True):
            flag=g['_norm'].isin(ref['texts']) & g['_norm'].ne('') if ref['texts'] else pd.Series(False,index=g.index)
            matches.append({'source_id':source,'original_candidate':ref['name'],
                            'reference_texts':len(ref['texts']), 'main_rows':len(g),
                            'matching_rows':int(flag.sum()), 'matching_unique_texts':int(g.loc[flag,'_norm'].nunique()),
                            'match_share_of_main':round(float(flag.mean()),5),
                            'extraction_status':ref['status'], 'extraction_notes':ref['detail']})
    matrix=pd.DataFrame(matches)
    save_csv(matrix,'08_source_match_evidence.csv')
    save_report('08_source_mapping_notes.txt',[
        'Source IDs D1-D5 are NOT automatically mapped to named original datasets.',
        'An exact-text match is evidence of shared text, not proof of origin or completeness.',
        'Source-file labels may be transformed and match rates may be low.',
        'PROMISE_exp.txt is not parsed until its schema is verified.',
        'Some sources (especially Zenodo) are not represented by a directly parsed reference file.',
        'Review 08_source_profiles.csv alongside 08_source_match_evidence.csv and original metadata.',
        'No original files were changed.'])
    print(matrix[['source_id','original_candidate','matching_rows','extraction_status']].to_string(index=False))

if __name__=='__main__': main()
