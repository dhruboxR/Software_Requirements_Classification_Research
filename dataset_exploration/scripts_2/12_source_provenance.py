"""Read-only checks against identifiable original sources; clearly mark untested sources."""
from diagnostics_utils import *


def main():
    df=dataset()
    references=get_reference_sets()
    status=[]
    flagged=df[[ID_COL,PROJECT_COL,SOURCE_COL,LABEL_COL,TEXT_COL]].copy()
    for item in references:
        key='candidate_match_'+item['name']
        flagged[key]=df['_norm'].isin(item['texts']) & df['_norm'].ne('') if item['texts'] else False
        status.append({'reference':item['name'], 'path':str(item['path']) if item['path'] else '',
                       'parse_status':item['status'], 'reference_unique_texts':len(item['texts']),
                       'matched_combined_rows':int(flagged[key].sum()),
                       'matched_unique_texts':int(df.loc[flagged[key],'_norm'].nunique()),
                       'notes':item['detail']})
    save_csv(pd.DataFrame(status),'12_reference_extraction_status.csv')
    save_csv(flagged,'12_candidate_provenance_matches.csv')
    counts=[]
    for sid,g in df.groupby(SOURCE_COL):
        row={'source_id':sid,'total_rows':len(g)}
        for item in references:
            row[item['name']+'_matched_rows']=int((g['_norm'].isin(item['texts']) & g['_norm'].ne('')).sum()) if item['texts'] else 0
        counts.append(row)
    save_csv(pd.DataFrame(counts),'12_matches_by_source.csv')
    save_report('12_provenance_limitations.txt',[
        'Candidate exact normalized-text matches do not prove definitive dataset origin.',
        'No exact match does not prove independence, especially after preprocessing or rewording.',
        'PROMISE_exp.txt is deliberately NOT parsed until its record format is inspected.',
        'Zenodo SRS and associated spreadsheets are NOT automatically text-matched here.',
        'FR_NFR and DOSSPRE extraction relies on recognized text-column headers; inspect extraction status.',
        'Combined source identifiers require original authors documentation for verified mappings.',
        'Do not treat this output as a complete near-duplicate or data-leakage analysis.',
        'No original files were modified.'])
    print(pd.DataFrame(status)[['reference','parse_status','reference_unique_texts','matched_combined_rows']].to_string(index=False))

if __name__=='__main__': main()
