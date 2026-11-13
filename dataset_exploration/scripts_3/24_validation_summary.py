from validation_common import *

needed=['19_source_class_support_summary.csv','20_per_class_duplicate_conflict_risk.csv',
        '21_per_class_project_coverage.csv','22_conflict_summary.csv','23_dataset_comparison_evidence.csv']
missing=[x for x in needed if not (TABLES/x).exists()]
if missing:
    raise RuntimeError('Run scripts 19-23 first. Missing: '+', '.join(missing))
comp=pd.read_csv(TABLES/'23_dataset_comparison_evidence.csv')
lines=['PHASE 1C: MINORITY-CLASS READINESS AUDIT','', 'Data source: original unified preprocessing CSV (unchanged)',
       'Dataset selection status: NOT YET APPROVED','',comp.to_string(index=False),'',
       'Outstanding validation gates:',
       '1. Define consistent label taxonomy and independently review conflicts for intended target.',
       '2. Verify near-duplicates and remove cross-partition contamination with frozen grouping.',
       '3. Confirm source-derived project identities where possible; reject unverified project split claims.',
       '4. Confirm minority class per-project coverage and minimum test support after grouping.',
       '5. Approve exact inclusion/exclusion and label decisions before generating any train/test split.',
       '6. Retain class-wise evaluation and uncertainty reporting; do not optimize for accuracy alone.',
       '', 'These checks do not predict model performance.']
report('24_phase_1c_validation_summary.txt',lines)
print('PHASE 1C COMPLETE: REVIEW REQUIRED; NO SOURCE RECORDS MODIFIED')
