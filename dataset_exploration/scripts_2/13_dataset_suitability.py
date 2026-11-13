"""Summarize measurable concerns and outstanding decisions; no model/task chosen."""
from diagnostics_utils import *


def main():
    df=dataset()
    n=len(df); norm=df['_norm']; labels=df['_label']; source=df[SOURCE_COL]
    nonempty=df[norm!='']
    grouped=nonempty.groupby('_norm')['_label']
    conflict_keys=set(grouped.nunique().loc[lambda x:x>1].index)
    minority=labels.value_counts()
    src=source.value_counts()
    indicators=[
        ('rows',n),('project_ids',df[PROJECT_COL].nunique()),
        ('sources',source.nunique()),('nonblank_labels',labels[labels!='[BLANK]'].nunique()),
        ('blank_text_rows',int((norm=='').sum())),
        ('blank_label_rows',int((labels=='[BLANK]').sum())),
        ('extra_duplicate_text_occurrences',int(nonempty['_norm'].duplicated().sum())),
        ('conflicting_normalized_text_groups',len(conflict_keys)),
        ('conflict_related_rows',int(norm.isin(conflict_keys).sum())),
        ('largest_source_share_pct',round(float(src.iloc[0]/n*100),3) if n else 0),
        ('largest_label_share_pct',round(float(minority.iloc[0]/n*100),3) if n else 0),
        ('nonblank_labels_under_30_records',int((minority.drop(index='[BLANK]',errors='ignore')<30).sum())),
    ]
    save_csv(pd.DataFrame(indicators,columns=['indicator','value']),'13_suitability_indicators.csv')
    tasks=pd.DataFrame([
        {'possible_task':'Binary FR vs NFR','status':'NOT DECIDED','required_validation':'Verify definitions and provenance of FR/NFR; determine treatment of F, NSR and other labels'},
        {'possible_task':'Within-source fine-grained classification','status':'NOT DECIDED','required_validation':'Review source-specific taxonomy, minority classes, groups and original documents'},
        {'possible_task':'Unified hierarchical classification','status':'NOT DECIDED','required_validation':'Require verified cross-source label mapping; audit duplicates, conflicts and lineage'},
    ])
    save_csv(tasks,'13_candidate_tasks_review.csv')
    report=['PHASE 1 DATASET SUITABILITY: NOT YET APPROVED FOR MODELING','']
    report.extend(f'{k}: {v}' for k,v in indicators)
    report.extend(['','Decision gates:',
        '[ ] Verify original meanings of D1-D5 against publisher documentation',
        '[ ] Verify labels using authoritative definitions, not abbreviations alone',
        '[ ] Audit conflicting annotations and blank labels',
        '[ ] Verify PROMISE source format and source overlap, including transformed texts',
        '[ ] Decide target label taxonomy and inclusion/exclusion with a written protocol',
        '[ ] Define leakage-controlled evaluation after project and duplicate-group checks',
        'No original files or class labels were modified.'])
    save_report('13_dataset_suitability_summary.txt',report)
    print('\n'.join(report))

if __name__=='__main__': main()
