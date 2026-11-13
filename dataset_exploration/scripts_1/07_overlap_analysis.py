"""Conservative exact-match checks against available original-source text.

Checks normalized requirement text against Kaggle 'Requirement' and PROMISE_exp
lines where parsing is defensible. Candidate matches do NOT prove provenance;
unmatched text does NOT prove dataset independence.
"""
import csv
import pandas as pd
from common import DATASET_ROOT, load_main, normalize_text, save_csv, save_report, TEXT_COL, ID_COL, LABEL_COL, SOURCE_COL


def kaggle_texts(path):
    refs = set()
    for enc in ('utf-8-sig','cp1252'):
        try:
            with path.open(encoding=enc, newline='') as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or 'Requirement' not in reader.fieldnames:
                    raise ValueError('Requirement column not found')
                for row in reader:
                    norm = normalize_text(row.get('Requirement', ''))
                    if norm:
                        refs.add(norm)
            return refs
        except UnicodeDecodeError:
            refs.clear()
    return refs


def promise_texts(path):
    # PROMISE_exp.txt structure can vary. Only use complete non-empty lines
    # after dropping possible leading labels separated by tab. Report limitations.
    for enc in ('utf-8-sig','cp1252'):
        try:
            lines = path.read_text(encoding=enc).splitlines()
            break
        except UnicodeDecodeError:
            continue
    else:
        return set()
    texts = set()
    for line in lines:
        if not line.strip() or line.lstrip().startswith(('@','%','#')):
            continue
        # For tab-separated labeled records, take the final field as candidate text.
        candidate = line.split('\t')[-1] if '\t' in line else line
        candidate = candidate.strip().strip('"\'')
        norm = normalize_text(candidate)
        if len(norm.split()) >= 4:
            texts.add(norm)
    return texts


def main():
    df = load_main()
    text_values = df[TEXT_COL].map(normalize_text)
    matches = pd.DataFrame({ID_COL: df[ID_COL], SOURCE_COL: df[SOURCE_COL], LABEL_COL: df[LABEL_COL],
                            'normalized_text': text_values})
    report = []
    specs = [('kaggle_extended','software_requirements_extended.csv',kaggle_texts),
             ('promise_exp','PROMISE_exp.txt',promise_texts)]
    for label, filename, loader in specs:
        paths = list(DATASET_ROOT.rglob(filename))
        if not paths:
            report.append(f'{label}: NOT TESTED, source file missing ({filename})')
            print(report[-1])
            continue
        if len(paths) != 1:
            report.append(f'{label}: NOT TESTED, found {len(paths)} paths for {filename}')
            continue
        ref_set = loader(paths[0])
        flag = matches['normalized_text'].isin(ref_set) & matches['normalized_text'].ne('')
        matches[f'match_{label}'] = flag
        report.append(f'{label}: source candidate strings={len(ref_set)}, matched main rows={int(flag.sum())}, distinct matched texts={matches.loc[flag,"normalized_text"].nunique()}')
        print(report[-1])
    save_csv(matches, '07_source_overlap_candidate_matches.csv')
    save_report('07_overlap_limitations.txt', report + [
        'Matches use Unicode NFKC + whitespace normalization + casefold only.',
        'PROMISE_exp.txt is parsed heuristically; inspect its raw format before interpreting counts.',
        'A positive match suggests overlapping text; a negative match does not establish independence.',
        'Assess other extracted sources (XLS/XLSX, DOSSPRE, FR_NFR, Zenodo) separately.',
        'Check transformed/near-duplicate requirements and project lineage before splitting data.',
        'No dataset was altered.'])

if __name__ == '__main__':
    main()
