"""Shared read-only helpers for phase-2 diagnostics. Imports existing common.py."""
from collections import Counter
import csv
import re
from pathlib import Path
import pandas as pd
from common import (DATASET_ROOT, TEXT_COL, LABEL_COL, ID_COL, SOURCE_COL,
                    PROJECT_COL, load_main, normalize_text, save_csv, save_report)


def dataset():
    d = load_main('preprocessing').copy()
    for col in (TEXT_COL, LABEL_COL, ID_COL, SOURCE_COL, PROJECT_COL):
        d[col] = d[col].fillna('').astype('string').str.strip()
    d['_norm'] = d[TEXT_COL].map(normalize_text)
    d['_label'] = d[LABEL_COL].replace('', '[BLANK]')
    return d


def source_catalog():
    """Find original source candidates, not including combined CSV files."""
    specs = [
        ('kaggle_extended', 'software_requirements_extended.csv', 'csv'),
        ('dosspre', 'DOSSPRE 1.0Original.csv', 'csv'),
        ('fr_nfr', 'FR_NFR_Dataset Original.xlsx', 'xlsx'),
        ('promise_exp', 'PROMISE_exp.txt', 'txt'),
    ]
    for name, filename, kind in specs:
        matches = list(DATASET_ROOT.rglob(filename))
        yield {'name':name, 'filename':filename, 'kind':kind,
               'path':matches[0] if len(matches)==1 else None,
               'num_paths':len(matches)}


def detect_encoding(path):
    for encoding in ('utf-8-sig', 'cp1252'):
        try:
            path.read_text(encoding=encoding)
            return encoding
        except UnicodeDecodeError:
            pass
    return None


def extract_csv_candidate_texts(path):
    encoding=detect_encoding(path)
    if not encoding:
        return set(), 'unreadable encoding'
    # Use only an explicitly identified requirement-text column.
    with path.open('r', encoding=encoding, newline='') as f:
        reader=csv.DictReader(f)
        fields=reader.fieldnames or []
        candidates=[c for c in fields if re.search(r'requirement|description|sentence|text', c, re.I)
                    and not re.search(r'id|type|class|label', c, re.I)]
        if not candidates:
            return set(), f'no identified text column; columns={fields}'
        column=sorted(candidates, key=lambda c: (c.strip().lower() not in ('requirement','requirement text'),len(c)))[0]
        texts={normalize_text(row.get(column,'')) for row in reader}
    texts.discard('')
    return texts, f'column={column}, encoding={encoding}'


def extract_xlsx_candidate_texts(path):
    try:
        xls=pd.ExcelFile(path)
    except (ImportError,ValueError,OSError) as e:
        return set(), f'Excel read failed: {e}'
    texts=set(); descriptions=[]
    for sheet in xls.sheet_names:
        try:
            sample=pd.read_excel(xls, sheet_name=sheet, dtype=str, nrows=0)
            fields=list(sample.columns)
            candidates=[c for c in fields if re.search(r'requirement|description|sentence|text',str(c),re.I)
                        and not re.search(r'id|type|class|label',str(c),re.I)]
            if not candidates:
                descriptions.append(f'{sheet}: no recognized text column ({fields[:12]})')
                continue
            column=sorted(candidates,key=lambda c: (str(c).strip().lower() not in ('requirement','requirement text'),len(str(c))))[0]
            frame=pd.read_excel(xls,sheet_name=sheet,dtype=str,usecols=[column])
            texts.update(normalize_text(x) for x in frame[column].fillna(''))
            descriptions.append(f'{sheet}: column={column}, rows={len(frame)}')
        except (ValueError, KeyError, ImportError, OSError) as e:
            descriptions.append(f'{sheet}: failed {e}')
    texts.discard('')
    return texts, '; '.join(descriptions)


def get_reference_sets():
    """PROMISE txt is intentionally left unparsed until format is verified."""
    result=[]
    for item in source_catalog():
        path=item['path']; texts=set(); detail=''
        if path is None:
            detail=f'expected exactly one path, found {item["num_paths"]}'
        elif item['kind']=='csv':
            texts,detail=extract_csv_candidate_texts(path)
        elif item['kind']=='xlsx':
            texts,detail=extract_xlsx_candidate_texts(path)
        elif item['kind']=='txt':
            detail='NOT PARSED: inspect PROMISE_exp.txt schema before matching'
        result.append({**item, 'texts':texts, 'detail':detail,
                       'status': 'parsed' if texts else 'not verified'})
    return result


def normalized_comparison(frame):
    if frame.empty:
        return pd.DataFrame()
    return frame.copy()
