from pathlib import Path
import sys
import re
import unicodedata
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
for dirname in ('scripts_1', 'scripts_2'):
    sys.path.insert(0, str(HERE.parent / dirname))
from common import load_main, TEXT_COL, LABEL_COL, SOURCE_COL, PROJECT_COL, ID_COL

TABLES = PROJECT_ROOT / 'dataset_exploration' / 'outputs' / 'tables'
REPORTS = PROJECT_ROOT / 'dataset_exploration' / 'outputs' / 'reports'
for folder in (TABLES, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)

def clean(x):
    return '' if pd.isna(x) else str(x).strip()

def norm(x):
    if pd.isna(x): return ''
    x = unicodedata.normalize('NFKC', str(x))
    return re.sub(r'\s+', ' ', x).strip().casefold()

def get_data():
    df = load_main('preprocessing').copy()
    for c in (SOURCE_COL, LABEL_COL, PROJECT_COL, ID_COL, 'Project Name'):
        df[c] = df[c].map(clean)
    df['text_key'] = df[TEXT_COL].map(norm)
    df['source'] = df[SOURCE_COL]
    df['label'] = df[LABEL_COL].replace({'': '[BLANK]'})
    return df

def save(df, name):
    path = TABLES / name
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print('Saved:', path.relative_to(PROJECT_ROOT))

def report(name, lines):
    path = REPORTS / name
    path.write_text('\n'.join(str(s) for s in lines) + '\n', encoding='utf-8')
    print('Saved:', path.relative_to(PROJECT_ROOT))
