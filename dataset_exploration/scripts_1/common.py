from pathlib import Path
import re
import unicodedata
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = (
    PROJECT_ROOT
    / 'A Unified Multi-Source Dataset of Software Require'
)
EXPLORE_ROOT = PROJECT_ROOT / 'dataset_exploration'
REPORTS = EXPLORE_ROOT / 'outputs' / 'reports'
TABLES = EXPLORE_ROOT / 'outputs' / 'tables'
FIGURES = EXPLORE_ROOT / 'outputs' / 'figures'
for folder in (REPORTS, TABLES, FIGURES):
    folder.mkdir(parents=True, exist_ok=True)

TEXT_COL = 'Requirement Text (Original Requirement)'
LABEL_COL = 'Class'
PROJECT_COL = 'Project ID'
SOURCE_COL = 'Datasets'
ID_COL = 'Requirement ID'


def find_main_csv(stage='preprocessing'):
    files = list(DATASET_ROOT.rglob('*.csv'))
    match = [p for p in files if p.name.lower().startswith('software requirement dataset')
             and f'after the {stage.lower()}.csv' in p.name.lower()]
    if len(match) != 1:
        raise FileNotFoundError(f'Expected one main {stage} CSV, found {len(match)}: {match}')
    return match[0]


def load_main(stage='preprocessing'):
    path = find_main_csv(stage)
    df = pd.read_csv(path, dtype='string', encoding='utf-8-sig', keep_default_na=False, low_memory=False)
    required = (TEXT_COL, LABEL_COL, PROJECT_COL, SOURCE_COL, ID_COL)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns in {path.name}: {missing}')
    return df


def normalize_text(value):
    if pd.isna(value):
        return ''
    text = unicodedata.normalize('NFKC', str(value))
    text = re.sub(r'\s+', ' ', text).strip().casefold()
    return text


def save_csv(frame, filename):
    path = TABLES / filename
    frame.to_csv(path, index=False, encoding='utf-8-sig')
    print(f'Saved: {path.relative_to(PROJECT_ROOT)}')
    return path


def save_report(filename, lines):
    path = REPORTS / filename
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Saved: {path.relative_to(PROJECT_ROOT)}')
    return path
