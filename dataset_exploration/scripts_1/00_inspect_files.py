"""Inventory all original files; list ZIP contents without extracting anything."""
from pathlib import Path
import zipfile
import pandas as pd
from common import DATASET_ROOT, save_csv, save_report


def main():
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f'Dataset folder not found: {DATASET_ROOT}')
    records, archive_records = [], []
    for path in sorted(DATASET_ROOT.rglob('*')):
        if not path.is_file():
            continue
        relative = str(path.relative_to(DATASET_ROOT))
        records.append({'relative_path': relative, 'extension': path.suffix.lower(),
                        'size_bytes': path.stat().st_size, 'size_mb': round(path.stat().st_size / 1048576, 3)})
        if path.suffix.lower() == '.zip':
            try:
                with zipfile.ZipFile(path) as z:
                    for entry in z.infolist():
                        archive_records.append({'archive': relative, 'member': entry.filename,
                                                'uncompressed_bytes': entry.file_size,
                                                'is_directory': entry.is_dir()})
            except (OSError, zipfile.BadZipFile) as err:
                print(f'WARNING: Cannot inspect archive {relative}: {err}')
    save_csv(pd.DataFrame(records, columns=['relative_path','extension','size_bytes','size_mb']), '00_file_inventory.csv')
    save_csv(pd.DataFrame(archive_records, columns=['archive','member','uncompressed_bytes','is_directory']), '00_zip_contents.csv')
    save_report('00_file_inventory_summary.txt', [f'Dataset root: {DATASET_ROOT}',
                f'Number of files (including extracted sources): {len(records)}',
                f'ZIP member records: {len(archive_records)}',
                'ZIP files were inspected but not extracted or modified.'])
    print(f'Inventory: {len(records)} files; ZIP entries: {len(archive_records)}')

if __name__ == '__main__':
    main()
