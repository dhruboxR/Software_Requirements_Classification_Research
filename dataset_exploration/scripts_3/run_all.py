from pathlib import Path
import subprocess
import sys

folder=Path(__file__).resolve().parent
for num in range(19,25):
    matches=sorted(folder.glob(f'{num}_*.py'))
    if len(matches)!=1: raise RuntimeError(f'Expected exactly one script {num}, got {matches}')
    print('\nRUNNING:',matches[0].name,flush=True)
    subprocess.run([sys.executable,str(matches[0])],check=True)
print('\nValidation package completed. Review output report 24_phase_1c_validation_summary.txt')
