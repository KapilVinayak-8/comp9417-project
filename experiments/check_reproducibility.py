"""Verify train.py recreates the same fitted model, then leave rebuilt artifacts in place."""
from pathlib import Path
import sys
import subprocess
import hashlib
import json
ROOT=Path(__file__).resolve().parents[1]


def main():
    artifact=ROOT/'forecast/artifacts/models.joblib'
    before=hashlib.sha256(artifact.read_bytes()).hexdigest()
    subprocess.run([sys.executable,str(ROOT/'forecast/train.py'),str(ROOT/'data/nem_weather_halfhourly_2010_2026.csv')],check=True)
    after=hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert before==after, 'Rebuilt fitted artifacts differ byte-for-byte'
    result={'status':'passed','comparison':'byte-identical models.joblib after rebuilding with train.py',
            'sha256':after,'platform':'local Windows Python 3.11; Linux environment not available here'}
    (ROOT/'results/reproducibility.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
