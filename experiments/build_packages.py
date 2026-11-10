"""Package the final model separately from experiment source/results. No report included."""
from pathlib import Path
import argparse
import zipfile
import shutil
import json
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir',type=Path,default=ROOT/'deliverables')
    args=ap.parse_args()
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'results/final_strategy.json').read_text())
    assert config['selected']['demand']['strategy']=='periodic_3y'
    assert config['selected']['price']['strategy']=='input_drift_3y'
    artifact=ROOT/'forecast/artifacts'
    size=sum(p.stat().st_size for p in artifact.iterdir() if p.is_file())
    assert size<=100*1024**2
    with zipfile.ZipFile(out/'submission.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ('model.py','train.py'):
            z.write(ROOT/'forecast'/name,name)
        for p in sorted(artifact.iterdir()):
            if p.is_file(): z.write(p,'artifacts/'+p.name)
    files=[]
    files.extend(ROOT.glob('*.md'))
    files.extend([ROOT/'requirements.txt',ROOT/'check_submission.py',ROOT/'run_all.py'])
    for folder in ('experiments','forecast','harness'):
        files.extend((ROOT/folder).glob('*.py'))
    # Small saved evidence is convenient; models/prediction parquet can be rebuilt.
    for p in (ROOT/'results').rglob('*'):
        if p.is_file() and p.suffix in ('.csv','.json','.png','.log'):
            files.append(p)
    with zipfile.ZipFile(out/'experiment_code.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)):
            z.write(p,p.relative_to(ROOT).as_posix())
    for name in ('IMPLEMENTATION.md','REQUIREMENTS_CHECKLIST.md'):
        shutil.copyfile(ROOT/name,out/name)
    plotout=out/'plots'
    plotout.mkdir(exist_ok=True)
    for p in (ROOT/'results/plots').glob('*.png'):
        shutil.copyfile(p,plotout/p.name)
    manifest={'artifact_bytes':size,'model_zip_bytes':(out/'submission.zip').stat().st_size,
              'experiment_zip_bytes':(out/'experiment_code.zip').stat().st_size,
              'report':'not generated; group-written report required by course specification'}
    (out/'package_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))
    print('Packaged into',out)

if __name__=='__main__': main()
