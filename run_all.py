"""One entry point. Run from the project environment; no network access required."""
from pathlib import Path
import argparse
import subprocess
import sys
ROOT=Path(__file__).resolve().parent


def run(script,*args):
    print('\nRunning '+script,flush=True)
    subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],cwd=ROOT,check=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir',type=Path,default=ROOT/'deliverables')
    args=ap.parse_args()
    for script in ('baselines','stage1','stage2','feature_analysis','make_plots','diagnostics','liveweek_analysis','verify'):
        run(f'experiments/{script}.py')
    run('forecast/train.py',ROOT/'data/nem_weather_halfhourly_2010_2026.csv')
    run('experiments/check_reproducibility.py')
    run('experiments/check_inference.py')
    run('experiments/build_packages.py','--output-dir',args.output_dir)
    # Save authoritative checker output, including the in-sample-score warning.
    checked=subprocess.run([sys.executable,str(ROOT/'check_submission.py'),str(args.output_dir/'submission.zip')],
                           cwd=ROOT,capture_output=True,text=True)
    print(checked.stdout,flush=True)
    print(checked.stderr,file=sys.stderr,flush=True)
    (ROOT/'results/submission_check.log').write_text(checked.stdout+'\n'+checked.stderr)
    if checked.returncode: raise SystemExit(checked.returncode)
    if 'All days ran successfully.' not in checked.stdout: raise SystemExit('Checker did not confirm success')
    # Refresh source ZIP so it includes this run's checker evidence. Model ZIP is unchanged.
    run('experiments/build_packages.py','--output-dir',args.output_dir)
    print('Completed. Report must be written by your group. Deliverables:',args.output_dir,flush=True)

if __name__=='__main__': main()
