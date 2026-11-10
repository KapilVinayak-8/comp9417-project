"""Check current Forecaster outputs against the marking schema."""
from pathlib import Path
import sys
import time
import json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from harness import dayahead as h
from harness.run_day import validate
from forecast.model import Forecaster


def main():
    df=h.load_dataset(ROOT/'data/nem_weather_halfhourly_2010_2026.csv')
    t=time.perf_counter()
    model=Forecaster(ROOT/'forecast/artifacts')
    init=time.perf_counter()-t
    records=[]
    for day in h.complete_days(df)[-14:]:
        history,weather=h.make_inputs(df,day)
        t=time.perf_counter()
        prediction=model.predict(history,weather)
        duration=time.perf_counter()-t
        _,error=validate(prediction,weather.index)
        assert error is None,error
        assert duration<20
        records.append({'day':str(day.date()),'predict_seconds':duration,'status':'ok'})
    assert init<60
    result={'stage':'current_final_models','init_seconds':init,'days':records,
            'note':'Schema and timing checks only; not the sandboxed final ZIP checker.'}
    (ROOT/'results/inference_checks.json').write_text(json.dumps(result,indent=2))
    print(f'All {len(records)} days passed; init {init:.3f}s; max predict {max(r["predict_seconds"] for r in records):.3f}s')

if __name__=='__main__':
    main()
