# COMP9417 NSW Electricity Forecasting

This is our COMP9417 group project. We built models to predict electricity demand and price direction in NSW. At the end of each day, our models predict demand in megawatts and whether the spot price will be above today's average for each of tomorrow's 48 half-hour intervals.

We compared our models with two simple baselines that copy the same half-hour from yesterday or last week. We also tested how the models performed as market conditions changed. In our historical experiments, scheduled retraining worked best for demand, while retraining triggered by changes in the inputs worked best for price direction among the strategies we tested.

## How to run it

You'll need Python 3.11 and the dataset from the course kit. Put the CSV in `data/` with the filename `nem_weather_halfhourly_2010_2026.csv`.

From the project folder, run these commands in Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe check_submission.py deliverables/submission.zip
```

Our submission package passed the local checker on all 14 days. The checker uses days included in final training, so we used it to verify that the package loads and produces valid forecasts.

To rerun the experiments and rebuild the models, plots, and submission ZIPs, run:

```powershell
.\.venv\Scripts\python.exe run_all.py
```

## What's in the repo

| Path | What it contains |
| --- | --- |
| `experiments/` | Baselines, model comparisons, drift analysis, and plotting scripts |
| `forecast/` | Code that trains the models and makes predictions |
| `results/` | Saved scores, tables, and plots |
| `deliverables/experiment_code.zip` | Experiment code and results |
| `deliverables/submission.zip` | Prediction code and trained models for the live evaluation |
| `IMPLEMENTATION.md` | More detail on how the project fits together |
| `specs/project_spec.pdf` | The official assignment brief |

The COMP9417 course kit provided the starter notebook, checker, and evaluation harness. The supplied dataset combines AEMO market data with Open-Meteo weather data (CC BY 4.0).
