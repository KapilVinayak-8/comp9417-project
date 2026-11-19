# COMP9417 NSW Electricity Forecasting

This is our COMP9417 group project. We built models to predict electricity demand and price direction in NSW. At the end of each day, our models predict demand in megawatts and whether the spot price will be above today's average for each of tomorrow's 48 half-hour intervals.

We compared our models with two simple baselines that copy the same half-hour from yesterday or last week. We also tested how the models performed as market conditions changed. In our historical experiments, scheduled retraining worked best for demand, while retraining triggered by changes in the inputs worked best for price direction among the strategies we tested.

## Before submitting: instructions for the uploader

Download the latest repository using **Code > Download ZIP**, then extract it.

Use these files:

- `deliverables/cleaned/submission.zip`: trained model package for live/practice evaluation.
- `deliverables/cleaned/experiment_code.zip`: experiment code and saved results.
- Final report PDF: review `deliverables/COMP9417_GroupProjectReport.docx` and export it to PDF.

For the **live/practice model evaluation form**, upload only `submission.zip` and enter the group name. For the **final report/code submission**, follow its specific upload instructions. If it requires one combined ZIP, include the report PDF and the two ZIPs above inside it. Do not add the report or experiment archive inside the model package. Do not upload the whole GitHub download as the model package.

Before uploading:

1. Check group details and make sure the report matches the results. Check the 10-page main-text limit, 11 pt font, 1.5 spacing, references, figure captions/units and live-week forecast with uncertainty ranges. In the introduction, market demand/prices should refer to NSW, Victoria and Queensland; Sydney, Melbourne and Brisbane are the weather locations.
2. Set up Python 3.11 using the commands below. Put the supplied course CSV in `data/nem_weather_halfhourly_2010_2026.csv`. The experiment ZIP does not include the dataset.
3. From the repository root, test the exact package being uploaded:

   ```powershell
   .\.venv\Scripts\python.exe -m unittest discover -s tests -v
   .\.venv\Scripts\python.exe check_submission.py deliverables/cleaned/submission.zip
   ```

   Expect six tests to pass and `All days ran successfully.` The checker extracts the ZIP automatically. These dates are included in final training, so the scores verify operation rather than unseen performance. Local checks were on Windows; Linux execution has not been verified here.
4. Keep `model.py`, `train.py` and `artifacts/` directly at the root of `submission.zip`. Do not remove the trained artifacts.
5. Running `run_all.py` rebuilds files in the current project's `deliverables/` folder and tests its generated `submission.zip`. It does not automatically replace the separate `cleaned/` ZIPs. Running from an extracted experiment folder creates a nested `deliverables/` folder there.
6. Exclude virtual environments, temporary/extracted test folders and the dataset unless the course asks for them. Confirm the upload form and filenames, then save the submission receipt.

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
