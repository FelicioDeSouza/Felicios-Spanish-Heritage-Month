# Function & Historical Data Lab

A self-contained Streamlit teaching app combining official World Bank historical series, polynomial curve fitting, a validated editable data table, exports, and a symbolic/numerical function-analysis workspace.

## Files

- `app.py` — entire application.
- `requirements.txt` — minimal, unpinned dependency list.

## Run locally

Use Python 3.10 or newer.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py` and `requirements.txt` to the repository root.
3. Open Streamlit Community Cloud and connect the repository.
4. Choose the branch, `app.py` as the main file, then deploy.
5. If deployment fails, open the app's logs and look for dependency, Python-version, network, or API errors.

No credentials are required. The app calls the public World Bank API.

## Deploy to Render

Create a **Python Web Service** connected to the GitHub repository.

Build command:

```bash
pip install -r requirements.txt
```

Start command (single line):

```bash
python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
```

## Controls and behavior

- Course level: Honors Algebra 2, Precalculus, Calculus.
- Workspace: World Bank historical data or function builder/calculus.
- Historical mode: country, indicator, polynomial degree, sample increment, forward display horizon, comparison countries, printer-friendly report, editable data, reset, single-year and batch calculations, and CSV/XLSX/PDF exports.
- Function mode: context, course level, suggested / data-fit / parameter-built / user-entered expression, variables and units, analysis interval, moving input, secant interval, curve visibility, contextual indicator, and exports.
- Expressions accept `x`, numbers, `+ - * /`, powers using `^` or `**`, parentheses, and `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, and `abs`.

## Validation checklist

After deploying, manually verify:

1. Source accuracy: select countries/indicators and check the source URL, definition, units, retrieval date, and actual year coverage against the World Bank API.
2. Editing/reset: change a value, confirm “user-modified scenario”, then use **Reset to source**.
3. Invalid data: try duplicate years, non-integer years, blank cells, nonnumeric values, and too few distinct years for the chosen degree.
4. Model: check the displayed centered-time equation, in-sample R²/MAE, residuals, observation count, range, and mean.
5. Prediction: calculate one year, request the same year in a batch, and confirm the estimates agree; test reversed ranges, malformed comma-separated years, duplicates, and out-of-range years.
6. Controls: change degree, sample increment, forward horizon, comparison countries, printer mode, course mode, function family, parameters, interval, point, and curve visibility; each should change its relevant output.
7. Domains: test `1/(x-2)`, `log(x)`, `sqrt(x)`, `sin(x)`, and `x^3-3*x`; inspect discontinuity/domain warnings and derivative behavior.
8. Exports: open the CSV, Excel workbook (Summary, Working Data, Calculated Years), and PDF. Verify they match the current selection and edited-data status.
9. Deployment: reload the hosted app, inspect logs, and check API access from the hosting environment.

## Troubleshooting

- `ModuleNotFoundError`: confirm `requirements.txt` is at repository root and the deployment build completed.
- World Bank API timeout or error: retry, check the URL, internet access, and the API response; the app does not fabricate replacement values.
- “Not enough distinct years”: choose a lower degree or a series with more observations.
- Duplicate years / invalid values: correct the editable table or reset to source.
- PDF/XLSX download failure: inspect deployment logs and verify ReportLab/XlsxWriter installed.
- Blank graph or undefined values: the expression may not be real-valued over the chosen interval.
- Render port errors: use the exact `$PORT` start command above.
- If an error persists, copy the traceback from Streamlit/Render logs and inspect the first application frame.

## Team explanation: functions and data flow

1. The historical workspace requests JSON from the official World Bank Indicators API for the 70 complete calendar years ending last year.
2. The original fetched observations are retained separately from the editable working table. The working table is validated before modeling.
3. Polynomial least squares uses `t = year - base_year`, reducing numerical instability from large year values. The same centered transformation is used for fitted values and predictions.
4. R² and MAE are explicitly in-sample. A separate chronological holdout reserves the latest 20% when enough training data remains. Residuals show observed minus fitted values.
5. Single-year and batch calculations share one prediction function. Results are held in session state and cleared when the model/data signature changes.
6. The function workspace parses mathematical expressions with SymPy, derives first and second derivatives symbolically, and combines symbolic solving with numerical sign checks and plots.
7. CSV, Excel, and PDF are generated in memory from the current app state.

## Mathematical assumptions and limitations

- A polynomial is a descriptive least-squares model, not proof of a causal relationship and not necessarily a good forecasting model.
- High-degree polynomials can oscillate, overfit, and extrapolate implausibly. The app flags out-of-range estimates but does not silently clip them.
- MAE/residuals are not validated 95% prediction intervals. No such interval is presented.
- World Bank series have country-specific coverage, missing years, revisions, and definitions. The app shows available observations rather than claiming 70 observations.
- The symbolic solver may not return closed-form critical points for every expression. Numerical extrema are labeled as grid estimates.
- The domain checker is sample-based, not a complete theorem prover. Always verify denominator restrictions, logarithm arguments, square-root domains, discontinuities, and endpoint behavior.
- Contextual visuals are conceptual normalized indicators, not medical advice, biomechanical simulation, or validated physical models.
