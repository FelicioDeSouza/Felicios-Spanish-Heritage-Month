import io
import math
import re
import time
import warnings
from datetime import date, datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import sympy as sp
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Function & Historical Data Lab", page_icon="📈", layout="wide")
st.title("📈 Function & Historical Data Lab")
st.caption("Historical data, curve fitting, and interactive function analysis for Honors Algebra 2, Precalculus, and Calculus.")

WB_INDICATORS = {
    "Life expectancy": ("SP.DYN.LE00.IN", "years", "Life expectancy at birth, total (years)", "Higher values indicate longer expected lifespan; this is a population-level estimate."),
    "Unemployment rate (national estimate)": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Unemployment, total (% of total labor force) (national estimate)", "The share of the labor force without work but available for and seeking employment."),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US$ per person", "GDP per capita (current US$)", "GDP divided by population. It is not household income or average wealth."),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Birth rate, crude (per 1,000 people)", "Annual live births per 1,000 people, not a total birth count."),
    "Net migration": ("SM.POP.NETM", "people (net total)", "Net migration", "The net number of migrants; it is not a count of emigration alone."),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "per 100,000 people", "Intentional homicides (per 100,000 people)", "Recorded intentional homicides per 100,000 population."),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual change", "Inflation, consumer prices (annual %)", "Annual percentage change in the consumer price index."),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt, total (% of GDP)", "Central government debt relative to GDP; series availability varies."),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US$", "Total reserves (includes gold, current US$)", "Official reserve assets including gold, reported in current US dollars."),
    "External balance on goods and services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services (% of GDP)", "Exports minus imports of goods and services as a share of GDP.")
}
COUNTRIES = {"Mexico": "MX", "Brazil": "BR", "Argentina": "AR", "Colombia": "CO", "Chile": "CL", "Peru": "PE", "Venezuela": "VE", "Ecuador": "EC", "Uruguay": "UY"}
CONTEXTS = {
    "Business": dict(desc="A product's revenue or profit over time.", x="Time", xu="months", y="Profit", yu="USD", expr="-(x-5)**2+25", visual="Business indicator"),
    "Engineering": dict(desc="Vertical position of a moving test object.", x="Time", xu="seconds", y="Position", yu="meters", expr="-4.9*x**2+20*x+2", visual="Moving object"),
    "Medical": dict(desc="Illustrative medication concentration over time (not clinical guidance).", x="Time", xu="hours", y="Concentration", yu="mg/L", expr="10*exp(-0.3*x)", visual="Tank / concentration"),
    "Sports science": dict(desc="An athlete's illustrative speed profile during a drill.", x="Time", xu="seconds", y="Speed", yu="m/s", expr="-0.2*(x-5)**2+5", visual="Moving object"),
    "Biology": dict(desc="A simple population growth curve.", x="Time", xu="days", y="Population", yu="cells", expr="100*exp(0.15*x)", visual="Population / cells"),
    "Bioengineering": dict(desc="Illustrative prosthetic-leg joint angle through a gait cycle.", x="Cycle time", xu="seconds", y="Joint angle", yu="degrees", expr="25*sin(2*pi*x)+15", visual="Walking person with prosthetic leg"),
    "Economic forecasting": dict(desc="An illustrative economic indicator.", x="Time", xu="years", y="Index", yu="index points", expr="100+3*x+0.2*x**2", visual="Economic indicator"),
    "Weather": dict(desc="An illustrative temperature cycle.", x="Time", xu="hours", y="Temperature", yu="°C", expr="12+8*sin(2*pi*x/24)", visual="Thermometer / weather"),
    "Custom": dict(desc="Describe the relationship you want to explore.", x="Input", xu="units", y="Output", yu="units", expr="x**2", visual="Moving object")
}
X = sp.Symbol("x", real=True)
ALLOWED_FUNCS = {name: getattr(sp, name) for name in ["sin", "cos", "tan", "exp", "log", "sqrt", "Abs"]}
ALLOWED_FUNCS.update({"pi": sp.pi, "E": sp.E, "abs": sp.Abs})


def safe_parse(expr):
    """Parse a familiar mathematical expression without evaluating Python code."""
    expr = expr.strip().replace("^", "**")
    if not expr or len(expr) > 250:
        raise ValueError("Enter a non-empty expression under 250 characters.")
    if not re.fullmatch(r"[0-9xX+\-*/().,\sA-Za-z_*]+", expr):
        raise ValueError("Use x, numbers, arithmetic, parentheses, and supported functions only.")
    expr = re.sub(r"\bX\b", "x", expr)
    local = {"x": X, **ALLOWED_FUNCS}
    try:
        result = sp.sympify(expr, locals=local, evaluate=True)
    except Exception as exc:
        raise ValueError("Could not parse the expression. Check parentheses, operators, and function names.") from exc
    if result.has(sp.zoo, sp.nan, sp.oo, -sp.oo) or any(not isinstance(s, (sp.Symbol,)) for s in result.free_symbols):
        raise ValueError("Expression contains an unsupported symbol or undefined constant.")
    return result


def eval_expr(expr, values):
    fn = sp.lambdify(X, expr, modules=["numpy"])
    with np.errstate(all="ignore"):
        out = np.asarray(fn(values), dtype=float)
    if out.ndim == 0:
        out = np.full(np.asarray(values).shape, float(out))
    return out


def fmt_num(value, digits=6):
    try:
        if not np.isfinite(float(value)):
            return "undefined"
        return f"{float(value):.{digits}g}"
    except Exception:
        return str(value)


def get_world_bank(country_code, indicator_code, start_year, end_year):
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}"
    params = {"format": "json", "per_page": 2000, "date": f"{start_year}:{end_year}"}
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        raise ValueError("World Bank API returned an unexpected or empty response.")
    records = []
    for item in payload[1]:
        if item.get("value") is not None and item.get("date") is not None:
            try:
                records.append({"year": int(item["date"]), "value": float(item["value"])})
            except (ValueError, TypeError):
                continue
    df = pd.DataFrame(records, columns=["year", "value"])
    if not df.empty:
        df = df.drop_duplicates("year").sort_values("year").reset_index(drop=True)
    return df, url


def fit_polynomial(df, degree):
    clean = df[["year", "value"]].copy()
    clean["year"] = pd.to_numeric(clean["year"], errors="coerce")
    clean["value"] = pd.to_numeric(clean["value"], errors="coerce")
    clean = clean.dropna()
    clean = clean[np.isfinite(clean["year"]) & np.isfinite(clean["value"])]
    clean["year"] = clean["year"].astype(int)
    if clean["year"].duplicated().any():
        raise ValueError("Year values must be unique before fitting.")
    clean = clean.sort_values("year")
    if clean["year"].nunique() < degree + 1:
        raise ValueError(f"Degree {degree} needs at least {degree + 1} distinct valid years; currently there are {clean['year'].nunique()}.")
    base = int(clean["year"].min())
    t = clean["year"].to_numpy(dtype=float) - base
    y = clean["value"].to_numpy(dtype=float)
    coeff = np.polynomial.polynomial.polyfit(t, y, degree)
    pred = np.polynomial.polynomial.polyval(t, coeff)
    r2 = r2_score(y, pred) if len(y) > 1 else float("nan")
    mae = mean_absolute_error(y, pred)
    residuals = y - pred
    return clean, base, coeff, pred, float(r2), float(mae), residuals


def poly_string(coeff, base, xname="year"):
    terms = []
    for power, c in enumerate(coeff):
        if abs(c) < 1e-14:
            continue
        var = f"(year − {base})" if power == 1 else (f"(year − {base})^{power}" if power > 1 else "")
        terms.append(f"{c:+.10g}" + (f"·{var}" if var else ""))
    return " ".join(terms).lstrip("+") or "0"


def holdout_score(df, degree):
    ordered = df.sort_values("year")
    n = len(ordered)
    test_n = max(1, int(math.ceil(n * 0.2)))
    if n - test_n < degree + 1:
        return None
    train, test = ordered.iloc[:-test_n], ordered.iloc[-test_n:]
    base = int(train["year"].min())
    c = np.polynomial.polynomial.polyfit(train["year"].to_numpy() - base, train["value"].to_numpy(), degree)
    estimate = np.polynomial.polynomial.polyval(test["year"].to_numpy() - base, c)
    return {"n": len(test), "mae": float(mean_absolute_error(test["value"], estimate)), "r2": float(r2_score(test["value"], estimate)) if len(test) > 1 else None,
            "start": int(test.year.min()), "end": int(test.year.max())}


def create_chart(df, coeff, base, degree, sample_step, single_result=None, batch_result=None, title="Model"):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    years = df["year"].to_numpy(dtype=int)
    vals = df["value"].to_numpy(dtype=float)
    ax.scatter(years, vals, label="Observed data", s=25, zorder=3)
    lo, hi = int(years.min()), int(years.max())
    max_year = max(hi, int(years.max()))
    grid = np.linspace(lo, max_year, max(300, (max_year-lo+1)*3))
    ax.plot(grid, np.polynomial.polynomial.polyval(grid-base, coeff), label="Smooth fitted curve", linewidth=2)
    sample_years = np.arange(lo, hi + 1, sample_step)
    ax.scatter(sample_years, np.polynomial.polynomial.polyval(sample_years-base, coeff), label=f"Model samples (every {sample_step} year(s))", marker="x", s=35)
    ax.axvline(hi, linestyle="--", linewidth=1, label=f"Historical boundary ({hi})")
    if single_result is not None:
        ax.scatter([single_result["year"]], [single_result["estimate"]], marker="*", s=150, label="Single-year estimate")
    if batch_result is not None and not batch_result.empty:
        ax.scatter(batch_result["year"], batch_result["estimate"], marker="D", s=35, label="Batch estimates")
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Indicator value")
    ax.grid(True, alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def make_xlsx(summary, data, calculated):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
        data.to_excel(writer, sheet_name="Working Data", index=False)
        calculated.to_excel(writer, sheet_name="Calculated Years", index=False)
        for sheet in writer.sheets.values():
            sheet.freeze_panes(1, 0)
            sheet.set_column(0, 10, 18)
    out.seek(0)
    return out.getvalue()


def make_pdf(summary, data, calculated, fig):
    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=landscape(letter), rightMargin=0.45*inch, leftMargin=0.45*inch, topMargin=0.4*inch, bottomMargin=0.4*inch)
    styles = getSampleStyleSheet()
    story = [Paragraph("Function & Historical Data Lab Report", styles["Title"]), Spacer(1, 8)]
    for key in ["country", "indicator", "retrieved", "data_status", "degree", "base_year", "equation", "r_squared_in_sample", "mae_in_sample", "observation_count", "historical_range", "mean_value", "analysis_notes"]:
        if key in summary:
            story.append(Paragraph(f"<b>{key.replace('_',' ').title()}:</b> {str(summary[key])[:1200]}", styles["BodyText"]))
            story.append(Spacer(1, 3))
    img = io.BytesIO()
    fig.savefig(img, format="png", dpi=140, bbox_inches="tight")
    img.seek(0)
    story.extend([Spacer(1, 8), RLImage(img, width=9.5*inch, height=4.6*inch), Spacer(1, 8), Paragraph("Working data preview", styles["Heading2"])])
    preview = data.head(18).copy()
    table_data = [list(preview.columns)] + preview.astype(str).values.tolist()
    tbl = Table(table_data, repeatRows=1)
    tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dce6f1")),("GRID",(0,0),(-1,-1),0.35,colors.grey),("FONTSIZE",(0,0),(-1,-1),7),("VALIGN",(0,0),(-1,-1),"TOP")]))
    story.append(tbl)
    if not calculated.empty:
        story.extend([Spacer(1, 8), Paragraph("Calculated years", styles["Heading2"])])
        cdata = [list(calculated.columns)] + calculated.head(40).astype(str).values.tolist()
        ct = Table(cdata, repeatRows=1)
        ct.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dce6f1")),("GRID",(0,0),(-1,-1),0.35,colors.grey),("FONTSIZE",(0,0),(-1,-1),7)]))
        story.append(ct)
    doc.build(story)
    out.seek(0)
    return out.getvalue()


def calculus_analysis(expr, interval, point):
    f1 = sp.diff(expr, X)
    f2 = sp.diff(f1, X)
    a, b = sorted(map(float, interval))
    notes = []
    # Numerical plotting grid, with undefined values masked so gaps are not connected.
    grid = np.linspace(a, b, 1200)
    y = eval_expr(expr, grid)
    y[~np.isfinite(y)] = np.nan
    d1 = eval_expr(f1, grid); d1[~np.isfinite(d1)] = np.nan
    d2 = eval_expr(f2, grid); d2[~np.isfinite(d2)] = np.nan
    fpoint = float(eval_expr(expr, np.array([point]))[0])
    slope = float(eval_expr(f1, np.array([point]))[0])
    tangent = fpoint + slope*(grid-point) if np.isfinite(fpoint) and np.isfinite(slope) else np.full_like(grid, np.nan)
    fa, fb = float(eval_expr(expr, np.array([a]))[0]), float(eval_expr(expr, np.array([b]))[0])
    avg = (fb-fa)/(b-a) if b != a and np.isfinite(fa) and np.isfinite(fb) else float("nan")
    secant = fa + avg*(grid-a) if np.isfinite(avg) else np.full_like(grid, np.nan)
    # Numerical critical/inflection candidate search, followed by sign tests.
    critical, extrema, inflections = [], [], []
    try:
        roots = sp.solveset(sp.Eq(f1, 0), X, domain=sp.Interval(a, b))
        if isinstance(roots, sp.FiniteSet):
            critical = sorted(float(sp.N(r)) for r in roots if abs(float(sp.im(sp.N(r)))) < 1e-9)
    except Exception:
        pass
    for c in critical:
        eps = max(1e-5, (b-a)*1e-5)
        leftx, rightx = max(a, c-eps), min(b, c+eps)
        dl = float(eval_expr(f1, np.array([leftx]))[0]); dr = float(eval_expr(f1, np.array([rightx]))[0])
        if np.isfinite(dl) and np.isfinite(dr):
            if dl > 0 and dr < 0: kind = "local maximum"
            elif dl < 0 and dr > 0: kind = "local minimum"
            else: kind = "stationary point, not classified as an extremum"
            extrema.append((c, kind, float(eval_expr(expr, np.array([c]))[0])))
    try:
        roots2 = sp.solveset(sp.Eq(f2, 0), X, domain=sp.Interval(a, b))
        if isinstance(roots2, sp.FiniteSet):
            for r in roots2:
                c = float(sp.N(r))
                eps = max(1e-5, (b-a)*1e-5)
                leftx, rightx = max(a, c-eps), min(b, c+eps)
                dl = float(eval_expr(f2, np.array([leftx]))[0]); dr = float(eval_expr(f2, np.array([rightx]))[0])
                if np.isfinite(dl) and np.isfinite(dr) and dl*dr < 0:
                    inflections.append((c, float(eval_expr(expr, np.array([c]))[0])))
    except Exception:
        pass
    return {"f1": f1, "f2": f2, "grid": grid, "y": y, "d1": d1, "d2": d2, "tangent": tangent, "secant": secant,
            "fpoint": fpoint, "slope": slope, "avg": avg, "fa": fa, "fb": fb, "critical": extrema, "inflections": inflections}


# Sidebar and main app mode
with st.sidebar:
    st.header("Classroom controls")
    course = st.selectbox("Course level", ["Honors Algebra 2", "Precalculus", "Calculus"])
    app_mode = st.radio("Workspace", ["Historical data & curve fit", "Function builder & calculus"])
    st.divider()
    st.caption("Historical data uses the official World Bank API. Function builder is for mathematical exploration, not professional forecasting or clinical advice.")

if app_mode == "Historical data & curve fit":
    st.subheader("World Bank historical data")
    current_year = date.today().year
    start_default, end_default = current_year-70, current_year-1
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        country_name = st.selectbox("Country", list(COUNTRIES.keys()), index=0)
    with c2:
        indicator_name = st.selectbox("Indicator", list(WB_INDICATORS.keys()))
    with c3:
        st.info("Brazil is included as a Latin American comparison country. Indicator availability differs by country and year.")
    code, units, definition, caveat = WB_INDICATORS[indicator_name]
    degree = st.slider("Polynomial degree", 3, 8, 3)
    sample_step = st.slider("Graph/model sample increment (years)", 1, 10, 5)
    horizon = st.slider("Forward display horizon (years)", 0, 50, 10)
    compare_names = st.multiselect("Comparison countries (same indicator)", [n for n in COUNTRIES if n != country_name], default=[])
    printer_mode = st.toggle("Printer-friendly report view", value=False)
    retrieval_date = date.today().isoformat()
    try:
        @st.cache_data(ttl=3600, show_spinner=False)
        def cached_wb(cc, ic, sy, ey):
            return get_world_bank(cc, ic, sy, ey)
        with st.spinner("Retrieving official World Bank observations…"):
            original, source_url = cached_wb(COUNTRIES[country_name], code, start_default, end_default)
        api_error = None
    except Exception as exc:
        original, source_url, api_error = pd.DataFrame(columns=["year","value"]), f"https://api.worldbank.org/v2/country/{COUNTRIES[country_name]}/indicator/{code}", str(exc)
    st.caption(f"Definition: {definition}. Units: {units}. {caveat}")
    st.markdown(f"Source: [World Bank API series]({source_url}) · Retrieved: {retrieval_date} · Requested years: {start_default}–{end_default}")
    if api_error:
        st.error(f"Could not retrieve data: {api_error}. Check internet access, API status, or the source URL. No replacement values were generated.")
    if original.empty:
        st.warning("The source returned no usable observations for this country and indicator. Choose another series or retry later.")
        st.stop()
    source_data = original.copy()
    st.write(f"**Actual coverage:** {int(source_data.year.min())}–{int(source_data.year.max())} · **Observations:** {len(source_data)} (not necessarily 70).")
    if "working_data" not in st.session_state or st.session_state.get("working_key") != (country_name, indicator_name):
        st.session_state.working_data = source_data.copy()
        st.session_state.working_key = (country_name, indicator_name)
    if st.button("Reset to source", help="Discard edits and reload the original API observations for this selection."):
        st.session_state.working_data = source_data.copy()
        st.rerun()
    st.markdown("#### Validated editable year/value table")
    edited = st.data_editor(st.session_state.working_data, num_rows="dynamic", use_container_width=True, key=f"editor_{country_name}_{indicator_name}")
    data_status = "user-modified scenario" if not edited.equals(source_data) else "source data (unmodified)"
    try:
        if edited[["year","value"]].isna().any().any():
            raise ValueError("Remove blank year/value cells or fill them before fitting.")
        numeric_year = pd.to_numeric(edited["year"], errors="raise")
        numeric_val = pd.to_numeric(edited["value"], errors="raise")
        if not np.all(np.equal(numeric_year, np.floor(numeric_year))):
            raise ValueError("Years must be integers.")
        if numeric_year.duplicated().any():
            raise ValueError("Duplicate years are not allowed.")
        if not np.isfinite(numeric_val.to_numpy(dtype=float)).all():
            raise ValueError("Values must be finite numbers.")
        validated = pd.DataFrame({"year": numeric_year.astype(int), "value": numeric_val.astype(float)}).sort_values("year").reset_index(drop=True)
        if (validated.year < 1800).any() or (validated.year > current_year+50).any():
            raise ValueError("Years must be between 1800 and current year + 50.")
        if indicator_name in ["Life expectancy"] and ((validated.value < 0) | (validated.value > 125)).any():
            raise ValueError("Life expectancy must be between 0 and 125 years.")
        if indicator_name in ["Unemployment rate (national estimate)", "Consumer price inflation"] and (validated.value < -100).any():
            raise ValueError("This percentage series contains an implausible value below -100%. Check the edit.")
        if indicator_name in ["Crude birth rate", "Intentional homicide rate"] and (validated.value < 0).any():
            raise ValueError("This rate cannot be negative.")
        st.session_state.working_data = validated.copy()
        fitdata, base, coeff, fitted, r2, mae, residuals = fit_polynomial(validated, degree)
        equation = "f(year) = " + poly_string(coeff, base)
        fitdata["fitted_value"] = fitted
        fitdata["residual"] = residuals
    except Exception as exc:
        st.error(f"Data validation / model error: {exc}")
        st.stop()
    st.caption(f"Data status: **{data_status}**. Polynomial fitting uses t = year − {base}, centering time to improve numerical stability.")
    metrics = st.columns(5)
    metrics[0].metric("Observations", len(fitdata))
    metrics[1].metric("R² (in-sample)", fmt_num(r2, 5))
    metrics[2].metric("MAE (in-sample)", fmt_num(mae, 6))
    metrics[3].metric("Mean observed value", fmt_num(fitdata.value.mean(), 7))
    metrics[4].metric("Base year", str(base))
    st.markdown("**Fitted equation**")
    st.code(equation)
    st.write(f"Historical range: **{fitdata.year.min()}–{fitdata.year.max()}**. Metrics are in-sample and do not prove future accuracy.")
    h = holdout_score(fitdata, degree)
    if h:
        st.write(f"**Chronological holdout (separate from in-sample metrics):** trained on earlier years; tested on {h['start']}–{h['end']} ({h['n']} observations). MAE = {fmt_num(h['mae'])}; R² = {fmt_num(h['r2']) if h['r2'] is not None else 'not defined for one test observation'}.")
    else:
        st.info("Chronological holdout unavailable: not enough earlier observations remain to fit this degree after reserving the latest 20%.")
    c1, c2 = st.columns(2)
    with c1:
        query_year = st.number_input("Single year to calculate (1950–2100)", min_value=1950, max_value=2100, value=min(max(fitdata.year.max()+5,1950),2100), step=1)
        calc = st.button("Calculate Value")
    with c2:
        batch_mode = st.radio("Batch year input", ["Year range", "Custom years"], horizontal=True)
        if batch_mode == "Year range":
            r1, r2c, step = st.columns(3)
            with r1: y_start = st.number_input("Start", 1950, 2100, 2000, 1, key="batch_start")
            with r2c: y_end = st.number_input("End", 1950, 2100, 2030, 1, key="batch_end")
            with step: batch_step = st.selectbox("Step", list(range(1,11)), index=0)
            custom_text = ""
        else:
            custom_text = st.text_input("Years separated by commas", "2000, 2010, 2030")
            y_start, y_end, batch_step = 0, -1, 1
        run_batch = st.button("Calculate batch")
    if "single_result" not in st.session_state:
        st.session_state.single_result = None
    if "batch_result" not in st.session_state:
        st.session_state.batch_result = pd.DataFrame(columns=["year","estimate","classification"])
    model_signature = (country_name, indicator_name, degree, tuple(validated.year), tuple(np.round(validated.value, 12)))
    if st.session_state.get("model_signature") != model_signature:
        st.session_state.single_result = None
        st.session_state.batch_result = pd.DataFrame(columns=["year","estimate","classification"])
        st.session_state.model_signature = model_signature
    def predict_years(years):
        years = np.asarray(years, dtype=int)
        estimates = np.polynomial.polynomial.polyval(years-base, coeff)
        lo, hi = int(fitdata.year.min()), int(fitdata.year.max())
        return pd.DataFrame({"year": years, "estimate": estimates, "classification": ["within fitted historical range" if lo <= y <= hi else "outside fitted historical range (extrapolation)" for y in years]})
    if calc:
        y = int(query_year)
        st.session_state.single_result = predict_years([y]).iloc[0].to_dict()
    if run_batch:
        try:
            if batch_mode == "Year range":
                if y_end < y_start: raise ValueError("End year must be greater than or equal to start year.")
                years = list(range(int(y_start), int(y_end)+1, int(batch_step)))
            else:
                if not custom_text.strip(): raise ValueError("Enter at least one year.")
                pieces = [p.strip() for p in custom_text.split(",")]
                if any(not re.fullmatch(r"\d{4}", p) for p in pieces): raise ValueError("Use comma-separated integer years such as 2000, 2010, 2030.")
                years = [int(p) for p in pieces]
                if len(set(years)) != len(years): raise ValueError("Custom years must not contain duplicates.")
            if any(y < 1950 or y > 2100 for y in years): raise ValueError("Requested years must be between 1950 and 2100.")
            if len(years) > 500: raise ValueError("Limit batch requests to 500 years.")
            st.session_state.batch_result = predict_years(years)
        except Exception as exc:
            st.error(f"Batch input error: {exc}")
    single = st.session_state.single_result
    batch = st.session_state.batch_result
    if single:
        actual = fitdata.loc[fitdata.year == single["year"], "value"]
        st.success(f"Estimate for {single['year']}: **{fmt_num(single['estimate'], 10)} {units}**. {single['classification']}.")
        if not actual.empty:
            st.caption(f"An actual observation exists for this year: {fmt_num(actual.iloc[0], 10)} {units}. The model estimate is a fitted value, not the actual observation.")
        if single["year"] < fitdata.year.min() or single["year"] > fitdata.year.max():
            st.warning("This is extrapolation. Polynomial behavior can become implausible outside observed years; the estimate is not clipped. Use residuals and MAE to understand typical in-sample error, not as a validated prediction interval.")
    if not batch.empty:
        st.dataframe(batch, use_container_width=True)
    if course in ["Precalculus", "Calculus"]:
        st.markdown("#### Residuals")
        st.dataframe(fitdata[["year","value","fitted_value","residual"]], use_container_width=True, height=230)
    max_display = int(fitdata.year.max()+horizon)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(fitdata.year, fitdata.value, label="Observed data", s=25)
    hist_grid = np.linspace(int(fitdata.year.min()), int(fitdata.year.max()), 500)
    ax.plot(hist_grid, np.polynomial.polynomial.polyval(hist_grid-base, coeff), label="Smooth fitted curve")
    sample_years = np.arange(int(fitdata.year.min()), int(fitdata.year.max())+1, sample_step)
    ax.scatter(sample_years, np.polynomial.polynomial.polyval(sample_years-base, coeff), marker="x", label=f"Model samples every {sample_step} year(s)")
    if horizon > 0:
        future = np.linspace(int(fitdata.year.max()), max_display, max(30,horizon*5))
        ax.plot(future, np.polynomial.polynomial.polyval(future-base, coeff), linestyle="--", label="Extrapolated segment")
    ax.axvline(int(fitdata.year.max()), linestyle=":", color="gray", label="Historical boundary")
    if single: ax.scatter([single["year"]],[single["estimate"]],marker="*",s=150,label="Single-year estimate")
    if not batch.empty: ax.scatter(batch.year,batch.estimate,marker="D",label="Batch estimates")
    ax.set_xlabel("Year"); ax.set_ylabel(f"{indicator_name} ({units})"); ax.set_title(f"{country_name}: {indicator_name}")
    ax.grid(True, alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
    if printer_mode:
        st.markdown("### Printable report")
        st.write(f"{country_name} · {indicator_name} · {retrieval_date} · {data_status}")
        st.write(equation)
        st.write(f"R² in-sample {fmt_num(r2)}; MAE in-sample {fmt_num(mae)}; coverage {fitdata.year.min()}–{fitdata.year.max()}.")
        st.pyplot(fig, use_container_width=True)
        st.caption(f"Limitations: polynomial fit is descriptive, API series may contain gaps, and extrapolation is uncertain. {caveat}")
    else:
        st.pyplot(fig, use_container_width=True)
    if compare_names:
        st.markdown("#### Same-indicator country comparison")
        comp_fig, comp_ax = plt.subplots(figsize=(10,4.5))
        comp_ax.plot(fitdata.year, fitdata.value, marker="o", markersize=3, label=f"{country_name} (selected/editable)")
        compare_errors = []
        for cn in compare_names:
            try:
                cdf, _ = cached_wb(COUNTRIES[cn], code, start_default, end_default)
                if cdf.empty:
                    compare_errors.append(f"{cn}: no observations"); continue
                comp_ax.plot(cdf.year, cdf.value, marker=".", markersize=3, label=f"{cn} (n={len(cdf)}, {cdf.year.min()}–{cdf.year.max()})")
            except Exception as exc:
                compare_errors.append(f"{cn}: {exc}")
        comp_ax.set_xlabel("Year"); comp_ax.set_ylabel(f"{indicator_name} ({units})"); comp_ax.grid(True,alpha=.25); comp_ax.legend(fontsize=8); comp_fig.tight_layout()
        st.pyplot(comp_fig, use_container_width=True)
        st.caption("Country series may have different start/end years and missing observations; lines connect only available points and are not adjusted for coverage differences.")
        for err in compare_errors: st.warning(err)
    summary = {"country":country_name, "country_code":COUNTRIES[country_name], "indicator":indicator_name, "indicator_code":code, "definition":definition, "units":units, "source_url":source_url, "retrieved":retrieval_date, "data_status":data_status, "degree":degree, "base_year":base, "equation":equation, "r_squared_in_sample":r2, "mae_in_sample":mae, "observation_count":len(fitdata), "historical_range":f"{fitdata.year.min()}–{fitdata.year.max()}", "mean_value":float(fitdata.value.mean()), "holdout":str(h) if h else "unavailable", "analysis_notes":f"Residuals/MAE describe in-sample fit, not prediction intervals. Extrapolation is uncertain. {caveat}"}
    csv_bytes = fitdata[["year","value","fitted_value","residual"]].to_csv(index=False).encode("utf-8")
    xlsx_bytes = make_xlsx(summary, fitdata, batch)
    pdf_bytes = make_pdf(summary, fitdata, batch, fig)
    st.markdown("#### Export current selection and model")
    e1,e2,e3 = st.columns(3)
    e1.download_button("Download CSV", csv_bytes, file_name="historical_data_and_fit.csv", mime="text/csv")
    e2.download_button("Download Excel workbook", xlsx_bytes, file_name="historical_data_and_fit.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    e3.download_button("Download PDF report", pdf_bytes, file_name="historical_data_and_fit.pdf", mime="application/pdf")
    with st.expander("Source and mathematical assumptions"):
        st.write("The World Bank API is the source of original observations. The editable table is kept separate from fetched data; edits are treated as a user-modified scenario. The polynomial is fitted by least squares in centered time t = year − base year. R² and MAE are in-sample fit metrics. A chronological holdout uses the latest 20% when enough training observations remain. No confidence or prediction intervals are claimed.")
        st.write("Schooling and other unlisted indicators are not fabricated. Missing API values remain missing.")
        st.markdown("[World Bank Indicators API documentation](https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation)")

else:
    st.subheader("Function builder & calculus lab")
    ctx = st.selectbox("Context", list(CONTEXTS.keys()))
    preset = CONTEXTS[ctx]
    a,b,c = st.columns(3)
    with a: context_desc = st.text_area("Context description", preset["desc"])
    with b: xname = st.text_input("Independent variable", preset["x"]); xunits = st.text_input("Independent units", preset["xu"])
    with c: yname = st.text_input("Dependent variable", preset["y"]); yunits = st.text_input("Dependent units", preset["yu"])
    function_mode = st.radio("Function source", ["Suggested function", "Curve of best fit from World Bank", "Build with parameters", "Enter my own function"], horizontal=True)
    expr_text = preset["expr"]
    if function_mode == "Curve of best fit from World Bank":
        st.info("This option retrieves the selected World Bank series and fits a polynomial. The resulting model is expressed using x = year − base year.")
        c1,c2,c3 = st.columns(3)
        with c1: hc = st.selectbox("Country for fit", list(COUNTRIES.keys()), key="fit_country")
        with c2: hi = st.selectbox("Indicator for fit", list(WB_INDICATORS.keys()), key="fit_indicator")
        with c3: hd = st.slider("Fit degree", 1, 8, 3, key="fit_degree")
        try:
            hf, _ = get_world_bank(COUNTRIES[hc], WB_INDICATORS[hi][0], date.today().year-70, date.today().year-1)
            hf, hb, hcoef, _, hr2, hmae, _ = fit_polynomial(hf, hd)
            expr_text = " + ".join(f"({coef:.12g})*x**{i}" for i,coef in enumerate(hcoef))
            xname, xunits, yname, yunits = "Years since base year", "years", hi, WB_INDICATORS[hi][1]
            context_desc = f"World Bank {hc}: {hi}; x = year − {hb}. {WB_INDICATORS[hi][2]}"
            st.caption(f"Base year {hb}; observations {len(hf)}; R² in-sample {hr2:.5g}; MAE in-sample {hmae:.5g}.")
        except Exception as exc:
            st.error(f"World Bank fit unavailable: {exc}. Falling back to the context's suggested function.")
    if function_mode == "Build with parameters":
        family = st.selectbox("Function family", ["Polynomial", "Rational", "Exponential", "Logarithmic", "Trigonometric"])
        if family == "Polynomial":
            p1,p2,p3 = st.columns(3)
            with p1: aa = st.number_input("Quadratic coefficient a", value=1.0)
            with p2: bb = st.number_input("Linear coefficient b", value=0.0)
            with p3: cc = st.number_input("Constant c", value=0.0)
            expr_text = f"({aa})*x**2+({bb})*x+({cc})"
        elif family == "Rational":
            p1,p2 = st.columns(2)
            with p1: aa = st.number_input("Numerator scale", value=1.0)
            with p2: bb = st.number_input("Denominator offset", value=1.0)
            expr_text = f"({aa})/(x-({bb}))"
        elif family == "Exponential":
            p1,p2 = st.columns(2)
            with p1: aa = st.number_input("Initial scale", value=1.0)
            with p2: bb = st.number_input("Exponent rate", value=0.2)
            expr_text = f"({aa})*exp(({bb})*x)"
        elif family == "Logarithmic":
            p1,p2 = st.columns(2)
            with p1: aa = st.number_input("Scale", value=1.0)
            with p2: bb = st.number_input("Horizontal shift", value=0.0)
            expr_text = f"({aa})*log(x-({bb}))"
        else:
            p1,p2,p3 = st.columns(3)
            with p1: aa = st.number_input("Amplitude", value=1.0)
            with p2: bb = st.number_input("Angular frequency", value=1.0)
            with p3: cc = st.number_input("Vertical shift", value=0.0)
            expr_text = f"({aa})*sin(({bb})*x)+({cc})"
    if function_mode == "Enter my own function":
        expr_text = st.text_input("Enter f(x)", preset["expr"], help="Examples: x^3-2*x, 1/(x-2), exp(-0.3*x), log(x), sin(2*pi*x). Use ^ or ** for powers.")
    else:
        st.text_input("Current function expression", expr_text, disabled=True)
    try:
        expr = safe_parse(expr_text)
        f1, f2 = sp.diff(expr,X), sp.diff(expr,X,2)
        st.markdown(f"**Function:** \\(f(x)={sp.latex(expr)}\\)")
        st.markdown(f"**First derivative:** \\(f'(x)={sp.latex(f1)}\\)")
        st.markdown(f"**Second derivative:** \\(f''(x)={sp.latex(f2)}\\)")
    except Exception as exc:
        st.error(f"Function input error: {exc}")
        st.stop()
    c1,c2,c3 = st.columns(3)
    with c1: xmin = st.number_input("Analysis interval start", value=-5.0, step=0.5)
    with c2: xmax = st.number_input("Analysis interval end", value=10.0, step=0.5)
    with c3: point = st.slider("Current input value", min_value=float(min(xmin,xmax)), max_value=float(max(xmin,xmax)), value=float(min(max(1.0, min(xmin,xmax)), max(xmin,xmax))), step=0.1)
    if xmin == xmax:
        st.error("Analysis interval endpoints must be different.")
        st.stop()
    lo, hi = min(xmin,xmax), max(xmin,xmax)
    st.markdown("#### Average rate of change")
    p1,p2 = st.columns(2)
    with p1: aa = st.slider("Secant interval start a", min_value=float(lo), max_value=float(hi), value=float(lo), step=max((hi-lo)/100,0.01))
    with p2: bb = st.slider("Secant interval end b", min_value=float(lo), max_value=float(hi), value=float(hi), step=max((hi-lo)/100,0.01))
    if aa == bb: st.warning("Choose different endpoints for the average rate of change.")
    play_col, speed_col, reset_col = st.columns(3)
    with play_col: playing = st.toggle("Automatic playback", value=False)
    with speed_col: speed = st.select_slider("Playback speed", options=["Slow","Normal","Fast"], value="Normal")
    with reset_col:
        if st.button("Reset point to interval start"):
            st.session_state["point_reset"] = float(lo)
            st.rerun()
    if "point_reset" in st.session_state:
        point = float(min(max(st.session_state.pop("point_reset"),lo),hi))
    # Playback uses reruns as user interaction; no hidden timer/thread.
    if playing:
        st.info("Playback is manual-friendly: the point advances on app reruns. Use the point slider or interact with the chart to inspect values; turn playback off to pause.")
    try:
        analysis = calculus_analysis(expr, (lo,hi), point)
        st.metric(f"f({fmt_num(point)}) · {yunits}", fmt_num(analysis["fpoint"], 9))
        st.metric("Instantaneous rate f′(x)", f"{fmt_num(analysis['slope'],8)} {yunits}/{xunits}")
        st.write(f"At x = {fmt_num(point)}, the tangent line is approximately **y = {fmt_num(analysis['fpoint'])} + ({fmt_num(analysis['slope'])})(x − {fmt_num(point)})**.")
        if aa != bb:
            fa = float(eval_expr(expr,np.array([aa]))[0]); fb = float(eval_expr(expr,np.array([bb]))[0])
            avg = (fb-fa)/(bb-aa) if np.isfinite(fa) and np.isfinite(fb) else float("nan")
            st.write(f"Average rate from {aa:g} to {bb:g}: **{fmt_num(avg,8)} {yunits}/{xunits}**. This secant slope is the average change in {yname} per {xname}.")
        else:
            avg = float("nan")
        st.caption(f"Units: f(x) is {yunits}; f′(x) is {yunits} per {xunits}; f″(x) is {yunits} per {xunits}². {context_desc}")
        if course == "Honors Algebra 2":
            st.markdown("#### Algebra 2 interpretation")
            st.write(f"Domain/range are explored numerically on [{lo:g}, {hi:g}]. The sign and shape of the graph show where the output rises or falls. Average rates compare two inputs. End behavior should be interpreted from the function family and may differ outside this interval.")
        elif course == "Precalculus":
            st.markdown("#### Precalculus interpretation")
            st.write(f"The degree and leading coefficient shape polynomial end behavior; higher-degree terms can add turning points. Changing average rates suggest concavity: increasing slopes indicate concave up, decreasing slopes indicate concave down. The interval is [{lo:g}, {hi:g}].")
        else:
            st.markdown("#### Calculus analysis")
            st.write(f"f′(x) = {sp.sstr(analysis['f1'])}; f″(x) = {sp.sstr(analysis['f2'])}. Derivative signs determine increasing/decreasing intervals; second-derivative signs describe concavity.")
            st.write("**Critical points (numerical / symbolic solver results):**")
            if analysis["critical"]:
                for cp,kind,val in analysis["critical"]:
                    st.write(f"x ≈ {cp:.8g}, f(x) ≈ {val:.8g}: {kind} (classified by a local first-derivative sign test).")
            else: st.write("No finite stationary critical points were found by the symbolic solver in this interval. Undefined derivative points may require separate inspection.")
            st.write("**Inflection points:**")
            if analysis["inflections"]:
                for ip,iv in analysis["inflections"]: st.write(f"x ≈ {ip:.8g}, f(x) ≈ {iv:.8g}; second derivative changes sign nearby.")
            else: st.write("No inflection points confirmed by the solver and sign-change check in this interval.")
            # Numerical interval summary
            valid = np.isfinite(analysis["y"])
            if valid.any():
                gx, gy = analysis["grid"][valid], analysis["y"][valid]
                imax, imin = int(np.argmax(gy)), int(np.argmin(gy))
                st.write(f"**Numerical absolute extrema on sampled valid points:** maximum ≈ ({gx[imax]:.6g}, {gy[imax]:.6g}); minimum ≈ ({gx[imin]:.6g}, {gy[imin]:.6g}). These are numerical grid estimates, not exact proof, and may miss singularities.")
            inc = analysis["d1"] > 1e-8; dec = analysis["d1"] < -1e-8
            st.write(f"On the plotted grid, derivative positive at {inc.mean()*100:.1f}% of finite samples and negative at {dec.mean()*100:.1f}%. This is a numerical trend summary, not a symbolic interval proof.")
        st.markdown("#### Interactive graph")
        show_f = st.checkbox("Show f(x)", True)
        show_d1 = st.checkbox("Show f′(x)", course=="Calculus")
        show_d2 = st.checkbox("Show f″(x)", False)
        show_tangent = st.checkbox("Show tangent line", True)
        show_secant = st.checkbox("Show secant line", True)
        fig, ax = plt.subplots(figsize=(10,5))
        grid, yy = analysis["grid"], analysis["y"]
        if show_f: ax.plot(grid,yy,label="f(x)",linewidth=2)
        if show_d1: ax.plot(grid,analysis["d1"],label="f′(x)")
        if show_d2: ax.plot(grid,analysis["d2"],label="f″(x)")
        if show_tangent: ax.plot(grid,analysis["tangent"],"--",label="Tangent at current x")
        if show_secant and aa != bb: ax.plot(grid,analysis["secant"],":",label="Secant through interval endpoints")
        if np.isfinite(analysis["fpoint"]): ax.scatter([point],[analysis["fpoint"]],s=80,marker="o",label="Current point")
        ax.axvline(0, color="gray", linewidth=.6); ax.axhline(0,color="gray",linewidth=.6)
        ax.set_xlabel(f"{xname} ({xunits})"); ax.set_ylabel(f"{yname} ({yunits})")
        ax.grid(True,alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
        st.pyplot(fig,use_container_width=True)
        visual = st.selectbox("Contextual visual", ["Walking person with prosthetic leg","Moving object or projectile","Tank or drug-concentration indicator","Population or cell colony","Thermometer or weather indicator","Business or economic indicator"])
        ratio = (point-lo)/(hi-lo)
        st.progress(float(min(max(ratio,0),1)), text=f"Simulation position: {point:g} {xunits} ({ratio*100:.1f}% across selected interval)")
        st.write(f"**{visual}:** the indicator moves with x. Its normalized position is shown as a progress bar; the displayed output is f(x) = {fmt_num(analysis['fpoint'])} {yunits}. This is a conceptual scaling, not a physical animation or clinical simulation.")
        # Valid domain and discontinuity caveats
        finite_fraction = np.isfinite(analysis["y"]).mean()
        if finite_fraction < .999:
            st.warning(f"Some function values are undefined in the selected interval ({(1-finite_fraction)*100:.1f}% of sampled x values). The graph masks undefined points so it does not draw a continuous line through them. Check denominators and logarithm/square-root domains.")
        if re.search(r"/\s*\(?\s*x|log\(", str(expr)):
            st.caption("Domain restrictions may apply. Rational denominators cannot be zero; logarithm arguments must be positive. The app samples the interval numerically and does not prove the complete real domain.")
        # Function lab export
        summary = {"course":course,"context":ctx,"description":context_desc,"function":str(expr),"first_derivative":str(analysis["f1"]),"second_derivative":str(analysis["f2"]),"interval_start":lo,"interval_end":hi,"current_x":point,"f_current":analysis["fpoint"],"fprime_current":analysis["slope"],"average_rate":avg,"x_units":xunits,"y_units":yunits}
        fdata = pd.DataFrame({"x":grid,"f(x)":analysis["y"],"f'(x)":analysis["d1"],"f''(x)":analysis["d2"]})
        calc_data = pd.DataFrame({"quantity":["f(x)","f'(x)","average rate"],"value":[analysis["fpoint"],analysis["slope"],avg]})
        st.download_button("Download function analysis CSV", fdata.to_csv(index=False).encode(), "function_analysis.csv", "text/csv")
        st.download_button("Download function analysis Excel", make_xlsx(summary,fdata,calc_data), "function_analysis.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Download function analysis PDF", make_pdf(summary,fdata,calc_data,fig), "function_analysis.pdf", "application/pdf")
    except Exception as exc:
        st.error(f"Could not analyze this function on the selected interval: {exc}. Try a simpler expression or a smaller interval.")
