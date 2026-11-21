"""Core logic: Employee Turnover Prediction & Talent Retention Intelligence (Project 7).

Works on (a) built-in SYNTHETIC demo data or (b) a file the user uploads and maps.
Models: Linear Regression (linear probability model), Logistic Regression, XGBoost.
"""
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:  # fallback only so the code still runs where xgboost is absent
    HAS_XGB = False

TARGET = "Attrition"
DEPARTMENTS = ["R&D", "Sales", "HR", "Finance"]

# standard field -> (description, aliases used to auto-guess the user's column)
STANDARD_FIELDS = {
    "Attrition": ("REQUIRED: did the employee leave? (target)",
                  ["attrition", "left", "resigned", "turnover", "exited", "quit", "churn"]),
    "Employee_ID": ("Employee identifier (used only for grouped splitting, never as a feature)",
                    ["employeeid", "employeenumber", "empid", "staffid", "employeecode", "id"]),
    "Snapshot_Quarter": ("Snapshot period, e.g. 1-4 (enables the grouped temporal split)",
                         ["snapshotquarter", "quarter", "snapshot", "period"]),
    "Department": ("Department / business unit", ["department", "dept", "businessunit", "function"]),
    "Salary_Band": ("Pay grade or job level", ["salaryband", "joblevel", "grade", "band", "level"]),
    "Monthly_Salary": ("Monthly pay", ["monthlysalary", "monthlyincome", "salary", "income", "monthlypay"]),
    "Years_At_Company": ("Tenure in years", ["yearsatcompany", "tenure", "yearsofservice", "tenureyears"]),
    "Overtime_Hours": ("Overtime hours (a Yes/No column also works)",
                       ["overtimehours", "overtime", "overtimehrs", "ot"]),
    "Job_Satisfaction_Score": ("Satisfaction survey score",
                               ["jobsatisfactionscore", "jobsatisfaction", "satisfaction",
                                "satisfactionlevel", "engagement"]),
    "Years_Since_Last_Promotion": ("Years since last promotion",
                                   ["yearssincelastpromotion", "yearssincepromotion", "promotiongap",
                                    "lastpromotion"]),
    "Commute_Distance_KM": ("Commute distance", ["commutedistancekm", "distancefromhome",
                                                 "commutedistance", "commute"]),
}
NUM_BASE = ["Salary_Band", "Monthly_Salary", "Years_At_Company", "Overtime_Hours",
            "Job_Satisfaction_Score", "Years_Since_Last_Promotion", "Commute_Distance_KM"]
ENGINEERED = ["Promotion_Stagnation_Ratio", "Workload_Overdrive_Index", "Market_Comp_Gap"]
PROTECTED = {"gender", "sex", "race", "religion", "maritalstatus", "age", "ethnicity",
             "nationality", "disability"}


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# ---------------------------------------------------------------- mapping / standardising
def guess_mapping(columns):
    """Auto-guess which uploaded column matches each standard field."""
    normed = {c: _norm(c) for c in columns}
    used, guess = set(), {}
    for field, (_, aliases) in STANDARD_FIELDS.items():
        guess[field] = None
        for a in aliases:
            hit = [c for c, n in normed.items() if n == a and c not in used]
            if hit:
                guess[field] = hit[0]
                used.add(hit[0])
                break
    return guess


def guess_positive_value(values):
    vals = [str(v) for v in values]
    for pref in ["yes", "1", "true", "y", "left", "resigned", "1.0"]:
        for v in vals:
            if v.strip().lower() == pref:
                return v
    return vals[-1] if vals else None


def standardize(raw, mapping, positive_value):
    """Turn an uploaded table into the standard schema. Returns (df, notes)."""
    notes = []
    mapped = {k: v for k, v in mapping.items() if v}
    if "Attrition" not in mapped:
        raise ValueError("Choose the column that says whether the employee left (Attrition).")
    if positive_value is None:
        raise ValueError("Choose which value in the Attrition column means 'the employee left'.")
    if len(set(mapped.values())) != len(mapped):
        raise ValueError("Each uploaded column can be used for only one field.")
    bad = [c for c in mapped.values() if _norm(c) in PROTECTED]
    if bad:
        raise ValueError(f"Column(s) {bad} look like protected attributes (gender, age, race, religion, "
                         "marital status...). They must not be used in an attrition model. Remove them "
                         "from the mapping.")
    out = pd.DataFrame(index=raw.index)
    ycol = raw[mapped["Attrition"]]
    out[TARGET] = (ycol.astype(str).str.strip().str.lower()
                   == str(positive_value).strip().lower()).astype(int)
    for f in ["Employee_ID", "Department"]:
        if f in mapped:
            s = raw[mapped[f]].astype("object").where(raw[mapped[f]].notna(), np.nan)
            out[f] = s.map(lambda v: v if pd.isna(v) else str(v).strip())
    for f in ["Snapshot_Quarter"] + NUM_BASE:
        if f not in mapped:
            continue
        s = raw[mapped[f]]
        if f == "Overtime_Hours":
            low = s.astype(str).str.strip().str.lower()
            nn = s.notna()
            if nn.any() and low[nn].isin(["yes", "no", "y", "n", "true", "false"]).mean() > 0.9:
                s = low.map({"yes": 1, "y": 1, "true": 1, "no": 0, "n": 0, "false": 0})
                notes.append("Overtime was Yes/No: converted to 1/0 (overtime flag).")
        num = pd.to_numeric(s, errors="coerce")
        if f == "Snapshot_Quarter" and num.notna().mean() < 0.5:
            num = pd.to_numeric(s.astype(str).str.extract(r"(\d+)")[0], errors="coerce")
        if f == "Salary_Band" and num.notna().mean() < 0.5:
            num = pd.Series(pd.factorize(s.astype(str).str.strip(), sort=True)[0] + 1, index=raw.index)
            notes.append("Salary band was text: converted to ordered codes 1..n (alphabetical order; check it).")
        if num.notna().sum() == 0:
            raise ValueError(f"Column '{mapped[f]}' has no numeric values, so it cannot be used for {f}.")
        out[f] = num
    out = out[ycol.notna()].reset_index(drop=True)
    pos, neg = int(out[TARGET].sum()), int((1 - out[TARGET]).sum())
    if len(out) < 200 or pos < 60 or neg < 60:
        raise ValueError(f"Not enough data: {len(out)} rows, {pos} leavers, {neg} stayers. "
                         "Need at least 200 rows with 60+ leavers and 60+ stayers.")
    if len([f for f in out.columns if f in NUM_BASE + ["Department"]]) < 3:
        raise ValueError("Map at least 3 predictor fields (for example satisfaction, tenure, salary).")
    notes.append(f"{len(out):,} rows, {pos:,} leavers ({pos / len(out):.1%}).")
    return out, notes


# ---------------------------------------------------------------- synthetic demo data
def generate_data(n_employees=2500, seed=42, target_rate=0.16):
    """SYNTHETIC HR snapshots: n_employees x 4 quarters (already in the standard schema)."""
    rng = np.random.default_rng(seed)
    emp = pd.DataFrame({
        "Employee_ID": [f"E{100000 + i}" for i in range(n_employees)],
        "Department": rng.choice(DEPARTMENTS, n_employees, p=[0.35, 0.35, 0.12, 0.18]),
        "Salary_Band": rng.choice([1, 2, 3, 4, 5], n_employees, p=[0.2, 0.3, 0.25, 0.15, 0.1]),
        "Years_At_Company": np.round(rng.gamma(2.2, 2.6, n_employees).clip(0.2, 30), 1),
        "Commute_Distance_KM": np.round(rng.gamma(2.0, 7.0, n_employees).clip(1, 80), 1),
        "base_ot": rng.gamma(2.0, 6.0, n_employees),
        "base_sat": rng.normal(3.3, 0.9, n_employees),
    })
    emp["Years_Since_Last_Promotion"] = np.round(
        np.minimum(rng.gamma(1.6, 1.4, n_employees), emp["Years_At_Company"]), 1)
    emp["base_salary"] = (30000 + emp["Salary_Band"] * 14000
                          + emp["Department"].map({"R&D": 6000, "Sales": 2000, "HR": -3000, "Finance": 3000})
                          + rng.normal(0, 3500, n_employees)).round(-2)
    frames = []
    for q in range(1, 5):
        d = emp.copy()
        d["Snapshot_Quarter"] = q
        d["Overtime_Hours"] = (d["base_ot"] + rng.normal(0, 2.5, n_employees)).clip(0, 60).round(1)
        d["Job_Satisfaction_Score"] = np.clip(np.round(d["base_sat"] - 0.002 * d["Overtime_Hours"]
                                                       + rng.normal(0, 0.4, n_employees)), 1, 5).astype(float)
        d["Monthly_Salary"] = (d["base_salary"] / 12 * (1 + 0.005 * (q - 1))).round(0)
        d["Years_At_Company"] = d["Years_At_Company"] + 0.25 * (q - 1)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    gap = add_features(df)["Market_Comp_Gap"]
    logit = (0.07 * df["Overtime_Hours"] + 0.75 * (3 - df["Job_Satisfaction_Score"])
             + 0.45 * df["Years_Since_Last_Promotion"] - 0.30 * df["Salary_Band"]
             + 0.012 * df["Commute_Distance_KM"] - 2.0 * gap
             + df["Department"].map({"R&D": 0.2, "Sales": 0.45, "HR": -0.2, "Finance": 0.0})
             + 0.4 * ((df["Years_At_Company"] > 0.8) & (df["Years_At_Company"] < 1.6))
             + rng.normal(0, 0.3, len(df)))
    lo, hi = -15.0, 5.0
    for _ in range(60):  # bisection on the intercept to hit the target attrition rate
        mid = (lo + hi) / 2
        rate = (1 / (1 + np.exp(-(logit + mid)))).mean()
        lo, hi = (mid, hi) if rate < target_rate else (lo, mid)
    df[TARGET] = (rng.random(len(df)) < 1 / (1 + np.exp(-(logit + mid)))).astype(int)
    miss = rng.random(len(df)) < np.where(df[TARGET] == 1, 0.08, 0.03)  # leavers skip surveys (MNAR)
    df.loc[miss, "Job_Satisfaction_Score"] = np.nan
    keep = ["Employee_ID", "Snapshot_Quarter", "Department", "Salary_Band", "Monthly_Salary",
            "Years_At_Company", "Overtime_Hours", "Job_Satisfaction_Score",
            "Years_Since_Last_Promotion", "Commute_Distance_KM", TARGET]
    return df[keep].reset_index(drop=True)


# ---------------------------------------------------------------- features / audit / cleaning
def add_features(df):
    """Leakage-safe engineered features, built only when their inputs exist."""
    df = df.copy()
    c = df.columns
    if "Years_Since_Last_Promotion" in c and "Years_At_Company" in c:
        df["Promotion_Stagnation_Ratio"] = df["Years_Since_Last_Promotion"] / (df["Years_At_Company"] + 1)
    if "Overtime_Hours" in c and "Job_Satisfaction_Score" in c:
        sat = df["Job_Satisfaction_Score"].fillna(df["Job_Satisfaction_Score"].median()).clip(lower=1)
        df["Workload_Overdrive_Index"] = df["Overtime_Hours"] / sat
    if "Monthly_Salary" in c:
        keys = [k for k in ["Department", "Salary_Band"] if k in c]
        if keys:
            med = df.groupby(keys)["Monthly_Salary"].transform("median")
        else:
            med = df["Monthly_Salary"].median()
        df["Market_Comp_Gap"] = (df["Monthly_Salary"] - med) / med
    return df


def quality_audit(df):
    return pd.DataFrame({"Missing": df.isna().sum(), "Missing %": (df.isna().mean() * 100).round(2),
                         "Unique values": df.nunique()})


def clean(df):
    """Returns (clean_df, cleaning_log). The input is never modified."""
    out, log = df.copy(), []
    keys = [k for k in ["Employee_ID", "Snapshot_Quarter"] if k in out.columns]
    if keys:
        n = int(out.duplicated(keys).sum())
        out = out.drop_duplicates(keys)
        log.append([f"Duplicate {' + '.join(keys)}", n, "Dropped", "Same employee-snapshot twice",
                    f"{int(out.duplicated(keys).sum())} duplicates remain"])
    else:
        n = int(out.duplicated().sum())
        log.append(["Exact duplicate rows (no ID column)", n, "Kept, reported only",
                    "Identical rows can be genuine without an ID to tell them apart", "n/a"])
    bad = pd.Series(False, index=out.index)
    for col in ["Years_At_Company", "Overtime_Hours", "Commute_Distance_KM", "Years_Since_Last_Promotion"]:
        if col in out.columns:
            bad |= out[col] < 0
    if "Monthly_Salary" in out.columns:
        bad |= out["Monthly_Salary"] <= 0
    log.append(["Impossible values (negative tenure/overtime/commute/promotion gap, salary <= 0)",
                int(bad.sum()), "Quarantined and dropped", "Recording errors, not real employees",
                "No impossible values remain"])
    out = out[~bad]
    for col in NUM_BASE:
        if col in out.columns and out[col].isna().any():
            n, med = int(out[col].isna().sum()), out[col].median()
            out[f"{col}_Missing_Flag"] = out[col].isna().astype(int)
            out[col] = out[col].fillna(med)
            why = ("Leavers often skip surveys (likely MNAR): median fill plus a missing flag keeps the signal"
                   if col == "Job_Satisfaction_Score" else "Median is robust to skew; flag keeps the signal")
            log.append([f"Missing {col}", n, f"Median impute ({med:.2f}) + missing flag", why,
                        f"{int(out[col].isna().sum())} nulls remain"])
    if "Department" in out.columns and out["Department"].isna().any():
        n = int(out["Department"].isna().sum())
        out["Department"] = out["Department"].fillna("Missing")
        log.append(["Missing Department", n, "Separate 'Missing' category", "Text field: never guess a department",
                    "0 nulls remain"])
    out = add_features(out.reset_index(drop=True))
    return out, pd.DataFrame(log, columns=["Anomaly", "Rows affected", "Remediation",
                                           "Business justification", "Post-clean validation"])


def feature_lists(df):
    num = [c for c in NUM_BASE + ENGINEERED if c in df.columns]
    num += [c for c in df.columns if c.endswith("_Missing_Flag")]
    cat = ["Department"] if "Department" in df.columns else []
    return num, cat


# ---------------------------------------------------------------- modelling
def split(df):
    """Grouped temporal split when Employee_ID and 3+ quarters exist, else stratified random."""
    if "Employee_ID" in df.columns and "Snapshot_Quarter" in df.columns and df["Snapshot_Quarter"].nunique() >= 3:
        qs = sorted(df["Snapshot_Quarter"].unique())
        trq = qs[:-2] if len(qs) >= 4 else [qs[0]]
        ids = np.array(sorted(df["Employee_ID"].unique()), dtype=object)
        np.random.default_rng(7).shuffle(ids)
        n = len(ids)
        tr, va, te = ids[: int(.70 * n)], ids[int(.70 * n): int(.85 * n)], ids[int(.85 * n):]
        train = df[df.Employee_ID.isin(tr) & df.Snapshot_Quarter.isin(trq)]
        val = df[df.Employee_ID.isin(va) & (df.Snapshot_Quarter == qs[-2])]
        test = df[df.Employee_ID.isin(te) & (df.Snapshot_Quarter == qs[-1])]
        if min(len(train), len(val), len(test)) > 0 and all(d[TARGET].nunique() == 2 for d in (train, val, test)):
            note = (f"Grouped temporal split: train = 70% of employees, quarter(s) {list(trq)}; validation = "
                    f"unseen employees, quarter {qs[-2]}; test = unseen employees, quarter {qs[-1]}. "
                    "No employee appears in more than one split.")
            return train, val, test, note
    tv, test = train_test_split(df, test_size=0.15, stratify=df[TARGET], random_state=7)
    train, val = train_test_split(tv, test_size=0.15 / 0.85, stratify=tv[TARGET], random_state=7)
    note = ("LIMITATION: stratified random split (70/15/15), because the file has no employee ID with 3+ "
            "snapshot periods. This is weaker than the handbook's grouped temporal split.")
    return train, val, test, note


def _prep(num, cat):
    parts = [("num", StandardScaler(), num)]
    if cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat))
    return ColumnTransformer(parts)


def make_models(num, cat, pos_weight=1.0):
    if HAS_XGB:
        xgb = XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.9,
                            colsample_bytree=0.9, scale_pos_weight=pos_weight,
                            eval_metric="logloss", random_state=42)
    else:
        xgb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250, random_state=42)
    return {
        "Linear Regression (probability model)": Pipeline([("p", _prep(num, cat)), ("m", LinearRegression())]),
        "Logistic Regression": Pipeline([("p", _prep(num, cat)), ("m", LogisticRegression(max_iter=2000))]),
        "XGBoost": Pipeline([("p", _prep(num, cat)), ("m", xgb)]),
    }


def predict_proba(model, X):
    if isinstance(model.named_steps["m"], LinearRegression):
        return np.clip(model.predict(X), 0, 1)
    return model.predict_proba(X)[:, 1]


def recall_at_precision(y, p, target=0.80):
    prec, rec, _ = precision_recall_curve(y, p)
    ok = prec >= target
    return float(rec[ok].max()) if ok.any() else 0.0


def evaluate(y, p):
    return {"ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p),
            "Brier": brier_score_loss(y, np.clip(p, 0, 1)),
            "Recall @ 80% precision": recall_at_precision(y, p)}


def baseline_rule(df, train):
    """Documented heuristic baseline, chosen from the columns available."""
    if "Years_Since_Last_Promotion" in df.columns and "Job_Satisfaction_Score" in df.columns:
        return ("Baseline rule (>3 yrs no promotion & satisfaction <= 2)",
                ((df["Years_Since_Last_Promotion"] > 3) & (df["Job_Satisfaction_Score"] <= 2)).astype(float).values)
    if "Job_Satisfaction_Score" in df.columns:
        return "Baseline rule (satisfaction <= 2)", (df["Job_Satisfaction_Score"] <= 2).astype(float).values
    if "Overtime_Hours" in df.columns:
        cut = train["Overtime_Hours"].quantile(0.75)
        return "Baseline rule (overtime above 75th percentile)", (df["Overtime_Hours"] > cut).astype(float).values
    return None


def best_threshold(y, p, beta=2.0):
    y = np.asarray(y)
    if y.sum() == 0:
        return float(np.mean(p))
    prec, rec, thr = precision_recall_curve(y, p)
    if len(thr) == 0:
        return 0.5
    f = (1 + beta ** 2) * prec[:-1] * rec[:-1] / np.maximum(beta ** 2 * prec[:-1] + rec[:-1], 1e-9)
    return float(thr[int(np.argmax(f))])


def train_all(df_clean):
    num, cat = feature_lists(df_clean)
    xcols = num + cat
    train, val, test, split_note = split(df_clean)
    pw = (train[TARGET] == 0).sum() / max((train[TARGET] == 1).sum(), 1)
    models = make_models(num, cat, pos_weight=min(float(pw), 5.0))
    rows, fitted, test_p, val_p = [], {}, {}, {}
    for name, m in models.items():
        m.fit(train[xcols], train[TARGET])
        fitted[name] = m
        val_p[name] = predict_proba(m, val[xcols])
        test_p[name] = predict_proba(m, test[xcols])
        rows.append({"Model": name, **evaluate(test[TARGET], test_p[name])})
    base = baseline_rule(test, train)
    n_base = 0
    if base:
        rows.insert(0, {"Model": base[0], **evaluate(test[TARGET], base[1])})
        n_base = 1
    results = pd.DataFrame(rows).set_index("Model")
    best = results.iloc[n_base:]["ROC-AUC"].idxmax()
    return {"models": fitted, "results": results, "best": best,
            "threshold": best_threshold(val[TARGET], val_p[best]), "train": train, "val": val,
            "test": test, "test_p": test_p, "xcols": xcols, "num": num, "cat": cat,
            "split_note": split_note, "uses_real_xgboost": HAS_XGB}


def driver_importance(model, X, y, seed=0):
    if len(X) > 3000:
        idx = np.random.default_rng(seed).choice(len(X), 3000, replace=False)
        X, y = X.iloc[idx], np.asarray(y)[idx]
    def scorer(est, Xs, ys):  # works for the linear model too (it has no predict_proba)
        return roc_auc_score(ys, predict_proba(est, Xs))
    r = permutation_importance(model, X, y, scoring=scorer, n_repeats=5, random_state=seed)
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)


def subgroup_table(test, p, thr):
    d = test.copy()
    d["flag"] = np.asarray(p) >= thr
    out = []
    for col in [c for c in ["Department", "Salary_Band"] if c in d.columns]:
        for g, s in d.groupby(col):
            pos = int(s[TARGET].sum())
            tp = int(((s["flag"]) & (s[TARGET] == 1)).sum())
            out.append({"Group": f"{col}={g}", "Employees": len(s), "Actual leavers": pos,
                        "Recall": tp / pos if pos else np.nan, "Flag rate": s["flag"].mean()})
    return pd.DataFrame(out)


def score_frame(clean_df, values, xcols):
    """Build one model-ready row from user-entered base values (engineered features computed)."""
    base_cols = [c for c in NUM_BASE + ["Department"] if c in clean_df.columns]
    tmp = pd.concat([clean_df[base_cols], pd.DataFrame([{c: values[c] for c in base_cols}])],
                    ignore_index=True)
    tmp = add_features(tmp)
    row = tmp.iloc[[-1]].copy()
    for c in xcols:
        if c.endswith("_Missing_Flag"):
            row[c] = 0
    return row[xcols]


def build_bundle(std_df, label, notes=None, mapping=None):
    clean_df, log = clean(std_df)
    res = train_all(clean_df)
    imp = driver_importance(res["models"][res["best"]], res["test"][res["xcols"]], res["test"][TARGET])
    return {"raw": std_df, "clean": clean_df, "log": log, "res": res, "imp": imp, "label": label,
            "notes": notes or [], "mapping": mapping or {}}
