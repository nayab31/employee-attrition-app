import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve

import core

st.set_page_config(page_title="Employee Attrition Risk", page_icon="👥", layout="wide")


# ------------------------------------------------------------------ data loading
@st.cache_data(show_spinner=False)
def read_upload(name, data):
    if name.lower().endswith(".csv"):
        try:
            return pd.read_csv(io.BytesIO(data))
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(data), encoding="latin-1")
    return pd.read_excel(io.BytesIO(data))


@st.cache_data(show_spinner=False)
def sample_csv():
    return core.generate_data(1500, seed=11).to_csv(index=False)


@st.cache_resource(show_spinner="Training Linear, Logistic and XGBoost models on demo data...")
def load_demo():
    return core.build_bundle(core.generate_data(2500), "Built-in SYNTHETIC demo data (2,500 employees x 4 quarters)")


# ------------------------------------------------------------------ sidebar
st.sidebar.title("👥 Attrition Risk")
source = st.sidebar.radio("Data source", ["Upload my own file", "Demo data (synthetic)"])
bundle = load_demo() if source.startswith("Demo") else st.session_state.get("bundle_upload")

PAGES = ["1. Executive Overview", "2. Department Breakdown", "3. EDA", "4. Model Insights",
         "5. Retention Plan & Predictor", "6. Data, Cleaning & Model Card"]
if source.startswith("Upload"):
    PAGES = ["0. Upload & map data"] + PAGES
page = st.sidebar.radio("Dashboard page", PAGES)
st.sidebar.warning("Scores support employee retention only. Never use them to penalize employees or "
                   "make adverse decisions.")
st.sidebar.caption(f"Data refresh: {pd.Timestamp.today():%d %b %Y}")

# ------------------------------------------------------------------ Page 0: upload
if page.startswith("0"):
    st.title("Upload & map your data")
    st.markdown("Upload a **CSV or Excel** file with one row per employee (or per employee snapshot). "
                "It needs a column saying whether the employee left, and at least 3 predictor columns. "
                "**Do not include** gender, age, race, religion or marital status.")
    st.caption("Privacy: your file is processed in this browser session's memory only and is not saved "
               "by this app. Do not upload data you are not authorized to share.")
    st.download_button("Download a sample CSV to try the upload", sample_csv(), "sample_employee_data.csv",
                       mime="text/csv")
    f = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
    if f is None:
        st.info("No file yet. Upload one above, or switch to 'Demo data' in the sidebar to see how the app works.")
    else:
        try:
            raw_up = read_upload(f.name, f.getvalue())
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            raw_up = None
        if raw_up is not None:
            st.success(f"Read **{f.name}**: {raw_up.shape[0]:,} rows x {raw_up.shape[1]} columns")
            st.dataframe(raw_up.head(10), use_container_width=True)
            st.subheader("Match your columns to the fields")
            guess = core.guess_mapping(list(raw_up.columns))
            options = ["(none)"] + list(raw_up.columns)
            mapping = {}
            cols = st.columns(2)
            for i, (field, (desc, _)) in enumerate(core.STANDARD_FIELDS.items()):
                g = guess.get(field)
                with cols[i % 2]:
                    ch = st.selectbox(f"{field} - {desc}", options,
                                      index=options.index(g) if g in options else 0,
                                      key=f"map_{f.name}_{field}")
                mapping[field] = None if ch == "(none)" else ch
            positive = None
            if mapping.get("Attrition"):
                vals = raw_up[mapping["Attrition"]].dropna().unique()
                if len(vals) > 20:
                    st.error("The Attrition column has more than 20 different values. Pick a Yes/No or 1/0 column.")
                else:
                    vals = sorted(vals, key=str)
                    strs = [str(v) for v in vals]
                    positive = st.selectbox("Which value means the employee LEFT?", strs,
                                            index=strs.index(core.guess_positive_value(strs)),
                                            key=f"pos_{f.name}")
            if st.button("Validate & train models", type="primary"):
                try:
                    with st.spinner("Cleaning data and training Linear, Logistic and XGBoost models..."):
                        std, notes = core.standardize(raw_up, mapping, positive)
                        st.session_state["bundle_upload"] = core.build_bundle(
                            std, f"Uploaded file: {f.name}", notes, mapping)
                    st.success("Done. Open the dashboard pages in the sidebar.")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Something went wrong while training: {e}")
    if st.session_state.get("bundle_upload") is not None:
        st.caption(f"Currently loaded: {st.session_state['bundle_upload']['label']}")

elif bundle is None:
    st.title("No data loaded yet")
    st.info("Go to **0. Upload & map data** in the sidebar, upload your file and train the models.")

else:
    clean, clean_log, res, importance = bundle["clean"], bundle["log"], bundle["res"], bundle["imp"]
    best, thr_default, xcols = res["best"], res["threshold"], res["xcols"]
    has = lambda c: c in clean.columns
    st.sidebar.caption(f"Data: {bundle['label']}")
    if not res["uses_real_xgboost"]:
        st.sidebar.error("xgboost not installed: a gradient-boosting fallback is being used.")
    latest = clean[clean.Snapshot_Quarter == clean.Snapshot_Quarter.max()] if has("Snapshot_Quarter") else clean

    # -------------------------------------------------------------- Page 1
    if page.startswith("1"):
        st.title("Employee Turnover Prediction & Talent Retention Intelligence")
        st.caption("HR Analytics | Supervised binary classification | Target: employee leaves")
        p_all = core.predict_proba(res["models"][best], latest[xcols])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Voluntary attrition rate", f"{clean[core.TARGET].mean():.1%}")
        c2.metric("Employees (latest snapshot)", f"{len(latest):,}")
        c3.metric("Flight-risk count", f"{int((p_all >= thr_default).sum()):,}")
        c4.metric("Best model", best.split(" (")[0])
        if has("Snapshot_Quarter") and clean["Snapshot_Quarter"].nunique() > 1:
            q = clean.groupby("Snapshot_Quarter")[core.TARGET].mean() * 100
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.bar([f"{int(i)}" for i in q.index], q.values, color="#4C78A8")
            ax.set_xlabel("Snapshot period"); ax.set_ylabel("Attrition rate (%)")
            ax.set_title("Voluntary attrition by snapshot period"); st.pyplot(fig)
        st.info("Business objective: reduce voluntary turnover of critical knowledge workers by 15% per year. "
                "Replacing a skilled employee costs 50-150% of annual salary.")

    # -------------------------------------------------------------- Page 2
    elif page.startswith("2"):
        st.title("Department Breakdown")
        d = clean
        if has("Department"):
            depts = sorted(clean["Department"].unique())
            pick = st.multiselect("Departments", depts, default=depts)
            d = clean[clean.Department.isin(pick)]
        if not (has("Department") or has("Salary_Band")):
            st.info("Map a Department and/or Salary band column on the upload page to see this page.")
        c1, c2 = st.columns(2)
        if has("Department"):
            with c1:
                g = d.groupby("Department")[core.TARGET].mean() * 100
                fig, ax = plt.subplots(figsize=(5, 3.2)); ax.bar(g.index.astype(str), g.values, color="#F58518")
                ax.set_ylabel("Attrition rate (%)"); ax.set_title("Turnover by department")
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right"); st.pyplot(fig)
        if has("Salary_Band"):
            with c2:
                g = d.groupby("Salary_Band")[core.TARGET].mean() * 100
                fig, ax = plt.subplots(figsize=(5, 3.2)); ax.bar(g.index.astype(str), g.values, color="#54A24B")
                ax.set_xlabel("Salary band"); ax.set_ylabel("Attrition rate (%)")
                ax.set_title("Turnover by salary band"); st.pyplot(fig)
        if has("Department") and has("Salary_Band"):
            st.subheader("Department x salary band heatmap (attrition %)")
            pv = d.pivot_table(index="Department", columns="Salary_Band", values=core.TARGET, aggfunc="mean") * 100
            fig, ax = plt.subplots(figsize=(6, 3))
            im = ax.imshow(pv.values, cmap="Reds", aspect="auto")
            ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels(pv.columns)
            ax.set_yticks(range(len(pv.index))); ax.set_yticklabels(pv.index)
            for i in range(pv.shape[0]):
                for j in range(pv.shape[1]):
                    if not np.isnan(pv.values[i, j]):
                        ax.text(j, i, f"{pv.values[i, j]:.0f}", ha="center", va="center", fontsize=8)
            fig.colorbar(im, label="%"); ax.set_xlabel("Salary band"); st.pyplot(fig)

    # -------------------------------------------------------------- Page 3
    elif page.startswith("3"):
        st.title("Exploratory Analysis")
        st.caption("Association is not causation: these patterns do not prove that overtime or a missed "
                   "promotion causes resignation.")
        if has("Overtime_Hours") and has("Job_Satisfaction_Score"):
            s = clean.sample(min(2500, len(clean)), random_state=1)
            binary = set(clean["Overtime_Hours"].unique()) <= {0, 1, 0.0, 1.0}
            fig, ax = plt.subplots(figsize=(7, 4))
            if binary:
                g = clean.groupby(["Job_Satisfaction_Score", "Overtime_Hours"])[core.TARGET].mean().unstack() * 100
                g.plot(kind="bar", ax=ax); ax.legend(["No overtime", "Overtime"][: g.shape[1]])
                ax.set_ylabel("Attrition (%)"); ax.set_title("Attrition by satisfaction and overtime flag")
            else:
                jit = np.random.default_rng(0).normal(0, 0.08, len(s))
                ax.scatter(s.Job_Satisfaction_Score + jit, s.Overtime_Hours,
                           c=s[core.TARGET].map({0: "#9ecae9", 1: "#d62728"}), s=8, alpha=0.6)
                ax.set_xlabel("Job satisfaction"); ax.set_ylabel("Overtime hours")
                ax.set_title("Overtime vs. satisfaction (red = left)")
            st.pyplot(fig)
        c1, c2 = st.columns(2)
        if has("Years_Since_Last_Promotion"):
            with c1:
                mx = max(float(clean.Years_Since_Last_Promotion.max()), 6.0)
                b = pd.cut(clean.Years_Since_Last_Promotion, [-0.1, 1, 2, 3, 5, mx + 1],
                           labels=["0-1", "1-2", "2-3", "3-5", "5+"])
                g = clean.groupby(b, observed=True)[core.TARGET].mean() * 100
                fig, ax = plt.subplots(figsize=(5, 3.2)); ax.bar(g.index.astype(str), g.values, color="#B279A2")
                ax.set_xlabel("Years since last promotion"); ax.set_ylabel("Attrition (%)")
                ax.set_title("Promotion gap vs. attrition"); st.pyplot(fig)
        with c2:
            numcols = [c for c in res["num"] if not c.endswith("_Missing_Flag")]
            corr = clean[numcols + [core.TARGET]].corr()[core.TARGET].drop(core.TARGET).sort_values()
            fig, ax = plt.subplots(figsize=(5, 3.2)); ax.barh(corr.index, corr.values, color="#4C78A8")
            ax.set_title("Correlation with attrition"); st.pyplot(fig)

    # -------------------------------------------------------------- Page 4
    elif page.startswith("4"):
        st.title("Model Insights")
        test, tp = res["test"], res["test_p"]
        st.subheader("Model comparison on held-out test set")
        st.dataframe(res["results"].round(3), use_container_width=True)
        st.caption(res["split_note"] + " The Linear Regression is a linear probability model (outputs "
                   "clipped to 0-1). Accuracy is not reported: attrition is the minority class.")
        names = list(res["models"])
        model_name = st.selectbox("Model to inspect", names, index=names.index(best))
        p = tp[model_name]
        thr = st.slider("Decision threshold", 0.05, 0.9, float(min(max(round(thr_default, 2), 0.05), 0.9)), 0.01)
        y = test[core.TARGET].values
        c1, c2 = st.columns(2)
        with c1:
            cm = confusion_matrix(y, p >= thr, labels=[0, 1])
            fig, ax = plt.subplots(figsize=(4, 3.4)); ax.imshow(cm, cmap="Blues")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, v, ha="center", va="center", fontsize=14)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["Stay", "Leave"]); ax.set_yticks([0, 1])
            ax.set_yticklabels(["Stay", "Leave"]); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title("Confusion matrix"); st.pyplot(fig)
            tn, fp, fn, tpos = cm.ravel()
            st.write(f"Precision **{tpos / max(tpos + fp, 1):.2f}** | Recall **{tpos / max(tpos + fn, 1):.2f}**")
        with c2:
            pr, rc, _ = precision_recall_curve(y, p)
            fig, ax = plt.subplots(figsize=(4.5, 3.4)); ax.plot(rc, pr, color="#4C78A8")
            ax.axhline(y.mean(), ls="--", color="grey", label="No-skill")
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall curve")
            ax.legend(); st.pyplot(fig)
        c3, c4 = st.columns(2)
        with c3:
            fr, mp = calibration_curve(y, np.clip(p, 0, 1), n_bins=6, strategy="quantile")
            fig, ax = plt.subplots(figsize=(4.5, 3.4)); ax.plot(mp, fr, "o-", label="Model")
            ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect")
            ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed rate")
            ax.set_title("Calibration curve"); ax.legend(); st.pyplot(fig)
        with c4:
            top = importance.head(8)[::-1]
            fig, ax = plt.subplots(figsize=(4.5, 3.4)); ax.barh(top.index, top.values, color="#F58518")
            ax.set_xlabel("Drop in ROC-AUC when shuffled"); ax.set_title(f"Top drivers ({best.split(' (')[0]})")
            st.pyplot(fig)
        st.subheader("Subgroup performance")
        sg = core.subgroup_table(test, p, thr)
        if len(sg):
            st.dataframe(sg.round(2), use_container_width=True)
        else:
            st.caption("Map Department or Salary band to see subgroup performance.")
        st.subheader("Error analysis")
        st.markdown("- **False positives:** engaged employees working overtime for a project they love.\n"
                    "- **False negatives:** disengaged staff working normal hours who quietly job-hunt.\n"
                    "- Protected attributes (gender, age, race, religion, marital status) are not used.")

    # -------------------------------------------------------------- Page 5
    elif page.startswith("5"):
        st.title("Retention Plan & Risk Predictor")
        st.error("For HR support use only. Never use a score to penalize an employee, deny a bonus or "
                 "promotion, or automate any adverse decision. Individual scores must be hidden from "
                 "direct line managers.")
        st.subheader("Predict flight risk for one employee profile")
        base_cols = [c for c in core.NUM_BASE if has(c)]
        vals, cols = {}, st.columns(3)
        i = 0
        if has("Department"):
            with cols[i % 3]:
                vals["Department"] = st.selectbox("Department", sorted(clean["Department"].unique()))
            i += 1
        for c in base_cols:
            s = clean[c]
            with cols[i % 3]:
                label = c.replace("_", " ")
                if set(s.unique()) <= {0, 1, 0.0, 1.0}:
                    vals[c] = 1.0 if st.selectbox(label + " (flag)", ["No", "Yes"]) == "Yes" else 0.0
                elif c == "Monthly_Salary":
                    vals[c] = st.number_input(label, float(s.min()), float(s.max()) * 2, float(s.median()),
                                              float(max(round(s.median() / 50), 1)))
                else:
                    lo, hi = float(s.quantile(0.01)), float(s.quantile(0.99))
                    if hi <= lo:
                        hi = lo + 1.0
                    is_int = bool(np.allclose(s, np.round(s)))
                    if is_int:
                        lo, hi, dv, step = int(lo), int(np.ceil(hi)), int(round(s.median())), 1
                        dv = min(max(dv, lo), hi)
                        vals[c] = float(st.slider(label, lo, hi, dv, step))
                    else:
                        dv = min(max(float(s.median()), lo), hi)
                        vals[c] = st.slider(label, float(lo), float(hi), float(dv), float((hi - lo) / 100))
            i += 1
        thr = st.slider("Flag threshold", 0.05, 0.9, float(min(max(round(thr_default, 2), 0.05), 0.9)), 0.01,
                        key="pthr")
        if st.button("Predict risk", type="primary"):
            row = core.score_frame(clean, vals, xcols)
            out = {n: float(core.predict_proba(m, row)[0]) for n, m in res["models"].items()}
            for col, (n, v) in zip(st.columns(len(out)), out.items()):
                col.metric(n.split(" (")[0], f"{v:.1%}")
            main = out[best]
            if main >= thr:
                st.warning(f"Flagged (best model {main:.1%} >= {thr:.0%}). Suggested action: HR Business "
                           "Partner offers a supportive career conversation within 14 days.")
            else:
                st.success(f"Below threshold ({main:.1%}). No action needed.")
            if "Market_Comp_Gap" in row.columns:
                st.caption(f"Compensation vs. peer median: {float(row['Market_Comp_Gap'].iloc[0]):+.1%}")
        st.divider()
        st.subheader("Strategic action plan")
        st.dataframe(pd.DataFrame([
            ["Compensation equity review: market adjustment for high performers >15% below median",
             "Total Rewards Lead", "Top-tier attrition < 5%"],
            ["Workload and burnout rebalancing: manager check-ins and overtime caps above 15 hrs/week",
             "People Operations", "Team pulse score > 4.0 / 5"],
            ["Career-path conversations for staff with 3+ years since promotion",
             "Department Heads + HRBP", "Promotion-stalled attrition -20%"],
        ], columns=["Intervention", "Owner", "KPI"]), use_container_width=True, hide_index=True)
        st.caption("Pay raises alone will not fix burnout: combine compensation with workload and career action.")

    # -------------------------------------------------------------- Page 6
    else:
        st.title("Data, Cleaning & Model Card")
        st.subheader("Data source and gate decision")
        if bundle["label"].startswith("Built-in"):
            st.markdown("**Source:** SYNTHETIC data generated by the app. It is not real enterprise data and "
                        "proves nothing about real attrition. **Gate decision: Approved with Limits** "
                        "(educational use only).")
        else:
            st.markdown(f"**Source:** {bundle['label']}. Record its provenance, license, collection date and "
                        "access authorization in your report. **Gate decision: Approved with Limits** by "
                        "default; change it in your report if your data review says otherwise.")
            if bundle["mapping"]:
                st.write("Column mapping used:")
                st.dataframe(pd.DataFrame([(k, v) for k, v in bundle["mapping"].items() if v],
                                          columns=["Field", "Your column"]), hide_index=True)
            for n in bundle["notes"]:
                st.caption(n)
        st.subheader("Quality audit")
        st.dataframe(core.quality_audit(bundle["raw"]), use_container_width=True)
        st.subheader("Immutable cleaning log")
        st.dataframe(clean_log, use_container_width=True, hide_index=True)
        st.download_button("Download cleaning log (CSV)", clean_log.to_csv(index=False), "cleaning_log.csv")
        st.subheader("Split design")
        st.markdown(f"{res['split_note']} Rows: train {len(res['train']):,}, validation {len(res['val']):,}, "
                    f"test {len(res['test']):,}.")
        st.subheader("Feature register")
        reg = [[c, "Input column (known before the snapshot date)"] for c in core.NUM_BASE if has(c)]
        if has("Department"):
            reg.append(["Department", "Input column (one-hot encoded)"])
        if has("Promotion_Stagnation_Ratio"):
            reg.append(["Promotion_Stagnation_Ratio", "Years_Since_Last_Promotion / (Years_At_Company + 1)"])
        if has("Workload_Overdrive_Index"):
            reg.append(["Workload_Overdrive_Index", "Overtime_Hours / Job_Satisfaction_Score"])
        if has("Market_Comp_Gap"):
            reg.append(["Market_Comp_Gap", "(Monthly_Salary - peer median) / peer median"])
        st.dataframe(pd.DataFrame(reg, columns=["Feature", "Definition"]), hide_index=True)
        st.subheader("Model card")
        st.markdown("""
- **Models:** Linear Regression (linear probability model), Logistic Regression, XGBoost.
- **Intended use:** supporting HR Business Partners in offering retention conversations and planning budgets.
- **Out of scope:** discipline, firing, pay or promotion decisions, performance reviews, sharing individual scores with line managers.
- **Metrics:** ROC-AUC, PR-AUC, Brier score, recall at 80% precision. Accuracy is not used.
- **Limitations:** association is not causation; results depend on the data supplied; small test sets give noisy metrics; synthetic demo results do not transfer to real firms.
- **Monitoring:** retrain quarterly; investigate if ROC-AUC drops by more than 0.05 or the attrition rate shifts by more than 3 points.
""")
