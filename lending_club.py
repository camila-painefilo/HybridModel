# app.py — Lending Club Dashboard (robust version with error display)

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

if "welcome_done" not in st.session_state:
    st.session_state.welcome_done = False


def main():
    # heavy / optional libs imported inside main so ModuleNotFoundError doesn't kill the whole app
    from scipy import stats
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import statsmodels.api as sm
    from pandas.api.types import is_numeric_dtype

    # -------------------- Page & Theme --------------------
    st.set_page_config(page_title="Hybrid Model Agent", page_icon="💳", layout="wide")
    alt.data_transformers.disable_max_rows()
    pd.set_option("display.max_columns", 200)

    CSS = """
    <style>
    html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial; }
    section.main > div { padding-top: 1rem; }
    .hero{background:linear-gradient(135deg,#0ea5e9 0%,#8b5cf6 100%);color:white;border-radius:20px;padding:24px;box-shadow:0 8px 30px rgba(27,31,35,.15)}
    .card{background:white;border-radius:16px;padding:18px;box-shadow:0 10px 30px rgba(0,0,0,.06);border:1px solid rgba(0,0,0,.04);margin-bottom:12px}
    .kpi{border-radius:16px;padding:14px 16px;background:#f8fafc;border:1px solid #e5e7eb}
    .kpi .label{font-size:.92rem;color:#475569}.kpi .value{font-size:1.35rem;font-weight:700;color:#0f172a}
    </style>
    """
    st.markdown(CSS, unsafe_allow_html=True)

    # -------------------- HERO + WELCOME --------------------
    TITLE = "Hybrid Model Agent"
    SUBTITLE = "Unified EDA, statistical tests, and predictive modeling in one intelligent agent"
    LOGO_URL = "https://github.com/altyn02/lending_club/releases/download/lending_photo/lending.webp"

    st.markdown(
        f"""
        <div class="hero">
          <div style="display:flex; align-items:center; gap:20px; flex-wrap:wrap;">
            <img src="{LOGO_URL}" alt="Logo"
                 style="height:56px; border-radius:8px;">
            <div>
              <div style="font-size:2rem; font-weight:800; line-height:1.2;">
                {TITLE}
              </div>
              <div style="opacity:.95; margin-top:6px; font-size:1.05rem;">
                {SUBTITLE}
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    # -------------------- 💠 WELCOME PAGE --------------------
    if not st.session_state.welcome_done:
        st.markdown(
            """
            <div style='text-align:center; margin-top:40px;'>
                <h1 style='font-size:42px;'>💳 Welcome to the Hybrid Model Agent</h1>
                <p style='font-size:20px; color:gray;'>
                    A powerful and intelligent platform for LendingClub prediction, EDA, and hybrid modeling.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        start = st.button("🚀 Start the Analysis")
        if start:
            st.session_state.welcome_done = True
            st.rerun()

        st.stop()  # ⛔ prevents the rest of the dashboard from loading

    # -------------------- 1. Data upload --------------------
    st.markdown("## 1. Data upload")
    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="Upload the dataset you want to analyze (CSV format)."
    )

    @st.cache_data(show_spinner=True)
    def load_uploaded_csv(file) -> pd.DataFrame:
        return pd.read_csv(file, low_memory=False)

    if uploaded_file is None:
        st.info("Please upload a CSV file to start the analysis.")
        st.stop()

    # Load raw data
    df_raw = load_uploaded_csv(uploaded_file)
    rows_before, cols_before = df_raw.shape

    # Generic cleaning: drop fully empty rows and columns
    df_full = df_raw.dropna(how="all")
    df_full = df_full.dropna(axis=1, how="all")

    rows_after, cols_after = df_full.shape
    removed_rows = rows_before - rows_after
    removed_cols = cols_before - cols_after

    msg = "Data upload completed."
    if removed_rows > 0 or removed_cols > 0:
        msg += f" Removed {removed_rows} completely empty rows and {removed_cols} completely empty columns."
    st.success(msg)

    # -------------------- 2. Analysis settings --------------------
    st.markdown("## 2. Analysis settings")

    c1, c2, c3 = st.columns([2, 1, 1])

    # 1) Target column
    with c1:
        target_col = st.selectbox(
            "Select target (label) column",
            options=df_full.columns,
            help="If the selected column is not binary (0/1), you will be able to map which values correspond to target=1 and target=0."
        )

    target_raw = df_full[target_col]
    unique_vals = target_raw.dropna().unique()
    labels = [str(v) for v in unique_vals]
    label_to_val = dict(zip(labels, unique_vals))

    # detect already-binary
    try:
        numeric_unique = pd.to_numeric(unique_vals, errors="coerce")
        is_binary = (
            len(unique_vals) == 2
            and set(pd.Series(numeric_unique).dropna().astype(int).unique()) <= {0, 1}
        )
    except Exception:
        is_binary = False

    # -------------------- Target mapping UI --------------------
    st.sidebar.markdown("### Configure binary target mapping")

    default_bad_labels, default_good_labels = [], []
    if is_binary:
        numeric_map = dict(zip(labels, numeric_unique))
        for lab, val in numeric_map.items():
            if pd.isna(val):
                continue
            if int(val) == 1:
                default_bad_labels.append(lab)
            elif int(val) == 0:
                default_good_labels.append(lab)

    st.sidebar.caption(
        "Select which values should be considered BAD (target=1) and GOOD (target=0). "
        "Values not selected in either group will be ignored in modeling."
    )

    bad_labels = st.sidebar.multiselect(
        "BAD class values (target = 1):",
        options=sorted(labels),
        default=sorted(default_bad_labels),
        help="Choose values that represent default, churn, bad loans, etc."
    )

    good_labels = st.sidebar.multiselect(
        "GOOD class values (target = 0):",
        options=sorted(labels),
        default=sorted(default_good_labels),
        help="Choose values that represent fully paid, retained customers, etc."
    )

    conflict = set(bad_labels) & set(good_labels)
    if conflict:
        st.sidebar.error(f"The following values are assigned to both 1 and 0: {conflict}. Please fix this.")
        st.stop()

    bad_vals = [label_to_val[l] for l in bad_labels]
    good_vals = [label_to_val[l] for l in good_labels]

    df_full["target"] = pd.NA
    df_full.loc[target_raw.isin(bad_vals), "target"] = 1
    df_full.loc[target_raw.isin(good_vals), "target"] = 0
    df_full["target"] = pd.to_numeric(df_full["target"], errors="coerce")

    if df_full["target"].isna().all():
        st.sidebar.error("No valid target mapping. Please assign at least one value to 1 or 0.")
        st.stop()
    if df_full["target"].nunique() < 2:
        st.sidebar.error("Target has only one class after mapping. Please assign values to both 1 and 0.")
        st.stop()

    st.sidebar.success(
        f"Mapped {len(bad_vals)} values to target=1 (BAD) and {len(good_vals)} values to target=0 (GOOD). "
        "Unselected values will be ignored."
    )

    # 2) Other analysis settings
    with c2:
        test_size = st.slider(
            "Test data ratio",
            min_value=0.10,
            max_value=0.50,
            value=0.30,
            step=0.05,
            help="Proportion of data reserved for testing."
        )

    with c3:
        random_state = st.number_input(
            "Random state",
            min_value=0,
            max_value=10_000,
            value=42,
            step=1,
            help="Controls the randomness in model evaluation."
        )

    missing_strategy = st.radio(
        "Missing value handling (for model training)",
        options=["Impute with mean", "Impute with 0", "Drop rows with missing values"],
        index=0,
        horizontal=True
    )

    max_missing_pct = st.slider(
        "Maximum allowed missing rate per column (%)",
        min_value=0,
        max_value=100,
        value=50,
        step=5,
        help="Columns with a higher percentage of missing values will be excluded from modeling.",
    )

    st.write("")

    # -------------------- Light typing/cleanup --------------------
    def to_float_pct(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.replace("%","", regex=False).str.replace(",","", regex=False).str.strip()
        s = s.str.extract(r"([-+]?\d*\.?\d+)", expand=False)
        return pd.to_numeric(s, errors="coerce")

    for col in ["int_rate", "revol_util", "dti"]:
        if col in df_full and not pd.api.types.is_numeric_dtype(df_full[col]):
            df_full[col] = to_float_pct(df_full[col])

    if "issue_year" not in df_full.columns and "issue_d" in df_full.columns:
        issue_dt = pd.to_datetime(df_full["issue_d"], errors="coerce", format="%b-%Y")
        if issue_dt.isna().all():
            issue_dt = pd.to_datetime(df_full["issue_d"], errors="coerce")
        df_full["issue_year"] = issue_dt.dt.year

    # -------------------- Sidebar Filters --------------------
    with st.sidebar:
        st.subheader("Filters")

        year_range = None
        if "issue_year" in df_full.columns and df_full["issue_year"].notna().any():
            years = pd.to_numeric(df_full["issue_year"], errors="coerce").dropna().astype(int)
            min_year, max_year = int(years.min()), int(years.max())
            year_range = st.slider("Filter by Issue Year", min_value=min_year, max_value=max_year, value=(min_year, max_year))
        else:
            st.caption("No issue_year column found — charts won’t filter by year.")

        grade_sel = None
        if "grade" in df_full.columns:
            opts = sorted(pd.Series(df_full["grade"]).dropna().astype(str).unique().tolist())
            grade_sel = st.multiselect("Grade", options=opts, default=[])

        term_sel = None
        if "term" in df_full.columns:
            term_opts = pd.Series(df_full["term"]).dropna().astype(str).unique().tolist()
            term_sel = st.multiselect("Term", options=term_opts, default=[])

    # -------------------- Apply Filters --------------------
    df = df_full.copy()

    if year_range and "issue_year" in df.columns:
        iy = pd.to_numeric(df["issue_year"], errors="coerce")
        df = df[(iy >= year_range[0]) & (iy <= year_range[1])]
    if grade_sel:
        df = df[df["grade"].astype(str).isin(grade_sel)]
    if term_sel:
        df = df[df["term"].astype(str).isin(term_sel)]

    if "target" in df.columns:
        df["target"] = df["target"].astype("category")

    # -------------------- KPIs --------------------
    total_rows = len(df)
    total_cols = df.shape[1]

    pos_ratio = "—"
    if "target" in df.columns:
        try:
            pos = (
                pd.to_numeric(df["target"], errors="coerce")
                .fillna(0)
                .astype(int) == 1
            ).mean()
            pos_ratio = f"{pos * 100:.1f}%"
        except Exception:
            pass

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(
            f'<div class="kpi"><div class="label">Filtered rows</div>'
            f'<div class="value">{total_rows:,}</div></div>',
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f'<div class="kpi"><div class="label">Columns</div>'
            f'<div class="value">{total_cols}</div></div>',
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f'<div class="kpi"><div class="label">Positive class rate (target = 1)</div>'
            f'<div class="value">{pos_ratio}</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown(
        f"""
        <div style="background:#f1f5f9;padding:10px 14px;border-radius:10px;font-size:0.95rem;">
          🎯 <b>Target column:</b> <code>{target_col}</code><br><br>
          This dashboard uses a binary target where <b>1 = bad outcome (event)</b> and 
          <b>0 = good outcome (non-event)</b>.<br>
          You can configure which raw values map to 0 or 1 in the sidebar; 
          any values not assigned to either class will be <b>ignored</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    # -------------------- EDA variables (fixed) --------------------
    REQUIRED = ["loan_amnt", "int_rate", "delinq_2yrs", "annual_inc", "dti"]
    df_eda = df.copy()
    EDA_VARS = [c for c in REQUIRED if c in df_eda.columns and is_numeric_dtype(df_eda[c])]
    missing = [c for c in REQUIRED if c not in EDA_VARS]
    if missing:
        st.warning("These requested variables are absent or non-numeric in the filtered data: " + ", ".join(missing))

    def get_featured_vars(df_num, k=6):
        numeric_pool = [c for c in df_num.select_dtypes(include=[np.number]).columns if c != "target"]
        target_num = None
        if "target" in df_num.columns:
            t = pd.to_numeric(df_num["target"], errors="coerce")
            if t.notna().sum() >= 2 and t.nunique(dropna=True) >= 2:
                target_num = t
        if target_num is not None and numeric_pool:
            tmp = pd.concat([target_num.rename("target_num"), df_num[numeric_pool]], axis=1).dropna()
            if not tmp.empty:
                cabs = tmp.corr(numeric_only=True)["target_num"].drop("target_num", errors="ignore").abs()
                top_num = cabs.sort_values(ascending=False).index.tolist()[:k]
            else:
                top_num = numeric_pool[:k]
        else:
            top_num = numeric_pool[:k]
        return top_num

    # build modeling matrix helper
    def build_model_matrix(df_in: pd.DataFrame) -> pd.DataFrame:
        if "target" not in df_in.columns:
            return pd.DataFrame()
        num_cols = [c for c in df_in.select_dtypes(include=[np.number]).columns if c != "target"]
        if not num_cols:
            return pd.DataFrame()
        miss_pct = df_in[num_cols].isna().mean() * 100
        keep_cols = miss_pct[miss_pct <= max_missing_pct].index.tolist()
        if not keep_cols:
            return pd.DataFrame()
        d = df_in[["target"] + keep_cols].copy()
        if missing_strategy == "Impute with mean":
            for c in keep_cols:
                d[c] = d[c].fillna(d[c].mean())
        elif missing_strategy == "Impute with 0":
            d[keep_cols] = d[keep_cols].fillna(0)
        else:
            d = d.dropna(subset=keep_cols)
        d["target"] = pd.to_numeric(d["target"], errors="coerce")
        d = d[d["target"].isin([0, 1])]
        return d

    def tree_select_features(X_train, y_train, max_features=15, random_state=42):
        tree = DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=50,
            random_state=random_state,
        )
        tree.fit(X_train, y_train)
        importances = tree.feature_importances_
        pairs = [(f, imp) for f, imp in zip(X_train.columns, importances) if imp > 0]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return [f for f, _ in pairs[:max_features]]

    def stepwise_select_features(X_train, y_train, X_val, y_val, max_features=15):
        remaining = list(X_train.columns)
        selected = []
        best_auc = 0.0
        while remaining and len(selected) < max_features:
            best_candidate = None
            best_candidate_auc = best_auc
            for f in remaining:
                feats = selected + [f]
                model = LogisticRegression(
                    max_iter=1000,
                    solver="liblinear",
                    class_weight="balanced"
                )
                model.fit(X_train[feats], y_train)
                probs = model.predict_proba(X_val[feats])[:, 1]
                auc = roc_auc_score(y_val, probs)
                if auc > best_candidate_auc + 1e-4:
                    best_candidate_auc = auc
                    best_candidate = f
            if best_candidate is None:
                break
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            best_auc = best_candidate_auc
        return selected

    # -------------------- Tabs --------------------
    tab_data, tab_dist, tab_corr, tab_ttest, tab_pred, tab_logit = st.tabs([
        "🧭 Data Exploration",
        "📈 Data Visualization",
        "🧮 Correlation Heatmap",
        "📏 t-Tests",
        "🔮 Prediction Models (Hybrid)",
        "🧠 Model Interpretation"
    ])

    # ========== Data Exploration ==========
    with tab_data:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Data Exploration — quick view")
        st.write("Sample of the dataframe used for visualizations (after filters).")

        SAMPLE_N = 10000
        sample = df if len(df) <= SAMPLE_N else df.sample(SAMPLE_N, random_state=42)

        rows, cols = sample.shape
        avg_missing = sample.isna().mean().mean() * 100

        c1_, c2_, c3_ = st.columns(3)
        with c1_:
            st.metric("Rows (sampled)", f"{rows:,}")
        with c2_:
            st.metric("Columns", f"{cols}")
        with c3_:
            st.metric("Avg missing (sample)", f"{avg_missing:.2f}%")

        st.markdown("#### Column data types")
        dtypes_df = (
            df.dtypes.reset_index().rename(columns={"index": "column", 0: "dtype"})
        )
        dtypes_df["dtype"] = dtypes_df["dtype"].astype(str)
        st.dataframe(dtypes_df, use_container_width=True)

        st.markdown("#### Missing values per column")
        missing_count = df.isna().sum()
        missing_pct_col = df.isna().mean() * 100
        missing_df = pd.DataFrame({
            "column": df.columns,
            "missing_count": missing_count.values,
            "missing_pct (%)": missing_pct_col.values,
        }).sort_values("missing_count", ascending=False)
        missing_df["missing_pct (%)"] = missing_df["missing_pct (%)"].round(2)
        st.dataframe(missing_df, use_container_width=True)

        st.markdown("#### Head (first non-empty rows)")
        head_df = sample.dropna(how="all").head(10)
        st.dataframe(head_df, use_container_width=True)

        st.markdown("#### Statistical summary (`describe`)")
        desc = sample.describe(include="all").T
        for col in desc.columns:
            try:
                desc[col] = pd.to_numeric(desc[col], errors="coerce").round(3).combine_first(desc[col])
            except Exception:
                pass
        st.dataframe(desc, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ========== Distributions ==========
    with tab_dist:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Distributions — Histograms, Boxplots & Line")
        st.caption("🎯 Target legend: 0 = Charged Off, 1 = Fully Paid")

        if not EDA_VARS:
            st.info("No suitable numeric columns from the requested list.")
        else:
            c1_, c2_, c3_, c4_ = st.columns([1, 1, 1, 2])
            with c1_:
                show_hist = st.checkbox("Histograms", value=True, key="dist_show_hist")
            with c2_:
                show_box = st.checkbox("Boxplots", value=True, key="dist_show_box")
            with c3_:
                show_line = st.checkbox("Line", value=True, key="dist_show_line")
            with c4_:
                bins = st.slider("Histogram bins", 10, 80, 40, 5, key="dist_bins")

            HIST_BOX_VARS = [v for v in EDA_VARS if v != "int_rate"]
            src_all = df_eda[EDA_VARS + (["target"] if "target" in df_eda.columns else [])].dropna()

            # Line chart
            if show_line:
                time_candidates = [c for c in ["issue_year", "issue_d"] if c in df.columns]
                if not time_candidates:
                    time_col = None
                    st.info("No time column available (e.g., issue_year or issue_d).")
                else:
                    time_col = st.selectbox(
                        "Time column for line chart",
                        options=time_candidates,
                        index=0,
                        help="Uses issue_year if available; otherwise choose a time-like column."
                    )

                y_var = st.selectbox(
                    "Y variable",
                    options=EDA_VARS,
                    index=(EDA_VARS.index("loan_amnt") if "loan_amnt" in EDA_VARS else 0),
                    key="line_y_var"
                )
                agg_choice = st.selectbox("Aggregation", ["Mean", "Median", "Count"], index=0, key="line_agg")

                if not time_col:
                    st.info("No time column available, so the line chart cannot be drawn.")
                else:
                    df_line = df[[time_col, y_var] + (["target"] if "target" in df.columns else [])].copy()
                    if time_col == "issue_year":
                        df_line["__time__"] = pd.to_numeric(df_line[time_col], errors="coerce")
                    else:
                        dt = pd.to_datetime(df_line[time_col], errors="coerce")
                        df_line["__time__"] = dt.dt.year
                    df_line = df_line.dropna(subset=["__time__"])

                    if agg_choice == "Mean":
                        agg_func, y_enc_title = "mean", f"Mean {y_var}"
                    elif agg_choice == "Median":
                        agg_func, y_enc_title = "median", f"Median {y_var}"
                    else:
                        agg_func, y_enc_title = "count", f"Count ({y_var} non-null)"

                    if "target" in df_line.columns:
                        g = df_line.groupby(["__time__", "target"], observed=False)
                    else:
                        g = df_line.groupby(["__time__"], observed=False)

                    plot_df = (g[y_var].count().reset_index(name="y")
                               if agg_choice == "Count"
                               else g[y_var].agg(agg_func).reset_index(name="y"))

                    enc_color = alt.Color("target:N", title="target") if "target" in plot_df.columns else alt.value(None)

                    line = (
                        alt.Chart(plot_df)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("__time__:O", title=("Issue Year" if time_col == "issue_year" else time_col)),
                            y=alt.Y("y:Q", title=y_enc_title),
                            color=enc_color,
                            tooltip=[alt.Tooltip("__time__:O", title="Time"),
                                     alt.Tooltip("y:Q", title=y_enc_title, format=".2f")]
                                    + ([alt.Tooltip("target:N", title="target")] if "target" in plot_df.columns else [])
                        )
                        .properties(height=300, title=f"{agg_choice} {y_var} by {time_col}")
                    )
                    st.altair_chart(line, use_container_width=True)

            # Histogram
            if show_hist:
                if not HIST_BOX_VARS:
                    st.info("No variables available for histograms.")
                else:
                    h_var = st.selectbox(
                        "Histogram variable",
                        options=HIST_BOX_VARS,
                        index=(HIST_BOX_VARS.index("loan_amnt") if "loan_amnt" in HIST_BOX_VARS else 0),
                        key="hist_var_single",
                    )
                    h_src = df_eda[[h_var] + (["target"] if "target" in df_eda.columns else [])].dropna()

                    if "target" in h_src.columns:
                        hist = (
                            alt.Chart(h_src)
                            .mark_bar(opacity=0.6)
                            .encode(
                                x=alt.X(f"{h_var}:Q", bin=alt.Bin(maxbins=bins), title=h_var),
                                y=alt.Y("count():Q", title="Count"),
                                color=alt.Color("target:N", title="target"),
                                tooltip=[alt.Tooltip(f"{h_var}:Q", title=h_var), "count()", alt.Tooltip("target:N", title="target")]
                            )
                            .properties(height=280, title=f"Histogram — {h_var}")
                        )
                    else:
                        hist = (
                            alt.Chart(h_src)
                            .mark_bar(opacity=0.8)
                            .encode(
                                x=alt.X(f"{h_var}:Q", bin=alt.Bin(maxbins=bins), title=h_var),
                                y=alt.Y("count():Q", title="Count"),
                                tooltip=[alt.Tooltip(f"{h_var}:Q", title=h_var), "count()"]
                            )
                            .properties(height=280, title=f"Histogram — {h_var}")
                        )
                    st.altair_chart(hist, use_container_width=True)

            # Boxplot
            if show_box:
                if "target" not in df_eda.columns:
                    st.info("Boxplots require 'target' to group by.")
                elif not HIST_BOX_VARS:
                    st.info("No variables available for boxplots.")
                else:
                    b_var = st.selectbox(
                        "Boxplot variable",
                        options=HIST_BOX_VARS,
                        index=(HIST_BOX_VARS.index("annual_inc") if "annual_inc" in HIST_BOX_VARS else 0),
                        key="box_var_single",
                    )
                    b_src = df_eda[["target", b_var]].dropna()
                    box = (
                        alt.Chart(b_src)
                        .mark_boxplot()
                        .encode(
                            x=alt.X("target:N", title="target"),
                            y=alt.Y(f"{b_var}:Q", title=b_var),
                            color=alt.Color("target:N", legend=None),
                            tooltip=[alt.Tooltip(f"{b_var}:Q", title=b_var), alt.Tooltip("target:N", title="target")]
                        )
                        .properties(height=280, title=f"Boxplot — {b_var} by target")
                    )
                    st.altair_chart(box, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ========== Correlation Heatmap ==========
    with tab_corr:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Correlation Heatmap (numeric only)")

        num_df = df.select_dtypes(include=[np.number]).copy()
        if num_df.empty or num_df.shape[1] < 2:
            st.info("Not enough numeric columns for a correlation heatmap.")
        else:
            defaults = [c for c in EDA_VARS if c in num_df.columns]
            ordered = defaults[:]
            if "target" in num_df.columns and "target" not in ordered:
                ordered = ["target"] + ordered
            if len(ordered) < 2:
                ordered = get_featured_vars(num_df, k=min(8, num_df.shape[1]))

            cmat = num_df[ordered].corr(numeric_only=True)
            corr_df = cmat.reset_index().melt("index")
            corr_df.columns = ["feature_x", "feature_y", "corr"]

            heat = alt.Chart(corr_df).mark_rect().encode(
                x=alt.X("feature_x:O", title="", sort=ordered),
                y=alt.Y("feature_y:O", title="", sort=ordered),
                color=alt.Color("corr:Q", scale=alt.Scale(scheme="blueorange", domain=[-1, 1])),
                tooltip=["feature_x", "feature_y", alt.Tooltip("corr:Q", format=".2f")]
            ).properties(height=420)
            st.altair_chart(heat, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ========== T-tests ==========
    with tab_ttest:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Welch’s t-tests (0 = Charged Off, 1 = Fully Paid)")

        if "target" not in df.columns:
            st.info("No 'target' column found.")
        else:
            tnum = pd.to_numeric(df["target"], errors="coerce")
            mask_valid = tnum.isin([0, 1])
            if mask_valid.sum() < 2 or tnum[mask_valid].nunique() < 2:
                st.info("Both target groups must be present to run t-tests.")
            else:
                VARS = [c for c in EDA_VARS if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
                if not VARS:
                    st.info("No numeric variables available for t-tests.")
                else:
                    apply_fdr = st.checkbox("Apply FDR correction (Benjamini–Hochberg)", value=True)
                    rows_t = []
                    for col in VARS:
                        s = pd.to_numeric(df[col], errors="coerce")
                        d = pd.DataFrame({"y": s, "t": tnum}).dropna()
                        g0 = d.loc[d["t"] == 0, "y"].values
                        g1 = d.loc[d["t"] == 1, "y"].values
                        if len(g0) < 2 or len(g1) < 2:
                            continue
                        m0, m1 = np.mean(g0), np.mean(g1)
                        s0, s1 = np.std(g0, ddof=1), np.std(g1, ddof=1)
                        diff = m1 - m0
                        tstat, pval = stats.ttest_ind(g1, g0, equal_var=False)
                        v0, v1 = s0**2, s1**2
                        se2 = v0/len(g0) + v1/len(g1)
                        df_welch = (se2**2) / (((v0/len(g0))**2)/(len(g0)-1) + ((v1/len(g1))**2)/(len(g1)-1))
                        tcrit = stats.t.ppf(0.975, df_welch)
                        ci_low = diff - tcrit*np.sqrt(se2)
                        ci_high = diff + tcrit*np.sqrt(se2)
                        sp2 = (((len(g0)-1)*v0)+((len(g1)-1)*v1)) / (len(g0)+len(g1)-2)
                        sp = np.sqrt(sp2)
                        d_cohen = diff / sp if np.isfinite(sp) and sp > 0 else np.nan
                        J = 1 - (3 / (4*(len(g0)+len(g1)) - 9))
                        g_hedges = d_cohen * J
                        rows_t.append({
                            "variable": col,
                            "n_0": len(g0), "n_1": len(g1),
                            "mean_0": m0, "mean_1": m1,
                            "std_0": s0, "std_1": s1,
                            "diff_(1-0)": diff,
                            "t": tstat, "df": df_welch, "p_value": pval,
                            "cohen_d": d_cohen, "hedges_g": g_hedges,
                            "ci_low": ci_low, "ci_high": ci_high
                        })
                    if not rows_t:
                        st.info("No valid data to compute t-tests.")
                    else:
                        res = pd.DataFrame(rows_t)
                        if apply_fdr:
                            p = res["p_value"].values
                            m = len(p)
                            order = np.argsort(p)
                            ranks = np.empty_like(order)
                            ranks[order] = np.arange(1, m+1)
                            q = p * m / ranks
                            q_adj = np.minimum.accumulate(q[np.argsort(order)][::-1])[::-1]
                            res["q_value"] = q_adj
                            res["significant"] = res["q_value"] < 0.05
                        else:
                            res["significant"] = res["p_value"] < 0.05
                        st.dataframe(res, use_container_width=True)
                        st.caption("Welch’s t-test with 95% CI and effect sizes.")

        st.markdown('</div>', unsafe_allow_html=True)

    # ========== Prediction Models ==========
    with tab_pred:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Prediction Models — Logistic Regression, Decision Tree & Hybrid")
        st.caption("Target legend — 0: good outcome, 1: bad outcome (as defined in the sidebar mapping).")

        if "target" not in df.columns:
            st.info("No 'target' column found.")
        else:
            d_model = build_model_matrix(df)
            if d_model.empty:
                st.info("Not enough numeric features or valid 0/1 target after preprocessing.")
            else:
                X = d_model.drop(columns=["target"])
                y = d_model["target"].astype(int)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=int(random_state),
                    stratify=y
                )
                st.markdown(f"**Number of numeric features used for modeling:** {X.shape[1]}")

                # Baseline Logit
                st.markdown("### A. Baseline Logistic Regression")
                C_logit = st.slider(
                    "Regularization strength C (Logit)",
                    min_value=0.01,
                    max_value=10.0,
                    value=1.0,
                    step=0.01,
                    key="baseline_logit_C"
                )
                balance_logit = st.checkbox(
                    "Use class_weight='balanced' for Logit",
                    value=True,
                    key="baseline_logit_balanced"
                )
                if st.button("Run baseline Logistic Regression", key="btn_baseline_logit"):
                    logit_pipe = Pipeline([
                        ("scaler", StandardScaler()),
                        ("logit", LogisticRegression(
                            C=C_logit,
                            class_weight=("balanced" if balance_logit else None),
                            max_iter=1000,
                            solver="liblinear",
                        )),
                    ])
                    logit_pipe.fit(X_train, y_train)
                    probs_logit = logit_pipe.predict_proba(X_test)[:, 1]
                    preds_logit = (probs_logit >= 0.5).astype(int)
                    auc_logit = roc_auc_score(y_test, probs_logit)
                    acc_logit = accuracy_score(y_test, preds_logit)
                    f1_logit = f1_score(y_test, preds_logit)
                    st.write(
                        f"**Logit — Test AUC:** {auc_logit:.4f} · "
                        f"**Accuracy:** {acc_logit:.4f} · "
                        f"**F1-score (thr=0.5):** {f1_logit:.4f}"
                    )

                st.divider()

                # Tree
                st.markdown("### B. Decision Tree (rule-based model)")
                max_depth_tree = st.slider(
                    "Max depth (Decision Tree)",
                    min_value=2,
                    max_value=15,
                    value=5,
                    key="tree_max_depth_main"
                )
                min_leaf_tree = st.slider(
                    "Minimum samples per leaf (Decision Tree)",
                    min_value=10,
                    max_value=200,
                    value=50,
                    step=10,
                    key="tree_min_leaf_main"
                )
                if st.button("Run Decision Tree model", key="btn_tree_alone"):
                    tree_clf = DecisionTreeClassifier(
                        max_depth=max_depth_tree,
                        min_samples_leaf=min_leaf_tree,
                        random_state=int(random_state),
                    )
                    tree_clf.fit(X_train, y_train)
                    probs_tree = tree_clf.predict_proba(X_test)[:, 1]
                    preds_tree = (probs_tree >= 0.5).astype(int)
                    auc_tree = roc_auc_score(y_test, probs_tree)
                    acc_tree = accuracy_score(y_test, preds_tree)
                    f1_tree = f1_score(y_test, preds_tree)
                    st.write(
                        f"**Decision Tree — Test AUC:** {auc_tree:.4f} · "
                        f"**Accuracy:** {acc_tree:.4f} · "
                        f"**F1-score (thr=0.5):** {f1_tree:.4f}"
                    )

                st.divider()

                # Hybrid Tree → Logit
                st.markdown("### C. Hybrid Model — Tree-selected Logistic Regression")
                st.caption(
                    "Step 1: Use a Decision Tree to find the most important features. "
                    "Step 2: Train a Logistic Regression only on those features."
                )
                max_allowed_feats = min(20, X.shape[1])
                min_allowed_feats = min(3, max_allowed_feats)
                max_feats_tree = st.slider(
                    "Maximum number of features selected from the tree",
                    min_value=min_allowed_feats,
                    max_value=max_allowed_feats,
                    value=min_allowed_feats,
                    key="tree_hybrid_max_feats"
                )
                if st.button("Run Hybrid Model (Tree → Logit)", key="btn_tree_logit_hybrid"):
                    tree_clf2 = DecisionTreeClassifier(
                        max_depth=5,
                        min_samples_leaf=50,
                        random_state=int(random_state),
                    )
                    tree_clf2.fit(X_train, y_train)
                    feats_tree = tree_select_features(
                        X_train, y_train,
                        max_features=max_feats_tree,
                        random_state=int(random_state),
                    )
                    if not feats_tree:
                        st.warning("The decision tree did not select any informative features.")
                    else:
                        st.write(f"**Features selected by the Decision Tree** "f"({len(feats_tree)} features): "+ ", ".join(feats_tree))
                        hybrid_pipe = Pipeline([
                            ("scaler", StandardScaler()),
                            ("logit", LogisticRegression(
                                C=1.0,
                                class_weight="balanced",
                                max_iter=1000,
                                solver="liblinear",
                            )),
                        ])
                        hybrid_pipe.fit(X_train[feats_tree], y_train)
                        probs_hybrid = hybrid_pipe.predict_proba(X_test[feats_tree])[:, 1]
                        preds_hybrid = (probs_hybrid >= 0.5).astype(int)
                        auc_hybrid = roc_auc_score(y_test, probs_hybrid)
                        acc_hybrid = accuracy_score(y_test, preds_hybrid)
                        f1_hybrid = f1_score(y_test, preds_hybrid)
                        comp_rows = [
                            {"Model": "Baseline Logit (all features)", "AUC": None, "Accuracy": None, "F1": None},
                            {"Model": "Decision Tree", "AUC": None, "Accuracy": None, "F1": None},
                            {"Model": "Hybrid (Tree-selected Logit)", "AUC": auc_hybrid, "Accuracy": acc_hybrid, "F1": f1_hybrid},
                        ]
                        comp_df = pd.DataFrame(comp_rows)
                        st.write(
                            f"**Hybrid — Test AUC:** {auc_hybrid:.4f} · "
                            f"**Accuracy:** {acc_hybrid:.4f} · "
                            f"**F1-score (thr=0.5):** {f1_hybrid:.4f}"
                        )
                        st.subheader("Hybrid vs other models (AUC / Accuracy / F1)")
                        st.dataframe(
                            comp_df.style.format(
                                {"AUC": "{:.4f}", "Accuracy": "{:.4f}", "F1": "{:.4f}"}
                            ),
                            use_container_width=True
                        ) 
                        # 🔥 Final comparative analysis block (professor requirement)
                        from sklearn.metrics import confusion_matrix, precision_score, recall_score
                        
                        st.divider()
                        st.subheader("📌 Final Comparative Analysis (All Models)")
                        
                        comp_rows = []
                        
                        # Baseline Logit
                        if "preds_logit" in locals():
                            cm = confusion_matrix(y_test, preds_logit)
                            comp_rows.append({
                                "Model": "Baseline Logistic Regression",
                                "AUC": auc_logit,
                                "Accuracy": acc_logit,
                                "Precision": precision_score(y_test, preds_logit, zero_division=0),
                                "Recall": recall_score(y_test, preds_logit, zero_division=0),
                                "Confusion Matrix": cm
                            })
                        
                        # Stepwise Logit
                        if "preds_sw" in locals():
                            cm = confusion_matrix(y_test_sw, preds_sw)
                            comp_rows.append({
                                "Model": "Stepwise Logistic Regression",
                                "AUC": auc_sw,
                                "Accuracy": acc_sw,
                                "Precision": precision_score(y_test_sw, preds_sw, zero_division=0),
                                "Recall": recall_score(y_test_sw, preds_sw, zero_division=0),
                                "Confusion Matrix": cm
                            })
                        
                        # Decision Tree
                        if "preds_tree" in locals():
                            cm = confusion_matrix(y_test, preds_tree)
                            comp_rows.append({
                                "Model": "Decision Tree",
                                "AUC": auc_tree,
                                "Accuracy": acc_tree,
                                "Precision": precision_score(y_test, preds_tree, zero_division=0),
                                "Recall": recall_score(y_test, preds_tree, zero_division=0),
                                "Confusion Matrix": cm
                            })
                        
                        if comp_rows:
                            comp_df_all = pd.DataFrame(comp_rows)
                            st.dataframe(comp_df_all, use_container_width=True)
                            st.success("Comparative analysis completed successfully.")
                        else:
                            st.info("Run at least two models to compare them.")
                                            
                        

                # Stepwise (expander)
                with st.expander("Advanced: Forward stepwise feature selection + Logistic Regression"):
                    d_model_sw = build_model_matrix(df)
                    if d_model_sw.empty:
                        st.info("Not enough numeric features or valid target to run stepwise selection.")
                    else:
                        from sklearn.model_selection import train_test_split as tts
                        X_sw = d_model_sw.drop(columns=["target"])
                        y_sw = d_model_sw["target"].astype(int)
                        X_train_sw, X_temp_sw, y_train_sw, y_temp_sw = tts(
                            X_sw, y_sw,
                            test_size=test_size,
                            random_state=int(random_state),
                            stratify=y_sw
                        )
                        X_val_sw, X_test_sw, y_val_sw, y_test_sw = tts(
                            X_temp_sw, y_temp_sw,
                            test_size=0.5,
                            random_state=int(random_state),
                            stratify=y_temp_sw
                        )
                        st.write(f"Feature pool size for stepwise: **{X_sw.shape[1]}** numeric columns.")
                        max_allowed_sw = min(20, X_sw.shape[1])
                        min_allowed_sw = min(3, max_allowed_sw)
                        max_feats_sw = st.slider(
                            "Maximum number of features to select (stepwise)",
                            min_value=min_allowed_sw,
                            max_value=max_allowed_sw,
                            value=min_allowed_sw,
                            key="stepwise_max_feats_main"
                        )
                        if st.button("Run stepwise + Logit", key="btn_stepwise_main"):
                            feats_sw = stepwise_select_features(
                                X_train_sw, y_train_sw, X_val_sw, y_val_sw,
                                max_features=max_feats_sw
                            )
                            if not feats_sw:
                                st.warning("Stepwise did not find any feature that improves AUC over baseline.")
                            else:
                                st.write("**Stepwise-selected features:**", ", ".join(feats_sw))
                                model_sw = LogisticRegression(
                                    max_iter=1000,
                                    solver="liblinear",
                                    class_weight="balanced"
                                )
                                model_sw.fit(X_train_sw[feats_sw], y_train_sw)
                                probs_sw = model_sw.predict_proba(X_test_sw[feats_sw])[:, 1]
                                preds_sw = (probs_sw >= 0.5).astype(int)
                                auc_sw = roc_auc_score(y_test_sw, probs_sw)
                                acc_sw = accuracy_score(y_test_sw, preds_sw)
                                f1_sw = f1_score(y_test_sw, preds_sw)
                                st.write(
                                    f"**Stepwise Logit — Test AUC:** {auc_sw:.4f} · "
                                    f"**Accuracy:** {acc_sw:.4f} · "
                                    f"**F1-score (thr=0.5):** {f1_sw:.4f}"
                                )

        st.markdown('</div>', unsafe_allow_html=True)

    # ========== Logit Interpretation ==========
    with tab_logit:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Logistic Regression — Interpret the model")
        st.caption("Target legend — 0: Charged Off, 1: Fully Paid")

        if "target" not in df.columns:
            st.info("No 'target' column found.")
        else:
            numeric_pool = [c for c in df.select_dtypes(include=[np.number]).columns if c != "target"]
            if len(numeric_pool) == 0:
                st.info("No numeric features available for logit.")
            else:
                default_pool = [c for c in ["int_rate", "dti", "revol_util", "loan_amnt", "annual_inc"]
                                if c in numeric_pool] or numeric_pool[:8]
                with st.expander("⚙️ Model settings", expanded=False):
                    C = st.slider("Regularization strength (C)", 0.01, 10.0, 1.0, 0.01)
                    balance = st.checkbox("Class weight = 'balanced'", value=True)
                    max_k = min(12, len(default_pool))
                    min_k = min(3, max_k)
                    top_k = st.slider(
                        "Auto-select top-k features (by |coef|)",
                        min_value=min_k,
                        max_value=max_k,
                        value=min_k
                    )
                    feats_override = st.multiselect("(Optional) Manually choose features",
                                                    options=numeric_pool,
                                                    default=default_pool)

                try:
                    base_feats = feats_override if feats_override else default_pool
                    dtrain0 = df[["target"] + base_feats].dropna().copy()
                    X0 = dtrain0[base_feats].values
                    y0 = pd.to_numeric(dtrain0["target"], errors="coerce").astype(int).values
                    base_pipe = Pipeline([
                        ("scaler", StandardScaler()),
                        ("logit", LogisticRegression(C=C, class_weight=("balanced" if balance else None),
                                                     solver="liblinear", max_iter=400))
                    ])
                    base_pipe.fit(X0, y0)
                    init_coefs = base_pipe.named_steps["logit"].coef_.ravel()
                    order_idx = np.argsort(-np.abs(init_coefs))
                    feats = [base_feats[i] for i in order_idx[:top_k]]
                    dtrain = df[["target"] + feats].dropna().copy()
                    X = dtrain[feats].values
                    y = pd.to_numeric(dtrain["target"], errors="coerce").astype(int).values
                    pipe = Pipeline([
                        ("scaler", StandardScaler()),
                        ("logit", LogisticRegression(C=C, class_weight=("balanced" if balance else None),
                                                     solver="liblinear", max_iter=400))
                    ])
                    pipe.fit(X, y)
                    probs = pipe.predict_proba(X)[:, 1]
                    clf = pipe.named_steps["logit"]
                except Exception as e:
                    st.info("Scikit-learn / statsmodels are required for this tab.")
                    st.exception(e)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.stop()

                visual = st.radio("Visual type",
                                  ["Odds-ratio forest", "Probability vs one feature", "Interaction heatmap"],
                                  horizontal=True)

                if visual == "Odds-ratio forest":
                    coefs = clf.coef_.ravel()
                    odds = np.exp(coefs)
                    ci_low = ci_high = None
                    try:
                        Z = StandardScaler().fit_transform(dtrain[feats].values)
                        Z = sm.add_constant(Z)
                        sm_mod = sm.Logit(y, Z).fit(disp=False)
                        params = sm_mod.params[1:]
                        cov = sm_mod.cov_params().values[1:, 1:]
                        se = np.sqrt(np.diag(cov))
                        ci_low = np.exp(params - 1.96 * se)
                        ci_high = np.exp(params + 1.96 * se)
                    except Exception:
                        pass

                    coef_df = pd.DataFrame({"feature": feats, "odds_ratio": odds}).sort_values("odds_ratio", ascending=False)
                    if ci_low is not None:
                        coef_df["ci_low"] = ci_low
                        coef_df["ci_high"] = ci_high
                    base_chart = alt.Chart(coef_df).encode(y=alt.Y("feature:N", sort="-x", title=""))
                    bars = base_chart.mark_bar(size=10).encode(
                        x=alt.X("odds_ratio:Q", title="Odds Ratio (exp(coef))"),
                        tooltip=["feature", alt.Tooltip("odds_ratio:Q", format=".2f")]
                    )
                    if "ci_low" in coef_df.columns:
                        rules = base_chart.mark_rule().encode(
                            x="ci_low:Q", x2="ci_high:Q",
                            tooltip=["feature",
                                     alt.Tooltip("odds_ratio:Q", format=".2f"),
                                     alt.Tooltip("ci_low:Q", format=".2f"),
                                     alt.Tooltip("ci_high:Q", format=".2f")]
                        )
                        chart = bars + rules
                    else:
                        chart = bars
                    st.markdown("**Feature effects (Odds Ratios)** — > 1 increases odds of target=1; < 1 decreases.")
                    st.altair_chart(chart.properties(height=360), use_container_width=True)

                elif visual == "Probability vs one feature":
                    feasible = [f for f in feats if f in EDA_VARS] or feats
                    one_x = feasible[0]
                    plot_df = dtrain[[one_x]].copy()
                    plot_df["p1"] = probs
                    bins = np.linspace(plot_df[one_x].min(), plot_df[one_x].max(), 31)
                    plot_df["bin"] = pd.cut(plot_df[one_x], bins=bins, include_lowest=True)
                    line_df = plot_df.groupby("bin", observed=False).agg(
                        x=(one_x, "mean"), p=("p1", "mean"), n=("p1", "size")
                    ).dropna()
                    line = alt.Chart(line_df).mark_line(point=True).encode(
                        x=alt.X("x:Q", title=one_x),
                        y=alt.Y("p:Q", title="Mean P(target=1)"),
                        size=alt.Size("n:Q", legend=None, title="Bin size"),
                        tooltip=[alt.Tooltip("x:Q", format=".2f"),
                                 alt.Tooltip("p:Q", format=".3f"),
                                 "n:Q"]
                    ).properties(height=360)
                    st.altair_chart(line, use_container_width=True)

                else:  # interaction
                    if len(feats) < 2:
                        st.info("Need at least two features for an interaction.")
                    else:
                        f1_, f2_ = feats[0], feats[1]
                        tmp = dtrain[[f1_, f2_]].copy()
                        tmp["p1"] = probs
                        bx = pd.cut(tmp[f1_], bins=20, include_lowest=True)
                        by = pd.cut(tmp[f2_], bins=20, include_lowest=True)
                        grid = tmp.groupby([bx, by], observed=False)["p1"].mean().reset_index()
                        grid.columns = [f1_, f2_, "p"]

                        def mid(iv):
                            try:
                                return (iv.left + iv.right) / 2
                            except Exception:
                                return np.nan

                        grid["x"] = grid[f1_].apply(mid)
                        grid["y"] = grid[f2_].apply(mid)
                        grid = grid.dropna()
                        heat = alt.Chart(grid).mark_rect().encode(
                            x=alt.X("x:Q", title=f1_),
                            y=alt.Y("y:Q", title=f2_),
                            color=alt.Color("p:Q", title="Mean P(target=1)"),
                            tooltip=[alt.Tooltip("x:Q", format=".2f"),
                                     alt.Tooltip("y:Q", format=".2f"),
                                     alt.Tooltip("p:Q", format=".3f")]
                        ).properties(height=420)
                        st.altair_chart(heat, use_container_width=True)

                st.caption("Exploratory interpretation only • Standardized features • No performance metrics shown.")
        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------- Footer --------------------
    st.write("")
    st.markdown(
        """
        <div style="text-align:center; color:#64748b; font-size:.9rem; padding:10px 0 0 0;">
          Camila and Altynsara ✅
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # This will show the *real* Python error instead of the generic "Oh no" page
        st.error("The app crashed while running. Here is the detailed error message:")
        st.exception(e)
