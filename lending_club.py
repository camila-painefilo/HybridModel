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
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import statsmodels.api as sm
    from pandas.api.types import is_numeric_dtype
    from imblearn.over_sampling import SMOTE
    from sklearn.metrics import precision_score, recall_score


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
    SUBTITLE = "Unified EDA, statistical tests, and predictive modeling in One Intelligent Agent"
    LOGO_URL = "https://img.freepik.com/premium-photo/friendly-looking-ai-agent-as-logo-white-background-style-raw-job-id-ef2c5ef7e19b4dadbef969fcb37e_343960-103720.jpg"

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

        col1, col2, col3 = st.columns([1, 0.3, 2])

        with col1:
            st.image("welcome_image.png", width=220)

        with col3:
            st.markdown("## 💳 Welcome to the Hybrid Model Agent")
            st.markdown(
                """
A flexible and intelligent platform for  
- tabular data exploration 📊
- statistical testing 📏
- feature selection 🎯
- hybrid predictive modeling 🤖

Designed for credit scoring, churn prediction, customer analytics,
and any binary classification workflow ⚡
                """
            )

            start = st.button("🚀 Start the Analysis")
            if start:
                st.session_state.welcome_done = True
                st.rerun()

        st.stop()  # ⛔ prevents rest of dashboard from loading

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
        s = series.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip()
        s = s.str.extract(r"([-+]?\d*\.?\d+)", expand=False)
        return pd.to_numeric(s, errors="coerce")

    for col in ["int_rate", "revol_util", "dti"]:
        if col in df_full and not pd.api.types.is_numeric_dtype(df_full[col]):
            df_full[col] = to_float_pct(df_full[col])

    # -------------------- Sidebar Filters --------------------
    with st.sidebar:
        st.subheader("Filters")
    
        # 1) Detect possible date-like columns
        date_candidates = []
        for c in df_full.columns:
            # Already datetime dtype
            if pd.api.types.is_datetime64_any_dtype(df_full[c]):
                date_candidates.append(c)
            # Strings that look like potential date columns
            elif df_full[c].dtype == "object":
                col_lower = c.lower()
                if any(key in col_lower for key in ["date", "time", "issue", "pymnt", "pay", "earliest", "last"]):
                    date_candidates.append(c)
    
        date_candidates = sorted(set(date_candidates))
    
        # 2) Let the user choose a date column manually
        date_col_choice = None
        if date_candidates:
            date_col_choice = st.selectbox(
                "Select a date column to derive 'issue_year' (optional)",
                options=["(None)"] + date_candidates,
                index=0,
                help="If selected, the system extracts the year and creates an 'issue_year' column."
            )
        else:
            st.caption("No date-like columns detected.")
    
        # 3) Create issue_year using intelligent multi-format parsing
        if date_col_choice and date_col_choice != "(None)":
            raw_col = df_full[date_col_choice]
        
            # Attempt 1: Generic parsing (works for most formats)
            parsed_dates = pd.to_datetime(raw_col, errors="coerce")
            
            # Attempt 2: Lending Club format: "Dec-17"
            if parsed_dates.notna().mean() < 0.3:
                parsed_dates = pd.to_datetime(raw_col, format="%b-%y", errors="coerce")
        
            # Attempt 3: Format like "Dec-2017"
            if parsed_dates.notna().mean() < 0.3:
                parsed_dates = pd.to_datetime(raw_col, format="%b-%Y", errors="coerce")
        
            # Final validation
            valid_ratio = parsed_dates.notna().mean()
        
            if valid_ratio >= 0.3:
                df_full["issue_year"] = parsed_dates.dt.year
            else:
                st.warning(
                    f"Column '{date_col_choice}' does not appear to be a valid date column "
                    f"({valid_ratio*100:.1f}% valid dates)."
                )

    
        # 4) Year filter (only appears if issue_year exists)
        year_range = None
        if "issue_year" in df_full.columns and df_full["issue_year"].notna().any():
            years = pd.to_numeric(df_full["issue_year"], errors="coerce").dropna().astype(int)
            if not years.empty:
                min_year, max_year = int(years.min()), int(years.max())
        
                if min_year == max_year:
                    # Only one year -> no need for a slider
                    st.caption(
                        f"Only one year detected in the data: {min_year}. Year filter is disabled."
                    )
                    year_range = None  # no filtering by year
                else:
                    # Proper range slider when there is more than one year
                    year_range = st.slider(
                        "Filter by Issue Year",
                        min_value=min_year,
                        max_value=max_year,
                        value=(min_year, max_year),
                    )
        else:
            st.caption("No 'issue_year' detected — line charts will not use year grouping.")


        
        # 5) Other filters (disabled – no widgets shown)
        # We still define the variables so that the rest of the code does not fail.
        grade_sel = None
        term_sel = None


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
    missing_req = [c for c in REQUIRED if c not in EDA_VARS]
    if missing_req:
        st.warning("These requested variables are absent or non-numeric in the filtered data: " + ", ".join(missing_req))

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

    def undersample_train(X, y, random_state=42):
        """Random undersampling to make classes equally sized in the TRAIN set."""
        df_tmp = X.copy()
        df_tmp["__target__"] = y.values

        counts = df_tmp["__target__"].value_counts()
        min_n = counts.min()

        parts = []
        for cls in counts.index:
            part = df_tmp[df_tmp["__target__"] == cls].sample(
                n=min_n, random_state=random_state
            )
            parts.append(part)

        df_bal = pd.concat(parts).sample(frac=1, random_state=random_state)
        y_bal = df_bal["__target__"].astype(int)
        X_bal = df_bal.drop(columns=["__target__"])
        return X_bal, y_bal

    # -------------------- Tabs --------------------
    tab_data, tab_dist, tab_corr, tab_ttest, tab_balance, tab_pred = st.tabs([
        "🧭 Data Exploration",
        "📈 Data Visualization",
        "🧮 Correlation Heatmap",
        "📏 t-Tests & Stepwise",
        "⚖️ Class Balancing",
        "🔮 Prediction Models (Hybrid)"
    ])

    # ========== Data Exploration ==========
    with tab_data:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Data Exploration — quick view")
        st.write("Sample of the dataframe used for visualizations (after filters).")

        SAMPLE_N = 10000
        sample = df if len(df) <= SAMPLE_N else df.sample(SAMPLE_N, random_state=42)

        rows_s, cols_s = sample.shape
        avg_missing = sample.isna().mean().mean() * 100

        c1_, c2_, c3_ = st.columns(3)
        with c1_:
            st.metric("Rows (sampled)", f"{rows_s:,}")
        with c2_:
            st.metric("Columns", f"{cols_s}")
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
        # Numeric summary
        num_sample = sample.select_dtypes(include=["number"])
        if not num_sample.empty:
            st.markdown("##### Numeric summary")
            desc_num = num_sample.describe().T
            desc_num = desc_num.round(3)
            st.dataframe(desc_num, use_container_width=True)

        # Categorical summary
        cat_sample = sample.select_dtypes(exclude=["number"])
        if not cat_sample.empty:
            st.markdown("##### Categorical summary")
            desc_cat = cat_sample.describe().T
            st.dataframe(desc_cat, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ========== Distributions ==========
    with tab_dist:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Distributions — Histograms, Boxplots & Line")
        st.caption("🎯 Target legend: 0 = good, 1 = bad")

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

                    plot_df = (
                        g[y_var].count().reset_index(name="y")
                        if agg_choice == "Count"
                        else g[y_var].agg(agg_func).reset_index(name="y")
                    )

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

    # ========== T-tests & Stepwise ==========
    with tab_ttest:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("F-test + t-tests (student method: p < 0.05)")

        if "target" not in df.columns:
            st.info("No 'target' column found.")
        else:
            tnum = pd.to_numeric(df["target"], errors="coerce")
            mask_valid = tnum.isin([0, 1])
            if mask_valid.sum() < 2 or tnum[mask_valid].nunique() < 2:
                st.info("Both target groups (0 and 1) must be present to run t-tests.")
            else:
                # 🎯 Use ALL numeric predictors (except target)
                VARS = [
                    c for c in df.columns
                    if c != "target" and pd.api.types.is_numeric_dtype(df[c])
                ]

                if not VARS:
                    st.info("No numeric variables available for t-tests.")
                else:
                    rows_t = []
                    selected_features = []  # variables finally chosen by t-test (p < 0.05)

                    for col in VARS:
                        s = pd.to_numeric(df[col], errors="coerce")
                        d = pd.DataFrame({"y": s, "t": tnum}).dropna()

                        g0 = d.loc[d["t"] == 0, "y"].values  # target = 0
                        g1 = d.loc[d["t"] == 1, "y"].values  # target = 1

                        if len(g0) < 2 or len(g1) < 2:
                            continue

                        m0, m1 = np.mean(g0), np.mean(g1)
                        s0, s1 = np.std(g0, ddof=1), np.std(g1, ddof=1)

                        # 1️⃣ F-test for equality of variances (Levene)
                        f_stat, f_p = stats.levene(g0, g1)

                        # 2️⃣ t-tests (both versions)
                        t_eq, p_eq = stats.ttest_ind(g1, g0, equal_var=True)
                        t_uneq, p_uneq = stats.ttest_ind(g1, g0, equal_var=False)

                        # 3️⃣ choose which t-test to use based on F-test
                        if f_p >= 0.05:
                            used_test = "equal_var"
                            used_t = t_eq
                            used_p = p_eq
                        else:
                            used_test = "unequal_var"
                            used_t = t_uneq
                            used_p = p_uneq

                        is_sig = used_p < 0.05
                        if is_sig:
                            selected_features.append(col)

                        rows_t.append({
                            "variable": col,
                            "n_0": len(g0),
                            "n_1": len(g1),
                            "mean_0": m0,
                            "mean_1": m1,
                            "std_0": s0,
                            "std_1": s1,
                            "F_stat": f_stat,
                            "F_p_value": f_p,
                            "t_equal": t_eq,
                            "t_equal_p": p_eq,
                            "t_unequal": t_uneq,
                            "t_unequal_p": p_uneq,
                            "used_test": used_test,
                            "used_p_value": used_p,
                            "selected(p<0.05)": is_sig,
                        })

                    if not rows_t:
                        st.info("No valid data to compute t-tests.")
                    else:
                        res = pd.DataFrame(rows_t)
                        st.dataframe(res, use_container_width=True)
                        st.caption(
                            "F-test decides whether to use equal-variance or unequal-variance t-test. "
                            "Variables with used p-value < 0.05 are selected."
                        )

                        # ✅ Save selected features from t-test
                        st.session_state["ttest_sig_features"] = selected_features

                        if selected_features:
                            st.success(
                                f"{len(selected_features)} variables selected by t-tests (p < 0.05)."
                            )
                            st.caption(", ".join(selected_features))
                        else:
                            st.warning(
                                "No variables are significant at α = 0.05. "
                                "Stepwise will fall back to all numeric predictors."
                            )

                        st.markdown("---")
                        st.subheader("Forward stepwise logistic regression (based on t-test results)")

                        # ===== Build modeling matrix for stepwise =====
                        d_model_sw = build_model_matrix(df)
                        if d_model_sw.empty:
                            st.info("Not enough numeric features or valid 0/1 target to run stepwise selection.")
                        else:
                            # 🎯 Candidate pool: t-test-selected vars if available
                            if selected_features:
                                candidate_feats = [
                                    c for c in selected_features
                                    if c in d_model_sw.columns and c != "target"
                                ]
                                st.caption(
                                    f"Stepwise candidate pool: {len(candidate_feats)} "
                                    f"variables that passed the t-tests."
                                )
                            else:
                                # Fallback: all numeric predictors
                                candidate_feats = [
                                    c for c in d_model_sw.columns if c != "target"
                                ]
                                st.warning(
                                    "Running stepwise on all numeric predictors because "
                                    "no variables passed the t-tests."
                                )

                            if not candidate_feats or len(candidate_feats) < 2:
                                st.info(
                                    "Not enough candidate features to run stepwise selection "
                                    "(need at least 2 numeric predictors)."
                                )
                            else:
                                X_sw_all = d_model_sw[candidate_feats]
                                y_sw_all = d_model_sw["target"].astype(int)

                                from sklearn.model_selection import train_test_split as tts

                                X_train_sw, X_temp_sw, y_train_sw, y_temp_sw = tts(
                                    X_sw_all, y_sw_all,
                                    test_size=test_size,
                                    random_state=int(random_state),
                                    stratify=y_sw_all
                                )
                                X_val_sw, X_test_sw, y_val_sw, y_test_sw = tts(
                                    X_temp_sw, y_temp_sw,
                                    test_size=0.5,
                                    random_state=int(random_state),
                                    stratify=y_temp_sw
                                )

                                    # 🔧 Stepwise будет рассматривать все кандидатные признаки.
                                    # Он сам остановится, когда добавление новой переменной
                                    # перестанет улучшать AUC на валидации.
                                    max_feats_sw = X_sw_all.shape[1]
                                
                                    st.caption(
                                        f"Stepwise candidate pool: {max_feats_sw} variables that passed the t-tests. "
                                        "The algorithm will automatically decide how many features to keep."
                                    )
                                
                                    if st.button("Run stepwise selection", key="btn_stepwise_ttest_tab"):
                                        feats_sw = stepwise_select_features(
                                            X_train_sw, y_train_sw,
                                            X_val_sw, y_val_sw,
                                            max_features=max_feats_sw,
                                        )
                                        if not feats_sw:
                                            st.warning(
                                                "Stepwise did not find any feature that improves AUC over the baseline."
                                            )
                                        else:
                                            st.success(
                                                f"Stepwise selected {len(feats_sw)} features "
                                                f"(from {max_feats_sw} candidates):\n\n"
                                                + ", ".join(feats_sw)
                                            )
                                
                                            # Save raw stepwise features
                                            st.session_state["stepwise_features"] = feats_sw
                                
                                            # ✅ Final feature set for modeling
                                            final_feats = feats_sw
                                            st.success(
                                                f"Final feature set for modeling (t-test → stepwise): "
                                                f"{len(final_feats)} features."
                                            )
                                            st.caption(", ".join(final_feats))
                                
                                            # Save final feature set for Prediction Models tab
                                            st.session_state["selected_features_for_modeling"] = final_feats
                                            st.caption("✅ Final feature set saved for the Prediction Models tab.")


        st.markdown('</div>', unsafe_allow_html=True)

    # ========== Class Balancing ==========
    with tab_balance:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("⚖️ Class Balancing for Modeling")

        if "target" not in df.columns:
            st.info("No 'target' column found.")
        else:
            t_raw = pd.to_numeric(df["target"], errors="coerce")
            counts = t_raw.value_counts().sort_index()

            # 🎨 Nice-looking summary box 
            st.markdown("""
            <div style="
                padding:15px;
                border-radius:12px;
                background-color:#f7f9fc;
                border:1px solid #e3e6ee;
                margin-bottom:10px;
            ">
                <b>Current target distribution</b> (0 = good, 1 = bad):
            </div>
            """, unsafe_allow_html=True)

            # Two side-by-side cards 
            c1b, c2b = st.columns(2)
            with c1b:
                st.metric("Class 0 count", f"{counts.get(0, 0):,}")
            with c2b:
                st.metric("Class 1 count", f"{counts.get(1, 0):,}")

            st.markdown("---")

            # Save choice in session
            method = st.radio(
                "Select balancing method",
                ["None", "Undersampling", "SMOTE"],
                index=["None", "Undersampling", "SMOTE"].index(
                    st.session_state.get("balance_method", "None")
                ),
            )
            st.session_state.balance_method = method

            # Explanation message
            if method == "None":
                st.info("No balancing will be applied. Models will use the original training split.")
            elif method == "Undersampling":
                st.warning("Undersampling will downsample the majority class in the **train** set.")
            else:
                st.warning("SMOTE will generate synthetic samples for the minority class in the **train** set.")

            st.divider()

            st.markdown("### ✅ Last balancing result")

            if "balance_status_msg" in st.session_state:
                last_msg = st.session_state.balance_status_msg
                last_counts = st.session_state.balance_status_counts

                if last_counts is not None:
                    c0 = last_counts.get(0, 0)
                    c1 = last_counts.get(1, 0)
                    st.success(last_msg)
                    st.caption(f"Balanced TRAIN set → 0: {c0:,} | 1: {c1:,}")
                else:
                    st.info(last_msg)
            else:
                st.info("Run a model first to apply balancing and see updated counts.")

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
                all_feats = [c for c in d_model.columns if c != "target"]

                # 🔍 Use final feature set from t-Tests & Stepwise tab (if exists)
                selected_feats = st.session_state.get("selected_features_for_modeling")

                if selected_feats:
                    selected_feats = [f for f in selected_feats if f in all_feats]
                if selected_feats:
                    used_feats = selected_feats
                    st.success(
                        f"Using {len(used_feats)} features selected by t-tests AND stepwise:\n\n"
                        + ", ".join(used_feats)
                    )
                else:
                    used_feats = all_feats
                    st.info(
                        f"No final feature set found yet. "
                        f"Using all {len(used_feats)} numeric features for modeling."
                    )

                X = d_model[used_feats]
                y = d_model["target"].astype(int)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=int(random_state),
                    stratify=y
                )
                st.markdown(f"**Number of features used for modeling:** {len(used_feats)}")

                # ⭐ NEW: apply class balancing option (TRAIN only)
                balance_method = st.session_state.get("balance_method", "None")
                X_train_model, y_train_model = X_train, y_train  # default: no balancing

                if balance_method == "Undersampling":
                    X_train_model, y_train_model = undersample_train(
                        X_train, y_train, random_state=int(random_state)
                    )
                    counts_bal = y_train_model.value_counts().sort_index()
                    msg = (
                        f"✔ Undersampling applied on TRAIN set. "
                        f"New distribution — 0: {counts_bal.get(0, 0):,}, "
                        f"1: {counts_bal.get(1, 0):,}"
                    )
                    st.success(msg)

                    # ✅ Save for Class Balancing tab
                    st.session_state.balance_status_msg = msg
                    st.session_state.balance_status_counts = dict(counts_bal)

                elif balance_method == "SMOTE":
                    sm = SMOTE(random_state=int(random_state))
                    X_train_model, y_train_model = sm.fit_resample(X_train, y_train)
                    counts_bal = pd.Series(y_train_model).value_counts().sort_index()
                    msg = (
                        f"✔ SMOTE applied on TRAIN set. "
                        f"New distribution — 0: {counts_bal.get(0, 0):,}, "
                        f"1: {counts_bal.get(1, 0):,}"
                    )
                    st.success(msg)

                    # ✅ Save for Class Balancing tab
                    st.session_state.balance_status_msg = msg
                    st.session_state.balance_status_counts = dict(counts_bal)
                else:
                    msg = "No class balancing applied (using original train split)."
                    st.info(msg)

                    st.session_state.balance_status_msg = msg
                    st.session_state.balance_status_counts = None

                # ----------------- A. Baseline Logit -----------------
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
                    # ⭐ use X_train_model / y_train_model
                    logit_pipe.fit(X_train_model, y_train_model)

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

                    # Confusion matrix — Logistic Regression
                    cm_logit = confusion_matrix(y_test, preds_logit)
                    cm_logit_df = pd.DataFrame(
                        cm_logit,
                        index=["True 0 (good)", "True 1 (bad)"],
                        columns=["Pred 0 (good)", "Pred 1 (bad)"],
                    )
                    st.markdown("**Confusion matrix — Logistic Regression (test set)**")
                    st.dataframe(cm_logit_df, use_container_width=True)

                    # 🔐 Save baseline metrics to session_state
                    st.session_state["baseline_metrics"] = {
                        "AUC": auc_logit,
                        "Accuracy": acc_logit,
                        "F1": f1_logit,
                        "Precision": precision_score(y_test, preds_logit, zero_division=0),
                        "Recall": recall_score(y_test, preds_logit, zero_division=0),
                        "Confusion Matrix": cm_logit,
                    }


                st.divider()

                # ----------------- B. Decision Tree -----------------
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
                    # ⭐ fit on balanced train
                    tree_clf.fit(X_train_model, y_train_model)

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

                    # Confusion matrix — Decision Tree
                    cm_tree = confusion_matrix(y_test, preds_tree)
                    cm_tree_df = pd.DataFrame(
                        cm_tree,
                        index=["True 0 (good)", "True 1 (bad)"],
                        columns=["Pred 0 (good)", "Pred 1 (bad)"],
                    )
                    st.markdown("**Confusion matrix — Decision Tree (test set)**")
                    st.dataframe(cm_tree_df, use_container_width=True)

                    # 🔐 Save tree metrics to session_state
                    st.session_state["tree_metrics"] = {
                        "AUC": auc_tree,
                        "Accuracy": acc_tree,
                        "F1": f1_tree,
                        "Precision": precision_score(y_test, preds_tree, zero_division=0),
                        "Recall": recall_score(y_test, preds_tree, zero_division=0),
                        "Confusion Matrix": cm_tree,
                    }

                st.divider()

                # ----------------- C. Hybrid Model -----------------
                st.markdown("### C. Hybrid Model — Tree-selected Logistic Regression")
                st.caption(
                    "Step 1: Use a Decision Tree to find the most important features. "
                    "Step 2: Train a Logistic Regression only on those features."
                )
                
                max_allowed_feats = min(20, X.shape[1])
                min_allowed_feats = min(3, max_allowed_feats)
                
                # ⚠️ Guard: when min == max, don't show a slider (it would crash)
                if max_allowed_feats <= min_allowed_feats:
                    max_feats_tree = max_allowed_feats
                    st.caption(
                        f"Only {max_allowed_feats} numeric features available; "
                        f"using all of them for the hybrid model."
                    )
                else:
                    max_feats_tree = st.slider(
                        "Maximum number of features selected from the tree",
                        min_value=min_allowed_feats,
                        max_value=max_allowed_feats,
                        value=min_allowed_feats,
                        key="tree_hybrid_max_feats",
                    )
                
                if st.button("Run Hybrid Model (Tree → Logit)", key="btn_tree_logit_hybrid"):
                    tree_clf2 = DecisionTreeClassifier(
                        max_depth=5,
                        min_samples_leaf=50,
                        random_state=int(random_state),
                    )
                    # ⭐ fit on balanced train
                    tree_clf2.fit(X_train_model, y_train_model)
                
                    feats_tree = tree_select_features(
                        X_train_model, y_train_model,
                        max_features=max_feats_tree,
                        random_state=int(random_state),
                    )
                    if not feats_tree:
                        st.warning("The decision tree did not select any informative features.")
                    else:
                        st.write(
                            f"**Features selected by the Decision Tree** "
                            f"({len(feats_tree)} features): " + ", ".join(feats_tree)
                        )
                        hybrid_pipe = Pipeline([
                            ("scaler", StandardScaler()),
                            ("logit", LogisticRegression(
                                C=1.0,
                                class_weight="balanced",
                                max_iter=1000,
                                solver="liblinear",
                            )),
                        ])
                        # ⭐ use balanced train subset
                        hybrid_pipe.fit(X_train_model[feats_tree], y_train_model)
                
                        probs_hybrid = hybrid_pipe.predict_proba(X_test[feats_tree])[:, 1]
                        preds_hybrid = (probs_hybrid >= 0.5).astype(int)
                        auc_hybrid = roc_auc_score(y_test, probs_hybrid)
                        acc_hybrid = accuracy_score(y_test, preds_hybrid)
                        f1_hybrid = f1_score(y_test, preds_hybrid)
                
                        st.write(
                            f"**Hybrid — Test AUC:** {auc_hybrid:.4f} · "
                            f"**Accuracy:** {acc_hybrid:.4f} · "
                            f"**F1-score (thr=0.5):** {f1_hybrid:.4f}"
                        )
                
                        # Confusion matrix — Hybrid Model
                        cm_hybrid = confusion_matrix(y_test, preds_hybrid)
                        cm_hybrid_df = pd.DataFrame(
                            cm_hybrid,
                            index=["True 0 (good)", "True 1 (bad)"],
                            columns=["Pred 0 (good)", "Pred 1 (bad)"],
                        )
                        st.markdown("**Confusion matrix — Hybrid Model (test set)**")
                        st.dataframe(cm_hybrid_df, use_container_width=True)
                
                        # 🔐 Save hybrid metrics to session_state
                        st.session_state["hybrid_metrics"] = {
                            "AUC": auc_hybrid,
                            "Accuracy": acc_hybrid,
                            "F1": f1_hybrid,
                            "Precision": precision_score(y_test, preds_hybrid, zero_division=0),
                            "Recall": recall_score(y_test, preds_hybrid, zero_division=0),
                            "Confusion Matrix": cm_hybrid,
                        }
                
                # 🔥 Final comparative analysis block (professor requirement)
                st.divider()
                st.subheader("📌 Final Comparative Analysis (All Models)")
                
                comp_rows_all = []
                
                bm = st.session_state.get("baseline_metrics")
                tm = st.session_state.get("tree_metrics")
                hm = st.session_state.get("hybrid_metrics")
                
                def unpack_cm_row(label, m):
                    """Build one row (dict) from metrics + confusion matrix."""
                    cm = m["Confusion Matrix"]          # 2x2 matrix: [[TN, FP], [FN, TP]]
                    tn, fp, fn, tp = cm.ravel().tolist()
                    return {
                        "Model": label,
                        "AUC": m["AUC"],
                        "Accuracy": m["Accuracy"],
                        "Precision": m["Precision"],
                        "Recall": m["Recall"],
                        "TN": tn,
                        "FP": fp,
                        "FN": fn,
                        "TP": tp,
                    }
                
                if bm is not None:
                    comp_rows_all.append(unpack_cm_row("Baseline Logistic Regression", bm))
                
                if tm is not None:
                    comp_rows_all.append(unpack_cm_row("Decision Tree", tm))
                
                if hm is not None:
                    comp_rows_all.append(unpack_cm_row("Hybrid (Tree → Logit)", hm))
                
                if comp_rows_all:
                    comp_df_all = pd.DataFrame(comp_rows_all)
                    st.dataframe(
                        comp_df_all.style.format(
                            {"AUC": "{:.4f}", "Accuracy": "{:.4f}",
                             "Precision": "{:.4f}", "Recall": "{:.4f}"}
                        ),
                        use_container_width=True,
                    )
                    st.success("Comparative analysis completed successfully.")
                else:
                    st.info("Run at least one model to see comparison metrics.")

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
