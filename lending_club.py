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
    from sklearn.cluster import KMeans

    # ✅ GAN library (only loaded when the app actually runs)
    from ctgan import CTGAN   # or CTGANSynthesizer depending on your version
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="altair")


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
        msg += f" Removed {removed_rows} completely empty rows and {removed_cols} completely empty variable."
    st.success(msg)

    # -------------------- 2. Analysis settings --------------------
    st.markdown("## 2. Analysis settings")

    c1, c2, c3 = st.columns([2, 1, 1])

    # 1) Target column
    with c1:
        target_col = st.selectbox(
            "Select target (label) variable",
            options=df_full.columns,
            help="If the selected variable is not binary (0/1), you will be able to map which values correspond to target=1 and target=0."
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

    # -------------------- Target mapping UI (center) --------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Configure binary target mapping")

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

    st.caption(
        "Select which values should be considered BAD (target=1) and GOOD (target=0). "
        "Values not selected in either group will be ignored in modeling."
    )

    bad_labels = st.multiselect(
        "BAD class values (target = 1):",
        options=sorted(labels),
        default=sorted(default_bad_labels),
        help="Choose values that represent default, churn, bad loans, etc."
    )

    good_labels = st.multiselect(
        "GOOD class values (target = 0):",
        options=sorted(labels),
        default=sorted(default_good_labels),
        help="Choose values that represent fully paid, retained customers, etc."
    )

    conflict = set(bad_labels) & set(good_labels)
    if conflict:
        st.error(f"The following values are assigned to both 1 and 0: {conflict}. Please fix this.")
        st.stop()

    bad_vals = [label_to_val[l] for l in bad_labels]
    good_vals = [label_to_val[l] for l in good_labels]

    df_full["target"] = pd.NA
    df_full.loc[target_raw.isin(bad_vals), "target"] = 1
    df_full.loc[target_raw.isin(good_vals), "target"] = 0
    df_full["target"] = pd.to_numeric(df_full["target"], errors="coerce")

    if df_full["target"].isna().all():
        st.error("No valid target mapping. Please assign at least one value to 1 or 0.")
        st.stop()
    if df_full["target"].nunique() < 2:
        st.error("Target has only one class after mapping. Please assign values to both 1 and 0.")
        st.stop()

    st.success(
        f"Mapped {len(bad_vals)} values to target=1 (BAD) and {len(good_vals)} values to target=0 (GOOD). "
        "Unselected values will be ignored."
    )

    st.markdown('</div>', unsafe_allow_html=True)


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
        "Maximum allowed missing rate per variable (%)",
        min_value=0,
        max_value=100,
        value=50,
        step=5,
        help="Variables with a higher percentage of missing values will be excluded from modeling.",
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

     # -------------------- Filters (under Analysis settings) --------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Filters")

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
        st.caption("No date-like variables detected.")

    # 3) Create issue_year using intelligent multi-format parsing
    if date_col_choice and date_col_choice != "(None)":
        raw_col = df_full[date_col_choice].astype(str).str.strip()

        def is_good_parse(parsed):
            """Check if parsed dates are usable: enough valid & reasonable years."""
            if parsed.notna().sum() == 0:
                return False
            valid_ratio_local = parsed.notna().mean()
            years = parsed.dt.year.dropna()
            if years.empty:
                return False
            median_year = years.median()
            return (valid_ratio_local >= 0.3) and (median_year >= 1950)

        # Attempt 1: generic parsing
        parsed_dates = pd.to_datetime(raw_col, errors="coerce")

        # Attempt 2: Lending Club style "Dec-17"
        if not is_good_parse(parsed_dates):
            parsed_dates = pd.to_datetime(raw_col, format="%b-%y", errors="coerce")

        # Attempt 3: "Dec-2017"
        if not is_good_parse(parsed_dates):
            parsed_dates = pd.to_datetime(raw_col, format="%b-%Y", errors="coerce")

        # Attempt 4: "16-mar" meaning 2016-Mar (yy-MMM)
        if not is_good_parse(parsed_dates):
            parsed_dates = pd.to_datetime(raw_col, format="%y-%b", errors="coerce")

        valid_ratio = parsed_dates.notna().mean()

        if is_good_parse(parsed_dates):
            df_full["issue_year"] = parsed_dates.dt.year
        else:
            st.warning(
                f"Column '{date_col_choice}' does not appear to be a valid date variable "
                f"({valid_ratio*100:.1f}% valid dates)."
            )

    # 4) Year filter (only appears if issue_year exists)
    year_range = None
    if "issue_year" in df_full.columns and df_full["issue_year"].notna().any():
        years = pd.to_numeric(df_full["issue_year"], errors="coerce").dropna().astype(int)
        if not years.empty:
            min_year, max_year = int(years.min()), int(years.max())
            if min_year == max_year:
                st.caption(
                    f"Only one year detected in the data: {min_year}. Year filter is disabled."
                )
                year_range = None
            else:
                year_range = st.slider(
                    "Filter by Issue Year",
                    min_value=min_year,
                    max_value=max_year,
                    value=(min_year, max_year),
                )
    else:
        st.caption("No 'issue_year' detected — line charts will not use year grouping.")

    # 5) Other filters (placeholders)
    grade_sel = None
    term_sel = None

    st.markdown('</div>', unsafe_allow_html=True)

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
            f'<div class="kpi"><div class="label">Variables</div>'
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
          You can configure which raw values map to 0 or 1 in the analysis settings; 
          any values not assigned to either class will be <b>ignored</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    # -------------------- EDA variables (fixed) --------------------
    df_eda = df.copy()

    # Use all numeric columns except the target for EDA charts
    EDA_VARS = (
        df_eda.select_dtypes(include=[np.number])
              .columns
              .drop(["target"], errors="ignore")
              .tolist()
    )

    if not EDA_VARS:
        st.warning("No numeric variables available for EDA charts.")

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
        """
        Build the modeling matrix used by:
        - ⚖️ Class Balancing
        - 📏 t-Tests & Stepwise
        - 🔮 Prediction Models
    
        It:
        1) Converts easy binary categoricals (yes/no, y/n, 1/0) to 0/1.
        2) Converts boolean columns to 0/1.
        3) Keeps only numeric predictors + 'target'.
        4) Applies missing-value handling.
        5) Keeps rows where target is 0/1.
        """
        if "target" not in df_in.columns:
            return pd.DataFrame()
    
        # ✅ Work on a copy so EDA/graphs are NOT affected
        df = df_in.copy()
    
        # ---- 1) Convert boolean columns to 0/1 ----
        bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
        for c in bool_cols:
            df[c] = df[c].astype(int)
    
        # ---- 2) Convert easy binary categoricals to 0/1 ----
        cat_cols = [
            c for c in df.select_dtypes(include=["object", "category"]).columns
            if c != "target"
        ]

        for c in cat_cols:
            if c == "target":
                continue
    
            vals = df[c].dropna().unique()
            if len(vals) == 0 or len(vals) > 2:
                continue  # not binary → ignore
    
            norm_vals = {str(v).strip().lower() for v in vals}
    
            if norm_vals.issubset({"yes", "no"}):
                df[c] = (
                    df[c].astype(str)
                    .str.strip().str.lower()
                    .map({"yes": 1, "no": 0})
                )
            elif norm_vals.issubset({"y", "n"}):
                df[c] = (
                    df[c].astype(str)
                    .str.strip().str.lower()
                    .map({"y": 1, "n": 0})
                )
            elif norm_vals.issubset({"1", "0"}):
                df[c] = (
                    df[c].astype(str)
                    .str.strip()
                    .map({"1": 1, "0": 0})
                )
            # else: leave as-is (will be ignored later)
    
        # ---- 3) Select numeric predictors ----
        num_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != "target"
        ]
        if not num_cols:
            return pd.DataFrame()
    
        miss_pct = df[num_cols].isna().mean() * 100
        keep_cols = miss_pct[miss_pct <= max_missing_pct].index.tolist()
        if not keep_cols:
            return pd.DataFrame()
    
        d = df[["target"] + keep_cols].copy()
    
        # ---- 4) Missing-value strategy ----
        if missing_strategy == "Impute with mean":
            for c in keep_cols:
                d[c] = d[c].fillna(d[c].mean())
        elif missing_strategy == "Impute with 0":
            d[keep_cols] = d[keep_cols].fillna(0)
        else:
            d = d.dropna(subset=keep_cols)
    
        # ---- 5) Clean target ----
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
        
    def backward_stepwise_select_features(X_train, y_train, X_val, y_val, min_features=1):
        """
        Backward stepwise feature selection based on validation AUC.
    
        1. Начинаем со всех доступных признаков.
        2. На каждом шаге пробуем удалить по одному признаку.
        3. Удаляем тот признак, при удалении которого AUC на валидации улучшается больше всего.
        4. Останавливаемся, когда удаление любого признака НЕ улучшает AUC (с учётом маленького допуска).
        """
    
        selected = list(X_train.columns)
        if len(selected) <= min_features:
            return selected
    
        base_model = LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            class_weight="balanced"
        )
    
        # AUC с полным набором признаков
        base_model.fit(X_train[selected], y_train)
        probs_full = base_model.predict_proba(X_val[selected])[:, 1]
        best_auc = roc_auc_score(y_val, probs_full)
    
        improved = True
        tol = 1e-4  # маленький шаг улучшения
    
        while improved and len(selected) > min_features:
            improved = False
            best_candidate_auc = best_auc
            feature_to_remove = None
    
            for f in selected:
                feats = [g for g in selected if g != f]
                if not feats:
                    continue
    
                model = LogisticRegression(
                    max_iter=1000,
                    solver="liblinear",
                    class_weight="balanced"
                )
                model.fit(X_train[feats], y_train)
                probs = model.predict_proba(X_val[feats])[:, 1]
                auc = roc_auc_score(y_val, probs)
    
                if auc > best_candidate_auc + tol:
                    best_candidate_auc = auc
                    feature_to_remove = f
    
            if feature_to_remove is not None:
                selected.remove(feature_to_remove)
                best_auc = best_candidate_auc
                improved = True
            else:
                break
    
        return selected
#forward stepwise
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

    def gan_oversample(
        X,
        y,
        random_state=42,
        minority_label=1,
        epochs=30,
        max_discrete_card=20,
        max_rows_for_gan=5000,
        max_synthetic=10000,
    ):
        """
        GAN-based oversampling using CTGAN, with defensive checks so that
        the app does NOT crash if CTGAN fails.
    
        Returns balanced X, y. If anything goes wrong, returns original X, y.
        """
        import numpy as np
        import pandas as pd
    
        # Ensure pandas objects from the start (so we can safely return them on error)
        X = pd.DataFrame(X).copy()
        y = pd.Series(y).copy()
    
        # Try importing CTGAN; if not installed, silently fall back
        try:
            from ctgan import CTGAN
        except ImportError as e:
            print(f"[gan_oversample] ctgan not installed: {e}")
            return X, y
    
        try:
            # ---------- 1. Checks on target ----------
            vc = y.value_counts()
            if vc.shape[0] < 2:
                # Only one class present
                return X, y
    
            maj_count = vc.max()
            min_count = vc.min()
    
            # Already roughly balanced → no GAN
            if maj_count <= min_count * 1.1:
                return X, y
    
            # ---------- 2. Build full df and choose minority ----------
            df_all = X.copy()
            df_all["target"] = y.values
    
            if minority_label in vc.index:
                min_cls = minority_label
            else:
                min_cls = vc.idxmin()
    
            df_minority = df_all[df_all["target"] == min_cls].reset_index(drop=True)
    
            maj_cls = vc.idxmax()
            maj_count = vc[maj_cls]
            current_min_count = df_minority.shape[0]
            n_to_generate = maj_count - current_min_count
            if n_to_generate <= 0:
                return X, y
    
            # Hard caps
            n_to_generate = int(min(n_to_generate, max_synthetic))
            if df_minority.shape[0] > max_rows_for_gan:
                df_minority = df_minority.sample(
                    max_rows_for_gan, random_state=random_state
                ).reset_index(drop=True)
    
            # ---------- 3. Detect discrete columns ----------
            discrete_cols = []
            for col in df_minority.columns:
                nunique = df_minority[col].nunique(dropna=True)
                if nunique <= max_discrete_card:
                    discrete_cols.append(col)
    
            # ---------- 4. Set seeds ----------
            np.random.seed(random_state)
            try:
                import torch
                torch.manual_seed(random_state)
            except ImportError:
                pass
    
            # ---------- 5. Train CTGAN ----------
            ctgan = CTGAN(
                epochs=epochs,
                verbose=False,
            )
            ctgan.fit(df_minority, discrete_cols)
    
            # ---------- 6. Sample synthetic data ----------
            synth_minority = ctgan.sample(n_to_generate)
            synth_minority["target"] = min_cls
    
            df_balanced = pd.concat([df_all, synth_minority], ignore_index=True)
            df_balanced = df_balanced.sample(
                frac=1.0, random_state=random_state
            ).reset_index(drop=True)
    
            # ---------- 7. Split back & SANITIZE for ML / Streamlit ----------
            X_bal = df_balanced.drop(columns=["target"])
            y_bal = df_balanced["target"].astype(int)
    
            # a) Forcibly convert all features to numeric (non-numeric → NaN)
            X_bal = X_bal.apply(pd.to_numeric, errors="coerce")
    
            # b) Fill NaN with column medians (or 0 if median is NaN)
            for col in X_bal.columns:
                med = X_bal[col].median()
                if pd.isna(med):
                    med = 0.0
                X_bal[col] = X_bal[col].fillna(med)
    
            return X_bal, y_bal
    
        except Exception as e:
            # If anything fails, print to server logs and fall back gracefully
            print(f"[gan_oversample] Error during CTGAN training/sampling: {e}")
            return X, y



    # -------------------- Navigation sidebar --------------------
    with st.sidebar:
    
        # ----- MORE AESTHETIC -----
        st.markdown("""
        <style>
        
        /* ---- Color palette: Pastel Blue ---- */
        :root {
            --card-bg: rgba(0,0,0,0.04);
            --card-bg-hover: rgba(0,0,0,0.08);
            --card-border: rgba(0,0,0,0.15);
            --card-selected: rgba(0,0,0,0.12);
            --card-selected-border: rgba(0,0,0,0.35);
        }
        
        /* ----- Title style ----- */
        .sidebar-title {
            font-size: 1.5rem !important;
            font-weight: 700 !important;
            margin-bottom: 10px !important;
        }
        
        /* ----- Subtitle ----- */
        .sidebar-subtitle {
            font-size: 1.15rem !important;
            font-weight: 500 !important;
            opacity: 0.9 !important;
            margin-bottom: 8px !important;
        }
        
        /* ----- Buttons / Cards ----- */
        div[role="radiogroup"] > label {
            background: var(--card-bg);
            padding: 12px 15px !important;
            margin-bottom: 10px !important;
            border: 1px solid var(--card-border);
            border-radius: 12px;
            font-size: 1.15rem !important;
            display: flex;
            align-items: center;
            gap: 8px;
            width: 100%;
            transition: 0.15s ease;
        }
        
        /* Hover */
        div[role="radiogroup"] > label:hover {
            background: var(--card-bg-hover);
            cursor: pointer;
        }
        
        /* Selected option */
        div[aria-checked="true"] {
            background: var(--card-selected) !important;
            border-color: var(--card-selected-border) !important;
        }
        
        </style>
        """, unsafe_allow_html=True)
        
            
        # ----- CONTENIDO -----
        st.markdown('<div class="sidebar-title">🧭 Navigation Menu</div>', unsafe_allow_html=True)
    
        page = st.radio(
            "Go to section",
            [
                "🧭 Data Exploration",
                "📈 Data Visualization",
                "⚖️ Class Balancing",
                "📏 t-Tests & Stepwise",
                "🔮 Prediction Models (Hybrid)",
                "👥 Customer Segmentation",   
            ],
            index=0,
        )


    # ========== Data Exploration ==========
    if page == "🧭 Data Exploration":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧭 Data Exploration — quick view")
        st.write("Sample of the dataframe used for visualizations (after filters).")
    
        SAMPLE_N = 10000
        sample = df if len(df) <= SAMPLE_N else df.sample(SAMPLE_N, random_state=42)
    
        rows_s, cols_s = sample.shape
        avg_missing = sample.isna().mean().mean() * 100
    
        # If target exists, also show Target = 1 Rate (%)
        if "target" in df.columns:
            target_rate = (df["target"].astype(float) == 1).mean() * 100
    
            c1_, c2_, c3_, c4_ = st.columns(4)
            with c1_:
                st.metric("Rows (sampled)", f"{rows_s:,}")
            with c2_:
                st.metric("Variables", f"{cols_s}")
            with c3_:
                st.metric("Avg missing (sample)", f"{avg_missing:.2f}%")
            with c4_:
                st.metric("Target = 1 Rate (%)", f"{target_rate:.2f}%")
        else:
            # Fallback: original three metrics
            c1_, c2_, c3_ = st.columns(3)
            with c1_:
                st.metric("Rows (sampled)", f"{rows_s:,}")
            with c2_:
                st.metric("Variables", f"{cols_s}")
            with c3_:
                st.metric("Avg missing (sample)", f"{avg_missing:.2f}%")
    
        # IDs de variables v1, v2, v3...
        var_ids = {col: f"v{i+1}" for i, col in enumerate(df.columns)}
        
        # ---------- Column data types ----------
        st.markdown("#### Variable data types")
        dtypes_df = (
            df.dtypes.reset_index().rename(columns={"index": "Variable Name", 0: "dtype"})
        )
        dtypes_df["dtype"] = dtypes_df["dtype"].astype(str)
        dtypes_df.insert(0, "v", dtypes_df["Variable Name"].map(var_ids))
        dtypes_df = dtypes_df.reset_index(drop=True)

        try:
            st.dataframe(dtypes_df, use_container_width=True, hide_index=True)
        except TypeError:
            # por si tu versión de Streamlit no tiene hide_index
            st.dataframe(dtypes_df, use_container_width=True)

        # ---------- Missing values per column ----------
        st.markdown("#### Missing values per Variable")
        missing_count = df.isna().sum()
        missing_pct_col = df.isna().mean() * 100

        missing_df = pd.DataFrame({
            "Variable Name": df.columns,
            "missing_count": missing_count.values,
            "missing_pct (%)": missing_pct_col.values,
        })
        missing_df["missing_pct (%)"] = missing_df["missing_pct (%)"].round(2)
        missing_df.insert(0, "v", missing_df["Variable Name"].map(var_ids))
        missing_df = missing_df.sort_values("missing_count", ascending=False)
        missing_df = missing_df.reset_index(drop=True)

        try:
            st.dataframe(missing_df, use_container_width=True, hide_index=True)
        except TypeError:
            st.dataframe(missing_df, use_container_width=True)

        # ---------- Head ----------
        st.markdown("#### Head (first non-empty rows)")
        head_df = sample.dropna(how="all").head(10)
        head_df = head_df.reset_index(drop=True)
        try:
            st.dataframe(head_df, use_container_width=True, hide_index=True)
        except TypeError:
            st.dataframe(head_df, use_container_width=True)

        # ---------- Statistical summary (`describe`) ----------
        st.markdown("#### Statistical summary (`describe`)")

        # Numeric summary
        num_sample = sample.select_dtypes(include=["number"])
        if not num_sample.empty:
            st.markdown("##### Numeric summary")
            desc_num = num_sample.describe().T.round(3)

            # añadir v y column
            desc_num.insert(0, "v", desc_num.index.map(var_ids))
            desc_num.insert(1, "Variable name", desc_num.index)
            desc_num = desc_num.reset_index(drop=True)

            try:
                st.dataframe(desc_num, use_container_width=True, hide_index=True)
            except TypeError:
                st.dataframe(desc_num, use_container_width=True)

        # Categorical summary
        cat_sample = sample.select_dtypes(exclude=["number"])
        if not cat_sample.empty:
            st.markdown("##### Categorical summary")
            desc_cat = cat_sample.describe().T
        
            desc_cat.insert(0, "v", desc_cat.index.map(var_ids))
            desc_cat.insert(1, "Variable name", desc_cat.index)
            desc_cat = desc_cat.reset_index(drop=True)
        
            # ✅ CRITICAL FIX: force everything to string for Arrow safety
            desc_cat = desc_cat.astype(str)
        
            try:
                st.dataframe(desc_cat, use_container_width=True, hide_index=True)
            except TypeError:
                st.dataframe(desc_cat, use_container_width=True)


        st.markdown('</div>', unsafe_allow_html=True)


    # ========== Distributions ==========
    elif page == "📈 Data Visualization":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📈 Data Visualization — Histograms, Boxplots & Line")
        st.caption("🎯 Target legend: 0 = good, 1 = bad")

        if not EDA_VARS:
            st.info("No suitable numeric variables from the requested list.")
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
                # We require a date column selected in the analysis settings
                if not (date_col_choice and date_col_choice != "(None)" and date_col_choice in df.columns):
                    st.info("Please select a valid date variable in the analysis settings to enable the line chart.")
                else:
                    y_var = st.selectbox(
                        "Y variable",
                        options=EDA_VARS,
                        index=(EDA_VARS.index("loan_amnt") if "loan_amnt" in EDA_VARS else 0),
                        key="line_y_var"
                    )
    
                    agg_choice = st.selectbox(
                        "Aggregation",
                        ["Mean", "Median", "Count"],
                        index=0,
                        key="line_agg"
                    )
    
                    # New control: time granularity
                    granularity = st.radio(
                        "Time granularity for line chart",
                        ["Year", "Month", "Year-Month"],
                        index=0,
                        horizontal=True,
                        key="line_time_granularity"
                    )
    
                    # Build working dataframe
                    cols_base = [y_var]
                    if "target" in df.columns:
                        cols_base.append("target")
    
                    df_line = df[cols_base].copy()
        
                    # Parse the selected date column again (multi-format, similar to sidebar)
                    def parse_dates_for_chart(series):
                        """Try multiple formats to parse dates for the line chart."""
                        s = series.astype(str).str.strip()
    
                        parsed = pd.to_datetime(s, errors="coerce")
                        if parsed.notna().mean() < 0.3:
                            # Format like "Dec-17"
                            parsed = pd.to_datetime(s, format="%b-%y", errors="coerce")
                        if parsed.notna().mean() < 0.3:
                            # Format like "Dec-2017"
                            parsed = pd.to_datetime(s, format="%b-%Y", errors="coerce")
                        if parsed.notna().mean() < 0.3:
                            # Format like "16-mar" (= 2016-Mar)
                            parsed = pd.to_datetime(s, format="%y-%b", errors="coerce")
    
                        return parsed
    
                    dt = parse_dates_for_chart(df[date_col_choice])
    
                    df_line["__year__"] = dt.dt.year
                    df_line["__month__"] = dt.dt.month
                    df_line["__year_month__"] = dt.dt.to_period("M").astype(str)
    
                    # Drop rows with invalid dates
                    df_line = df_line.dropna(subset=["__year__"])

    
                    # Choose the time key based on granularity
                    if granularity == "Year":
                        time_key = "__year__"
                        x_title = "Year"
                    elif granularity == "Month":
                        time_key = "__month__"
                        x_title = "Month (1–12)"
                    else:  # "Year-Month"
                        time_key = "__year_month__"
                        x_title = "Year-Month"
    
                    if df_line.empty:
                        st.info("No valid dates available to draw the line chart.")
                    else:
                        # Aggregation
                        if agg_choice == "Mean":
                            agg_func, y_enc_title = "mean", f"Mean {y_var}"
                        elif agg_choice == "Median":
                            agg_func, y_enc_title = "median", f"Median {y_var}"
                        else:
                            agg_func, y_enc_title = "count", f"Count ({y_var} non-null)"
    
                        # Group by time (and target if available)
                        if "target" in df_line.columns:
                            g = df_line.groupby([time_key, "target"], observed=False)
                        else:
                            g = df_line.groupby([time_key], observed=False)
    
                        if agg_choice == "Count":
                            plot_df = g[y_var].count().reset_index(name="y")
                        else:
                            plot_df = g[y_var].agg(agg_func).reset_index(name="y")
    
                        # Color encoding if target exists
                        enc_color = (
                            alt.Color("target:N", title="target")
                            if "target" in plot_df.columns
                            else alt.value(None)
                        )
    
                        line = (
                            alt.Chart(plot_df)
                            .mark_line(point=True)
                            .encode(
                                x=alt.X(f"{time_key}:O", title=x_title),
                                y=alt.Y("y:Q", title=y_enc_title),
                                color=enc_color,
                                tooltip=[alt.Tooltip(f"{time_key}:O", title="Time"),
                                         alt.Tooltip("y:Q", title=y_enc_title, format=".2f")]
                                        + ([alt.Tooltip("target:N", title="target")] if "target" in plot_df.columns else [])
                            )
                            .properties(
                                height=300,
                                title=f"{agg_choice} {y_var} by {granularity}"
                            )
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
    elif page == "📏 t-Tests & Stepwise":
        st.markdown('<div class="card">', unsafe_allow_html=True)

        # Título y descripción del panel
        st.markdown("### 📏 t-Tests & Stepwise")
        st.caption("F-test + t-tests (student method) and forward stepwise logistic regression.")

        if "target" not in df.columns:
            st.info("No 'target' column found.")
        else:
            # 1) Matriz de modelado base (igual que para modelos)
            d_model_sw = build_model_matrix(df)
            if d_model_sw.empty:
                st.info("Not enough numeric features or valid 0/1 target to run t-tests / stepwise.")
                st.markdown('</div>', unsafe_allow_html=True)
                st.stop()

        #    2) Choose dataset for t-tests & stepwise:
        #    if a globally balanced dataset exists (from ⚖️ Class Balancing),
        #    use it. Otherwise, use the original modeling matrix.
            if "X_balanced" in st.session_state and "y_balanced" in st.session_state:
                d_for_tests = pd.concat(
                    [
                        st.session_state["X_balanced"],
                        st.session_state["y_balanced"].rename("target")
                    ],
                    axis=1
                )
                st.caption("Using globally balanced dataset from ⚖️ Class Balancing for t-tests & stepwise.")
            else:
                d_for_tests = d_model_sw.copy()
                st.caption("Using original (unbalanced) dataset for t-tests & stepwise.")
    


            # 3) t-tests sobre el dataset (posiblemente balanceado)
            tnum = d_for_tests["target"].astype(int)
            mask_valid = tnum.isin([0, 1])
            if mask_valid.sum() < 2 or tnum[mask_valid].nunique() < 2:
                st.info("Both target groups (0 and 1) must be present to run t-tests.")
            else:
                # Todas las variables numéricas excepto target
                VARS = [
                    c for c in d_for_tests.columns
                    if c != "target" and pd.api.types.is_numeric_dtype(d_for_tests[c])
                ]

                if not VARS:
                    st.info("No numeric variables available for t-tests.")
                else:
                    rows_t = []

                    for col in VARS:
                        s = pd.to_numeric(d_for_tests[col], errors="coerce")
                        d_loc = pd.DataFrame({"y": s, "t": tnum}).dropna()

                        g0 = d_loc.loc[d_loc["t"] == 0, "y"].values  # target = 0
                        g1 = d_loc.loc[d_loc["t"] == 1, "y"].values  # target = 1

                        if len(g0) < 2 or len(g1) < 2:
                            continue

                        m0, m1 = np.mean(g0), np.mean(g1)
                        s0, s1 = np.std(g0, ddof=1), np.std(g1, ddof=1)

                        # F-test (Levene) para igualdad de varianzas
                        f_stat, f_p = stats.levene(g0, g1)

                        # t-tests con y sin varianzas iguales
                        t_eq, p_eq = stats.ttest_ind(g1, g0, equal_var=True)
                        t_uneq, p_uneq = stats.ttest_ind(g1, g0, equal_var=False)

                        # Elegir el t-test según F-test
                        if f_p >= 0.05:
                            used_test = "equal_var"
                            used_t = t_eq
                            used_p = p_eq
                        else:
                            used_test = "unequal_var"
                            used_t = t_uneq
                            used_p = p_uneq

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
                            "used_t": used_t,
                            "used_p_value": used_p,
                        })

                    if not rows_t:
                        st.info("No valid data to compute t-tests.")
                    else:
                        res = pd.DataFrame(rows_t)

                        # Ordenar por p-value usado (más significativas arriba)
                        res_sorted = res.sort_values("used_p_value").reset_index(drop=True)

                        # IDs consistentes con Data Exploration: v1, v2, ...
                        var_ids = {col: f"v{i+1}" for i, col in enumerate(df.columns)}

                        # Tabla resumida que se muestra
                        df_display = res_sorted[["variable", "used_t", "used_p_value"]].copy()
                        df_display.insert(0, "variable_id", df_display["variable"].map(var_ids))
                        df_display.rename(
                            columns={
                                "variable": "variable_name",
                                "used_t": "t_value",
                                "used_p_value": "p_value",
                            },
                            inplace=True,
                        )

                        # p_value numérico crudo
                        p_raw = df_display["p_value"].astype(float)

                        # Columnas de alphas con x / xx / xxx
                        df_display["alpha_0.10"] = np.where(p_raw < 0.10, "x", "")
                        df_display["alpha_0.05"] = np.where(p_raw < 0.05, "xx", "")
                        df_display["alpha_0.01"] = np.where(p_raw < 0.01, "xxx", "")

                        # Mostrar p_value con 3 decimales
                        df_display["p_value"] = p_raw.round(3)

                        st.dataframe(df_display, use_container_width=True)
                        st.caption(
                            "Variable IDs (v1, v2, ...) match the Data Exploration tab. "
                            "x / xx / xxx indicate significance at α = 0.10 / 0.05 / 0.01 respectively."
                        )

                        # Definir selected_features SOLO desde la tabla (p < 0.05)
                        mask_005 = p_raw < 0.05
                        selected_features = df_display.loc[mask_005, "variable_name"].tolist()
                        st.session_state["ttest_sig_features"] = selected_features

                        num_sig = mask_005.sum()
                        if num_sig > 0:
                            st.success(f"{num_sig} variables selected by t-tests (p < 0.05).")
                            st.caption(", ".join(selected_features))
                        else:
                            st.warning(
                                "No variables are significant at α = 0.05. "
                                "Stepwise will fall back to all numeric predictors."
                            )

                        st.markdown("---")
                        st.subheader("Stepwise logistic regression (based on t-test results)")
                        
                        # 4) Stepwise using the same dataset (possibly balanced)
                        if d_for_tests.empty:
                            st.info("Not enough numeric features or valid 0/1 target to run stepwise selection.")
                        else:
                            # Pool of candidates: those that passed t-test, or all if none
                            if selected_features:
                                candidate_feats = [
                                    c for c in selected_features
                                    if c in d_for_tests.columns and c != "target"
                                ]
                                st.caption(
                                    f"Stepwise candidate pool: {len(candidate_feats)} "
                                    f"variables that passed the t-tests."
                                )
                            else:
                                candidate_feats = [
                                    c for c in d_for_tests.columns if c != "target"
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
                                X_sw_all = d_for_tests[candidate_feats]
                                y_sw_all = d_for_tests["target"].astype(int)
                        
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
                        
                                max_feats_sw = X_sw_all.shape[1]
                        
                                st.caption(
                                    f"Stepwise candidate pool: {max_feats_sw} variables. "
                                    "The algorithm will automatically decide how many features to keep."
                                )
                        
                                # 🔽 NEW: choice between Forward / Backward
                                stepwise_direction = st.radio(
                                    "Stepwise direction",
                                    options=["Forward", "Backward"],
                                    index=0,
                                    horizontal=True,
                                    key="stepwise_direction",
                                    help="Forward = add features one by one. Backward = start with all features and remove weak ones."
                                )
                        
                                if st.button("Run stepwise selection", key="btn_stepwise_ttest_tab"):
                        
                                    direction = st.session_state["stepwise_direction"]
                        
                                    if direction == "Forward":
                                        feats_sw = stepwise_select_features(
                                            X_train_sw, y_train_sw,
                                            X_val_sw, y_val_sw,
                                            max_features=max_feats_sw,
                                        )
                                    else:  # Backward
                                        min_feats_sw = min(3, X_sw_all.shape[1])  # keep at least 3 features
                                        feats_sw = backward_stepwise_select_features(
                                            X_train_sw, y_train_sw,
                                            X_val_sw, y_val_sw,
                                            min_features=min_feats_sw,
                                        )
                        
                                    # Handle results
                                    if not feats_sw:
                                        st.warning("Stepwise did not find any feature set that improves AUC.")
                                    else:
                                        st.success(
                                            f"{direction} stepwise selected {len(feats_sw)} features:\n" +
                                            ", ".join(feats_sw)
                                        )
                        
                                        st.session_state["stepwise_features"] = feats_sw
                                        st.session_state["selected_features_for_modeling"] = feats_sw
                                        st.caption("Final feature set saved for the Prediction Models tab.")


        st.markdown('</div>', unsafe_allow_html=True)


    # ========== Class Balancing ==========
    elif page == "⚖️ Class Balancing":
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

             # Summary cards for class counts and positive rate
            total = counts.sum()
            target_rate = (counts.get(1, 0) / total * 100) if total > 0 else 0.0

            c1b, c2b, c3b = st.columns(3)
            with c1b:
                st.metric("Class 0 count", f"{counts.get(0, 0):,}")
            with c2b:
                st.metric("Class 1 count", f"{counts.get(1, 0):,}")
            with c3b:
                st.metric("Target = 1 Rate (%)", f"{target_rate:.2f}%")


            st.markdown("---")

            # Save choice in session
            # Available balancing options (including GAN)
            balance_options = ["None", "Undersampling", "SMOTE", "GAN"]

            # Get the currently selected method from session state (default = "None")
            current_method = st.session_state.get("balance_method", "None")
            if current_method not in balance_options:
                current_method = "None"

            method = st.radio(
                "Select balancing method",
                balance_options,
                index=balance_options.index(current_method),
            )
            st.session_state.balance_method = method

            if st.button("🚀 Apply Class Balancing"):
                # Build base modeling matrix
                d_model_bal = build_model_matrix(df)
            
                if d_model_bal.empty:
                    st.warning("Not enough valid numeric features or target to apply class balancing.")
                    st.stop()
            
                X_base = d_model_bal.drop(columns=["target"])
                y_base = d_model_bal["target"].astype(int)
            
                if method == "Undersampling":
                    Xb, yb = undersample_train(X_base, y_base, random_state=int(random_state))
            
                elif method == "SMOTE":
                    sm = SMOTE(random_state=int(random_state))
                    Xb, yb = sm.fit_resample(X_base, y_base)
            
                elif method == "GAN":
                    Xb, yb = gan_oversample(
                        X_base, y_base, random_state=int(random_state)
                    )
            
                else:
                    Xb, yb = X_base.copy(), y_base.copy()
            
                # ✅ Store globally in session_state
                st.session_state["X_balanced"] = Xb
                st.session_state["y_balanced"] = yb
            
                counts_bal = yb.value_counts().sort_index()
                msg = (
                    f"✔ {method} applied globally. "
                    f"0: {counts_bal.get(0,0):,} | 1: {counts_bal.get(1,0):,}"
                )
            
                st.session_state.balance_status_msg = msg
                st.session_state.balance_status_counts = dict(counts_bal)
            
                st.success(msg)


            # Explanation message for each balancing option
            if method == "None":
                st.info("No balancing will be applied. Models will use the original training split.")
            elif method == "Undersampling":
                st.warning("Undersampling will downsample the majority class in the **train** set.")
            elif method == "SMOTE":
                st.warning("SMOTE will generate synthetic samples for the minority class in the **train** set.")
            elif method == "GAN":
                st.warning(
                    "GAN-based oversampling will generate synthetic samples for the minority class "
                    "in the **train** set (experimental method, currently a placeholder)."
                )


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
        
    # ==================== CUSTOMER SEGMENTATION PAGE ====================
    elif page == "👥 Customer Segmentation":
        st.subheader("👥 Customer Segmentation")
    
        if df.empty:
            st.info("Please upload a dataset first.")
            st.stop()
    
        tab1, tab2 = st.tabs(["📊 General segmentation", "💰 Risk × Value Segmentation"])
    
        # ----------------------------------------------------
        # TAB 1 — General segmentation (Before prediction)
        # ----------------------------------------------------
        with tab1:
            st.markdown("### 📊 General Segmentation (Demographic / Categorical)")
            st.caption("Use this to explore churn patterns across customer groups (gender, contract type, region, etc.).")
    
            # Select categorical variables only
            cat_cols = [
                c for c in df.columns
                if df[c].dtype == "object" or df[c].nunique() <= 20
            ]
    
            if not cat_cols:
                st.info("No suitable categorical variables found for segmentation.")
            else:
                seg_var = st.selectbox("Select segmentation variable:", cat_cols)
    
                if "target" not in df.columns:
                    st.warning("A binary 'target' column is required to calculate churn rates.")
                else:
                    temp = df.copy()
                    temp["target_num"] = pd.to_numeric(temp["target"], errors="coerce")
    
                    summary = (
                        temp.groupby(seg_var)
                        .agg(
                            n_customers=("target_num", "size"),
                            churn_rate=("target_num", "mean"),
                        )
                        .reset_index()
                    )
                    summary["churn_rate (%)"] = (summary["churn_rate"] * 100).round(2)
    
                    st.dataframe(summary, use_container_width=True)
    
                    # Visualization
                    chart = (
                        alt.Chart(summary)
                        .mark_bar()
                        .encode(
                            x=seg_var + ":N",
                            y="churn_rate (%):Q",
                            tooltip=[seg_var, "n_customers", "churn_rate (%)"],
                        )
                        .properties(height=350)
                    )
                    st.altair_chart(chart, use_container_width=True)
    
        # ----------------------------------------------------
        # TAB 2 — Risk × Value segmentation (After prediction)
        # ----------------------------------------------------
        with tab2:
            st.markdown("### 💰 Risk × Value Segmentation")
            st.caption(
                "This requires a **predicted churn probability** column from a model.\n"
                "Once the model runs, we will automatically store the predictions."
            )
    
            if "segmentation_df" not in st.session_state:
                st.info("Run a prediction model to activate this tab.")
                st.stop()
    
            seg_df = st.session_state["segmentation_df"].copy()
    
            if "churn_proba" not in seg_df.columns:
                st.warning("No churn probability found. Please re-run prediction.")
                st.stop()
    
            # Choose value variable
            num_cols = [
                c for c in seg_df.columns
                if seg_df[c].dtype != "object" and c not in ["target", "churn_proba"]
            ]
            if not num_cols:
                st.info("No numeric value columns available.")
                st.stop()
    
            value_col = st.selectbox("Select customer value metric:", num_cols)
    
            # Split into High / Low value (median)
            median_val = seg_df[value_col].median()
            seg_df["Value Segment"] = seg_df[value_col].apply(
                lambda x: "High Value" if x >= median_val else "Low Value"
            )
    
            # Define risk buckets
            low_thr, high_thr = st.slider(
                "Risk thresholds (based on churn probability)",
                0.0, 1.0, (0.3, 0.7), 0.05
            )
    
            def risk_bucket(p):
                if p < low_thr:
                    return "Low Risk"
                elif p < high_thr:
                    return "Medium Risk"
                else:
                    return "High Risk"
    
            seg_df["Risk Segment"] = seg_df["churn_proba"].apply(risk_bucket)
    
            # Build Risk x Value Matrix
            matrix = (
                seg_df.groupby(["Risk Segment", "Value Segment"])
                .agg(
                    customers=("target", "size"),
                    avg_value=(value_col, "mean"),
                    avg_risk=("churn_proba", "mean")
                )
                .reset_index()
            )
    
            matrix["avg_risk (%)"] = (matrix["avg_risk"] * 100).round(1)
            matrix["avg_value"] = matrix["avg_value"].round(2)
    
            st.dataframe(matrix, use_container_width=True)
    
            st.markdown("""
            ### Interpretation
            - 🔴 **High Risk + High Value** → Priority retention group  
            - 🟠 **High Risk + Low Value** → Retain if cost-efficient  
            - 🟡 **Medium Risk + High Value** → Monitor, cross-sell  
            - 🟢 **Low Risk + High Value** → Loyalty programs  
            - ⚪ **Low Risk + Low Value** → Low priority  
            """)


    # ========== Prediction Models ==========

    elif page == "🔮 Prediction Models (Hybrid)":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔮 Prediction Models — Logistic Regression, Decision Tree & Hybrid")
        st.caption("Target legend — 0: good outcome, 1: bad outcome (as defined in the analysis settings).")
    
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
    
                # Overall target = 1 rate on the modeling dataset
                target_rate_model = (y == 1).mean() * 100
    
                # Small metrics row for modeling context
                c_mod1, c_mod2 = st.columns(2)
                with c_mod1:
                    st.metric("Number of features used for modeling", len(used_feats))
                with c_mod2:
                    st.metric("Target = 1 Rate (%)", f"{target_rate_model:.2f}%")
    
                # ⭐ Use balanced dataset IF it already exists (applied in ⚖️ Class Balancing)
                # ⭐ Use balanced dataset IF it already exists (applied in ⚖️ Class Balancing)
                if "X_balanced" in st.session_state and "y_balanced" in st.session_state:
                    X_bal = st.session_state["X_balanced"].copy()
                    y_bal = st.session_state["y_balanced"].copy()
                
                    # Ensure we use the SAME features (and order) as in used_feats
                    missing = [f for f in used_feats if f not in X_bal.columns]
                
                    if missing:
                        st.warning(
                            "Balanced dataset is missing some selected features: "
                            + ", ".join(missing)
                            + ". Falling back to original (unbalanced) TRAIN split."
                        )
                        X_train_model, y_train_model = X_train, y_train
                    else:
                        # Align columns to used_feats → same names & order as X_test
                        X_train_model = X_bal[used_feats].copy()
                        y_train_model = y_bal.copy()
                        st.success("Using globally balanced dataset from ⚖️ Class Balancing (columns aligned).")
                else:
                    X_train_model, y_train_model = X_train, y_train
                    st.info("Using original (unbalanced) TRAIN split.")
                


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
                    seg_df = df.loc[X_test.index].copy()
                    seg_df["churn_proba"] = probs_logit
                    seg_df["predicted_label"] = preds_logit
                    st.session_state["segmentation_df"] = seg_df
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
                        "F1": m["F1"],
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
                            "F1" : "{:.4f}",
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
