import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="InsightGen",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0b0e14;
    color: #e0e6f0;
}

.stApp { background-color: #0b0e14; }

h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #7ee8fa 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    margin-bottom: 0.2rem;
}

.subtitle {
    color: #6b7a99;
    font-size: 0.85rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

.metric-card {
    background: #131720;
    border: 1px solid #1e2535;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
}

.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #7ee8fa;
    line-height: 1;
}

.metric-label {
    font-size: 0.72rem;
    color: #6b7a99;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 0.3rem;
}

.insight-card {
    background: #131720;
    border-left: 3px solid #a78bfa;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    font-size: 0.84rem;
    color: #c4cde0;
    line-height: 1.5;
}

.insight-card.warning {
    border-left-color: #f59e0b;
}

.insight-card.positive {
    border-left-color: #34d399;
}

.insight-card.info {
    border-left-color: #60a5fa;
}

.section-tag {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: #a78bfa;
    font-weight: 500;
    margin-bottom: 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e2535;
}

.quality-bar-bg {
    background: #1e2535;
    border-radius: 4px;
    height: 6px;
    margin-top: 6px;
}

.stSelectbox > div > div {
    background: #131720 !important;
    border: 1px solid #1e2535 !important;
    color: #e0e6f0 !important;
    border-radius: 8px !important;
}

.stMultiSelect > div > div {
    background: #131720 !important;
    border: 1px solid #1e2535 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #7ee8fa, #a78bfa) !important;
    color: #0b0e14 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    padding: 0.5rem 1.5rem !important;
}

.stFileUploader {
    background: #131720 !important;
    border: 1.5px dashed #1e2535 !important;
    border-radius: 12px !important;
}

div[data-testid="stExpander"] {
    background: #131720 !important;
    border: 1px solid #1e2535 !important;
    border-radius: 10px !important;
}

.stDataFrame { background: #131720 !important; }

.stTabs [data-baseweb="tab-list"] {
    background: #131720;
    border-radius: 10px;
    border: 1px solid #1e2535;
    gap: 4px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #6b7a99;
    border-radius: 7px;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
}
.stTabs [aria-selected="true"] {
    background: #1e2535 !important;
    color: #7ee8fa !important;
}

hr { border-color: #1e2535; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA QUALITY ENGINE
# ─────────────────────────────────────────────
class DataQualityEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.original_shape = df.shape
        self.issues: List[Dict] = []
        self.cleaned_df = df.copy()

    def run(self) -> Tuple[pd.DataFrame, List[Dict], float]:
        self._infer_and_cast_types()
        self._handle_missing_values()
        self._detect_and_cap_outliers()
        self._drop_duplicates()
        score = self._compute_quality_score()
        return self.cleaned_df, self.issues, score

    def _infer_and_cast_types(self):
        """Auto-infer and cast column types"""
        for col in self.cleaned_df.columns:
            if self.cleaned_df[col].dtype == object:
                # Try numeric
                converted = pd.to_numeric(self.cleaned_df[col], errors='coerce')
                if converted.notna().sum() / len(self.cleaned_df) > 0.8:
                    self.cleaned_df[col] = converted
                    self.issues.append({
                        'type': 'type_cast', 'col': col,
                        'msg': f'"{col}" auto-converted to numeric (was string)',
                        'severity': 'info'
                    })
                    continue
                # Try datetime
                try:
                    converted_dt = pd.to_datetime(self.cleaned_df[col], errors='coerce')
                    if converted_dt.notna().sum() / len(self.cleaned_df) > 0.7:
                        self.cleaned_df[col] = converted_dt
                        self.issues.append({
                            'type': 'type_cast', 'col': col,
                            'msg': f'"{col}" auto-detected as datetime',
                            'severity': 'info'
                        })
                except Exception:
                    pass

    def _handle_missing_values(self):
        """Impute missing values intelligently"""
        for col in self.cleaned_df.columns:
            missing = self.cleaned_df[col].isnull().sum()
            if missing == 0:
                continue
            pct = missing / len(self.cleaned_df) * 100
            if pct > 60:
                self.cleaned_df.drop(columns=[col], inplace=True)
                self.issues.append({
                    'type': 'dropped_col', 'col': col,
                    'msg': f'"{col}" dropped — {pct:.0f}% missing values',
                    'severity': 'warning'
                })
            elif pd.api.types.is_numeric_dtype(self.cleaned_df[col]):
                median_val = self.cleaned_df[col].median()
                self.cleaned_df[col].fillna(median_val, inplace=True)
                self.issues.append({
                    'type': 'imputed', 'col': col,
                    'msg': f'"{col}": {missing} missing values imputed with median ({median_val:.2f})',
                    'severity': 'warning'
                })
            else:
                mode_val = self.cleaned_df[col].mode()
                if len(mode_val) > 0:
                    self.cleaned_df[col].fillna(mode_val[0], inplace=True)
                    self.issues.append({
                        'type': 'imputed', 'col': col,
                        'msg': f'"{col}": {missing} missing values filled with mode ("{mode_val[0]}")',
                        'severity': 'warning'
                    })

    def _detect_and_cap_outliers(self):
        """IQR-based outlier capping"""
        num_cols = self.cleaned_df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            Q1 = self.cleaned_df[col].quantile(0.25)
            Q3 = self.cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 3 * IQR, Q3 + 3 * IQR
            outliers = ((self.cleaned_df[col] < lower) | (self.cleaned_df[col] > upper)).sum()
            if outliers > 0:
                self.cleaned_df[col] = self.cleaned_df[col].clip(lower, upper)
                self.issues.append({
                    'type': 'outlier', 'col': col,
                    'msg': f'"{col}": {outliers} outliers capped at [{lower:.2f}, {upper:.2f}]',
                    'severity': 'warning'
                })

    def _drop_duplicates(self):
        n_before = len(self.cleaned_df)
        self.cleaned_df.drop_duplicates(inplace=True)
        dropped = n_before - len(self.cleaned_df)
        if dropped > 0:
            self.issues.append({
                'type': 'duplicates', 'col': 'all',
                'msg': f'{dropped} duplicate rows removed',
                'severity': 'warning'
            })

    def _compute_quality_score(self) -> float:
        total_cells = self.original_shape[0] * self.original_shape[1]
        missing_cells = self.df.isnull().sum().sum()
        missing_score = max(0, 1 - missing_cells / total_cells)
        dup_score = max(0, 1 - self.df.duplicated().sum() / len(self.df))
        type_issues = sum(1 for i in self.issues if i['type'] == 'type_cast')
        type_score = max(0, 1 - type_issues / max(len(self.df.columns), 1))
        return round((missing_score * 0.5 + dup_score * 0.3 + type_score * 0.2) * 100, 1)


# ─────────────────────────────────────────────
# AUTO-INSIGHT ENGINE
# ─────────────────────────────────────────────
class AutoInsightEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.num_cols = list(df.select_dtypes(include=[np.number]).columns)
        self.cat_cols = list(df.select_dtypes(include=['object', 'category']).columns)
        self.dt_cols = list(df.select_dtypes(include=['datetime64']).columns)
        self.insights: List[Dict] = []
        self.auto_charts: List[go.Figure] = []

    def run(self):
        self._detect_distributions()
        self._detect_correlations()
        self._detect_trends()
        self._detect_group_disparities()
        self._detect_skewness()
        self._auto_select_charts()
        return self.insights, self.auto_charts

    def _detect_distributions(self):
        for col in self.num_cols:
            series = self.df[col].dropna()
            if len(series) < 10:
                continue
            _, p_value = stats.normaltest(series)
            skew = series.skew()
            kurt = series.kurtosis()
            if p_value > 0.05:
                self.insights.append({
                    'type': 'positive',
                    'text': f'<b>{col}</b> follows a roughly normal distribution (p={p_value:.3f}) — safe for parametric tests'
                })
            elif abs(skew) > 1.5:
                direction = "right (positive)" if skew > 0 else "left (negative)"
                self.insights.append({
                    'type': 'warning',
                    'text': f'<b>{col}</b> is heavily {direction}-skewed (skew={skew:.2f}) — consider log transformation'
                })
            if kurt > 3:
                self.insights.append({
                    'type': 'info',
                    'text': f'<b>{col}</b> has heavy tails (kurtosis={kurt:.2f}) — extreme values more likely than normal'
                })

    def _detect_correlations(self):
        if len(self.num_cols) < 2:
            return
        corr = self.df[self.num_cols].corr()
        for i in range(len(self.num_cols)):
            for j in range(i + 1, len(self.num_cols)):
                c = corr.iloc[i, j]
                if abs(c) > 0.75:
                    direction = "positively" if c > 0 else "negatively"
                    strength = "very strongly" if abs(c) > 0.9 else "strongly"
                    self.insights.append({
                        'type': 'positive' if c > 0 else 'warning',
                        'text': f'<b>{self.num_cols[i]}</b> & <b>{self.num_cols[j]}</b> are {strength} {direction} correlated (r={c:.2f})'
                    })
                elif abs(c) > 0.5:
                    self.insights.append({
                        'type': 'info',
                        'text': f'<b>{self.num_cols[i]}</b> & <b>{self.num_cols[j]}</b> show moderate correlation (r={c:.2f})'
                    })

    def _detect_trends(self):
        for dt_col in self.dt_cols:
            for num_col in self.num_cols:
                try:
                    sorted_df = self.df[[dt_col, num_col]].dropna().sort_values(dt_col)
                    x_numeric = (sorted_df[dt_col] - sorted_df[dt_col].min()).dt.days
                    slope, intercept, r, p, _ = stats.linregress(x_numeric, sorted_df[num_col])
                    if p < 0.05 and abs(r) > 0.4:
                        direction = "upward 📈" if slope > 0 else "downward 📉"
                        self.insights.append({
                            'type': 'positive' if slope > 0 else 'warning',
                            'text': f'<b>{num_col}</b> shows a statistically significant {direction} trend over time (R²={r**2:.2f}, p={p:.3f})'
                        })
                except Exception:
                    pass
        # Fallback: row-index trend
        if not self.dt_cols:
            for col in self.num_cols:
                series = self.df[col].dropna()
                if len(series) < 20:
                    continue
                x = np.arange(len(series))
                slope, _, r, p, _ = stats.linregress(x, series)
                if p < 0.05 and abs(r) > 0.5:
                    direction = "upward" if slope > 0 else "downward"
                    self.insights.append({
                        'type': 'info',
                        'text': f'<b>{col}</b> has a {direction} trend across rows (r={r:.2f}) — may indicate time-ordering'
                    })

    def _detect_group_disparities(self):
        if not self.cat_cols or not self.num_cols:
            return
        for cat in self.cat_cols[:2]:
            if self.df[cat].nunique() > 15:
                continue
            for num in self.num_cols[:3]:
                groups = [g[num].dropna().values for _, g in self.df.groupby(cat)]
                if len(groups) < 2 or any(len(g) < 3 for g in groups):
                    continue
                try:
                    f_stat, p = stats.f_oneway(*groups)
                    if p < 0.05:
                        means = self.df.groupby(cat)[num].mean()
                        top = means.idxmax()
                        bot = means.idxmin()
                        diff_pct = (means[top] - means[bot]) / abs(means[bot]) * 100 if means[bot] != 0 else 0
                        self.insights.append({
                            'type': 'positive',
                            'text': f'<b>{num}</b> differs significantly across <b>{cat}</b> groups (ANOVA p={p:.3f}). '
                                    f'"{top}" leads by {diff_pct:.0f}% over "{bot}"'
                        })
                except Exception:
                    pass

    def _detect_skewness(self):
        for col in self.num_cols:
            series = self.df[col].dropna()
            zeros = (series == 0).sum()
            if zeros / len(series) > 0.4:
                self.insights.append({
                    'type': 'warning',
                    'text': f'<b>{col}</b> is {zeros/len(series)*100:.0f}% zeros — consider whether these are true zeros or missing data'
                })
            if series.max() / (series.mean() + 1e-9) > 100:
                self.insights.append({
                    'type': 'warning',
                    'text': f'<b>{col}</b> has extreme range — max is {series.max()/series.mean():.0f}x the mean. Check for data errors.'
                })

    def _auto_select_charts(self):
        """Automatically select and create the most informative charts"""
        TEMPLATE = "plotly_dark"

        # 1. Distribution of most variable numeric columns
        if self.num_cols:
            top_var = sorted(self.num_cols, key=lambda c: self.df[c].std() / (self.df[c].mean() + 1e-9), reverse=True)
            for col in top_var[:2]:
                fig = px.histogram(self.df, x=col, nbins=35,
                                   title=f'Distribution — {col}',
                                   template=TEMPLATE, color_discrete_sequence=['#7ee8fa'])
                fig.update_layout(paper_bgcolor='#131720', plot_bgcolor='#0b0e14', font_color='#e0e6f0')
                self.auto_charts.append(('Distribution', fig))

        # 2. Correlation heatmap if multiple numeric cols
        if len(self.num_cols) >= 3:
            corr = self.df[self.num_cols].corr()
            fig = px.imshow(corr, title='Correlation Heatmap',
                            template=TEMPLATE, color_continuous_scale='RdBu_r', aspect='auto')
            fig.update_layout(paper_bgcolor='#131720', plot_bgcolor='#0b0e14', font_color='#e0e6f0')
            self.auto_charts.append(('Correlations', fig))

        # 3. Time series if datetime present
        for dt_col in self.dt_cols[:1]:
            for num_col in self.num_cols[:2]:
                try:
                    ts = self.df[[dt_col, num_col]].dropna().sort_values(dt_col)
                    fig = px.line(ts, x=dt_col, y=num_col,
                                  title=f'{num_col} Over Time', template=TEMPLATE,
                                  color_discrete_sequence=['#a78bfa'])
                    fig.update_layout(paper_bgcolor='#131720', plot_bgcolor='#0b0e14', font_color='#e0e6f0')
                    self.auto_charts.append(('Time Trend', fig))
                except Exception:
                    pass

        # 4. Box plots for categorical breakdown
        for cat in self.cat_cols[:1]:
            if self.df[cat].nunique() <= 12:
                for num in self.num_cols[:2]:
                    fig = px.box(self.df, x=cat, y=num,
                                 title=f'{num} by {cat}', template=TEMPLATE,
                                 color=cat, color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig.update_layout(paper_bgcolor='#131720', plot_bgcolor='#0b0e14',
                                      font_color='#e0e6f0', showlegend=False)
                    self.auto_charts.append(('Group Breakdown', fig))

        # 5. Scatter for top correlated pair
        if len(self.num_cols) >= 2:
            corr = self.df[self.num_cols].corr().abs()
            np.fill_diagonal(corr.values, 0)
            pair = corr.stack().idxmax()
            color_arg = self.cat_cols[0] if self.cat_cols and self.df[self.cat_cols[0]].nunique() <= 8 else None
            fig = px.scatter(self.df, x=pair[0], y=pair[1], color=color_arg,
                             trendline='ols',
                             title=f'Strongest Relationship: {pair[0]} vs {pair[1]}',
                             template=TEMPLATE,
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(paper_bgcolor='#131720', plot_bgcolor='#0b0e14', font_color='#e0e6f0')
            self.auto_charts.append(('Key Relationship', fig))


# ─────────────────────────────────────────────
# MANUAL ANALYSIS ENGINE
# ─────────────────────────────────────────────
TEMPLATE = "plotly_dark"
BGCOLOR = dict(paper_bgcolor='#131720', plot_bgcolor='#0b0e14', font_color='#e0e6f0')

def run_manual_analysis(df, analysis_type, params):
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    cat_cols = list(df.select_dtypes(include=['object', 'category']).columns)

    if analysis_type == 'Scatter Plot':
        fig = px.scatter(df, x=params['x'], y=params['y'],
                         color=params.get('color'),
                         trendline='ols' if params.get('trendline') else None,
                         title=f'{params["y"]} vs {params["x"]}', template=TEMPLATE,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(**BGCOLOR)
        return 'chart', fig

    elif analysis_type == 'Line Plot':
        fig = px.line(df, x=params['x'], y=params['y'],
                      color=params.get('color'),
                      title=f'{params["y"]} over {params["x"]}', template=TEMPLATE,
                      color_discrete_sequence=['#7ee8fa'])
        fig.update_layout(**BGCOLOR)
        return 'chart', fig

    elif analysis_type == 'Bar Chart':
        fig = px.bar(df, x=params['x'], y=params['y'],
                     color=params.get('color'),
                     title=f'{params["y"]} by {params["x"]}', template=TEMPLATE,
                     color_discrete_sequence=['#a78bfa'])
        fig.update_layout(**BGCOLOR)
        return 'chart', fig

    elif analysis_type == 'Histogram':
        fig = px.histogram(df, x=params['x'], nbins=params.get('bins', 30),
                           title=f'Distribution of {params["x"]}', template=TEMPLATE,
                           color_discrete_sequence=['#7ee8fa'])
        fig.update_layout(**BGCOLOR)
        return 'chart', fig

    elif analysis_type == 'Box Plot':
        fig = px.box(df, x=params.get('color'), y=params['x'],
                     title=f'Box Plot of {params["x"]}', template=TEMPLATE,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(**BGCOLOR)
        return 'chart', fig

    elif analysis_type == 'Correlation Heatmap':
        cols = params.get('x') or num_cols
        if len(cols) < 2:
            return 'error', 'Select at least 2 columns'
        corr = df[cols].corr()
        fig = px.imshow(corr, title='Correlation Heatmap', template=TEMPLATE,
                        color_continuous_scale='RdBu_r', aspect='auto')
        fig.update_layout(**BGCOLOR)
        return 'chart', fig

    elif analysis_type == 'Summary Statistics':
        cols = params.get('x') or num_cols
        result = df[cols].describe().T
        result['skewness'] = df[cols].skew()
        result['kurtosis'] = df[cols].kurtosis()
        return 'table', result.round(3)

    elif analysis_type == 'Group Analysis':
        result = df.groupby(params['x'])[params['y']].agg(
            ['count', 'mean', 'std', 'min', 'max']
        ).round(2)
        return 'table', result

    return 'error', 'Unknown analysis type'


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    # Header
    st.markdown('<div class="main-title">⚡ InsightGen</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Automated Data Intelligence Platform</div>', unsafe_allow_html=True)

    # Session init
    for key in ['df', 'cleaned_df', 'quality_issues', 'quality_score', 'insights', 'auto_charts']:
        if key not in st.session_state:
            st.session_state[key] = None

    # Upload
    uploaded_file = st.file_uploader(
        "Drop your dataset here — CSV or Excel",
        type=['csv', 'xlsx', 'xls'],
        label_visibility="collapsed"
    )

    if not uploaded_file:
        st.markdown("""
        <div style="text-align:center; color:#2d3750; padding: 4rem 0; font-size: 0.9rem;">
            Upload a dataset above to begin automatic analysis
        </div>
        """, unsafe_allow_html=True)
        return

    # Load raw data
    try:
        if uploaded_file.name.endswith('.csv'):
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Could not load file: {e}")
        return

    if raw_df.empty or len(raw_df) < 2:
        st.error("Dataset is empty or too small.")
        return

    # Run quality engine on first upload
    if st.session_state.df is None or st.session_state.get('filename') != uploaded_file.name:
        st.session_state.filename = uploaded_file.name
        with st.spinner("🔍 Analyzing data quality..."):
            engine = DataQualityEngine(raw_df)
            cleaned_df, issues, score = engine.run()
            st.session_state.df = raw_df
            st.session_state.cleaned_df = cleaned_df
            st.session_state.quality_issues = issues
            st.session_state.quality_score = score

        with st.spinner("⚡ Generating automatic insights..."):
            ai = AutoInsightEngine(cleaned_df)
            insights, charts = ai.run()
            st.session_state.insights = insights
            st.session_state.auto_charts = charts

    df = st.session_state.cleaned_df
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    cat_cols = list(df.select_dtypes(include=['object', 'category']).columns)
    dt_cols = list(df.select_dtypes(include=['datetime64']).columns)

    # ── TOP METRICS ──
    cols = st.columns(4)
    metrics = [
        (str(len(df)), "Rows"),
        (str(len(df.columns)), "Columns"),
        (f"{st.session_state.quality_score}%", "Data Quality Score"),
        (str(len(st.session_state.insights)), "Auto Insights Found"),
    ]
    for col, (val, label) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── TABS ──
    tab1, tab2, tab3, tab4 = st.tabs(["⚡ Auto Insights", "📊 Manual Analysis", "🧹 Data Quality", "🔎 Data Preview"])

    # TAB 1 — AUTO INSIGHTS
    with tab1:
        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.markdown('<div class="section-tag">Auto-Detected Insights</div>', unsafe_allow_html=True)
            insights = st.session_state.insights or []
            if not insights:
                st.markdown('<div class="insight-card info">No significant patterns detected in this dataset.</div>',
                            unsafe_allow_html=True)
            for ins in insights:
                cls = ins.get('type', 'info')
                st.markdown(f'<div class="insight-card {cls}">{ins["text"]}</div>', unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="section-tag">Auto-Generated Charts</div>', unsafe_allow_html=True)
            charts = st.session_state.auto_charts or []
            if not charts:
                st.info("Not enough data to auto-generate charts.")
            else:
                chart_tabs = st.tabs([label for label, _ in charts])
                for tab, (_, fig) in zip(chart_tabs, charts):
                    with tab:
                        st.plotly_chart(fig, use_container_width=True)

    # TAB 2 — MANUAL ANALYSIS
    with tab2:
        st.markdown('<div class="section-tag">Custom Analysis</div>', unsafe_allow_html=True)

        analysis_type = st.selectbox("Analysis Type", [
            'Scatter Plot', 'Line Plot', 'Bar Chart', 'Histogram',
            'Box Plot', 'Correlation Heatmap', 'Summary Statistics', 'Group Analysis'
        ])

        params = {}
        c1, c2, c3 = st.columns(3)

        if analysis_type in ['Scatter Plot', 'Line Plot', 'Bar Chart']:
            with c1:
                params['x'] = st.selectbox("X-axis", df.columns)
            with c2:
                params['y'] = st.selectbox("Y-axis", df.columns, index=min(1, len(df.columns)-1))
            with c3:
                color_opt = st.selectbox("Group by (optional)", ['None'] + list(df.columns))
                params['color'] = None if color_opt == 'None' else color_opt
            if analysis_type == 'Scatter Plot':
                params['trendline'] = st.checkbox("Add trendline", value=True)

        elif analysis_type == 'Histogram':
            with c1:
                params['x'] = st.selectbox("Column", num_cols) if num_cols else None
            with c2:
                params['bins'] = st.slider("Bins", 5, 100, 30)

        elif analysis_type == 'Box Plot':
            with c1:
                params['x'] = st.selectbox("Numeric column", num_cols) if num_cols else None
            with c2:
                color_opt = st.selectbox("Group by (optional)", ['None'] + cat_cols)
                params['color'] = None if color_opt == 'None' else color_opt

        elif analysis_type == 'Correlation Heatmap':
            params['x'] = st.multiselect("Columns (min 2)", num_cols, default=num_cols[:min(5, len(num_cols))])

        elif analysis_type == 'Summary Statistics':
            params['x'] = st.multiselect("Columns", num_cols, default=num_cols[:min(5, len(num_cols))])

        elif analysis_type == 'Group Analysis':
            with c1:
                params['x'] = st.selectbox("Group by", cat_cols) if cat_cols else st.selectbox("Group by", df.columns)
            with c2:
                params['y'] = st.selectbox("Value column", num_cols) if num_cols else None

        if st.button("Generate Analysis"):
            kind, result = run_manual_analysis(df, analysis_type, params)
            if kind == 'chart':
                st.plotly_chart(result, use_container_width=True)
            elif kind == 'table':
                st.dataframe(result, use_container_width=True)
            else:
                st.error(result)

    # TAB 3 — DATA QUALITY
    with tab3:
        st.markdown('<div class="section-tag">Data Quality Report</div>', unsafe_allow_html=True)

        score = st.session_state.quality_score or 0
        color = "#34d399" if score >= 80 else "#f59e0b" if score >= 50 else "#f87171"
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid {color};">
            <div class="metric-value" style="color:{color};">{score}%</div>
            <div class="metric-label">Overall Data Quality Score</div>
            <div class="quality-bar-bg">
                <div style="background:{color}; width:{score}%; height:6px; border-radius:4px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        issues = st.session_state.quality_issues or []
        if not issues:
            st.markdown('<div class="insight-card positive">✓ No data quality issues detected.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f"**{len(issues)} issues detected and auto-resolved:**")
            for issue in issues:
                sev = issue.get('severity', 'info')
                icon = "⚠️" if sev == 'warning' else "ℹ️"
                st.markdown(f'<div class="insight-card {sev}">{icon} {issue["msg"]}</div>',
                            unsafe_allow_html=True)

        # Side by side raw vs clean
        st.markdown("---")
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**Raw Data** (original)")
            st.dataframe(st.session_state.df.head(8), use_container_width=True)
        with r2:
            st.markdown("**Cleaned Data** (after auto-processing)")
            st.dataframe(df.head(8), use_container_width=True)

    # TAB 4 — DATA PREVIEW
    with tab4:
        st.markdown('<div class="section-tag">Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

        with st.expander("Column Info"):
            info_df = pd.DataFrame({
                'dtype': df.dtypes,
                'non_null': df.notna().sum(),
                'nulls': df.isnull().sum(),
                'unique': df.nunique(),
            })
            st.dataframe(info_df, use_container_width=True)


if __name__ == "__main__":
    main()