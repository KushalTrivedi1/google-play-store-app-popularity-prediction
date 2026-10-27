import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Play Store Predictor", page_icon="📱", layout="wide")

INDIGO, VIOLET, ORANGE, GREEN = "#4f46e5", "#7c3aed", "#f97316", "#10b981"
CSV_PATH = "playstore_apps_cleaned.csv"


# ------------------------------------------------------------------ data + model
@st.cache_resource(show_spinner="Loading data and training model...")
def load_everything():
    df = pd.read_csv(CSV_PATH)

    data = df.copy().dropna(subset=["content_rating"])
    data["log_reviews"] = np.log1p(data["reviews"])
    data["is_free"] = data["free"].astype(int)
    data["pricing"] = data["is_free"].map({1: "Free", 0: "Paid"})

    top_cats = data["category"].value_counts().head(8).index.tolist()
    data["category_grp"] = data["category"].where(data["category"].isin(top_cats), "Other")

    cat_options = sorted(data["category_grp"].unique())
    cr_options = sorted(data["content_rating"].unique())

    X = pd.get_dummies(
        data[["log_reviews", "rating", "is_free", "category_grp", "content_rating"]],
        columns=["category_grp", "content_rating"], drop_first=True, dtype=float)
    y = data["log_installs"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)

    test_pred = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, test_pred),
        "mae": mean_absolute_error(y_test, test_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
        "resid_std": np.std(y_train - model.predict(X_train)),
    }
    return dict(data=data, cat_options=cat_options, cr_options=cr_options, X=X, y=y,
                X_train=X_train, X_test=X_test, y_test=y_test, model=model,
                test_pred=test_pred, **metrics)


S = load_everything()
data, X, y, model = S["data"], S["X"], S["y"], S["model"]
cat_options, cr_options = S["cat_options"], S["cr_options"]
r2, mae, rmse, resid_std = S["r2"], S["mae"], S["rmse"], S["resid_std"]


# ------------------------------------------------------------------ helpers
def predict_log(reviews, rating, is_free, category, content_rating):
    row = pd.DataFrame([{
        "log_reviews": np.log1p(max(reviews or 0, 0)), "rating": rating,
        "is_free": is_free, "category_grp": category, "content_rating": content_rating}])
    row = pd.get_dummies(row, columns=["category_grp", "content_rating"],
                         dtype=float).reindex(columns=X.columns, fill_value=0)
    return float(model.predict(row)[0])


def tier(n):
    if n < 10_000:
        return "Niche", "#64748b", "🌱"
    if n < 1_000_000:
        return "Growing", "#0ea5e9", "📈"
    if n < 100_000_000:
        return "Popular", "#f97316", "🔥"
    return "Blockbuster", "#7c3aed", "🚀"


def stat_card(label, value, color):
    return (f'<div style="flex:1;min-width:150px;background:white;border:1px solid #e2e8f0;'
            f'border-radius:14px;padding:14px 18px;color:#0f172a">'
            f'<div style="font-size:12px;color:#64748b;letter-spacing:.06em">{label}</div>'
            f'<div style="font-size:26px;font-weight:800;color:{color}">{value}</div></div>')


def result_card(est, low, high, pct):
    name, color, icon = tier(est)
    return f"""
    <div style="background:white;border:1px solid #e2e8f0;border-radius:16px;padding:22px;color:#0f172a">
      <div style="font-size:13px;color:#64748b;letter-spacing:.06em">PREDICTED INSTALLS</div>
      <div style="font-size:42px;font-weight:800;color:{INDIGO};line-height:1.1">{est:,.0f}</div>
      <div style="margin:6px 0 14px;color:#475569">95% range: <b>{low:,.0f}</b> to <b>{high:,.0f}</b></div>
      <span style="background:{color};color:white;padding:5px 14px;border-radius:999px;font-weight:600">
        {icon} {name}</span>
      <div style="margin-top:18px;font-size:13px;color:#475569">
        Ahead of <b>{pct:.0f}%</b> of the {len(data)} apps in the dataset</div>
      <div style="background:#e2e8f0;border-radius:999px;height:10px;margin-top:6px">
        <div style="width:{pct:.0f}%;background:linear-gradient(90deg,{INDIGO},{VIOLET});
                    height:10px;border-radius:999px"></div></div>
    </div>"""


def explore_fig(chart, category):
    d = data if category == "All" else data[data["category_grp"] == category]
    if chart == "Rating distribution":
        fig = px.histogram(d, x="rating", nbins=20, color_discrete_sequence=[INDIGO])
    elif chart == "Installs distribution (log)":
        fig = px.histogram(d, x="log_installs", nbins=20, color_discrete_sequence=[ORANGE])
    elif chart == "Reviews vs installs":
        fig = px.scatter(d, x="log_reviews", y="log_installs", color="category_grp",
                         hover_name="name", opacity=0.7)
    elif chart == "Free vs paid":
        fig = px.pie(d, names="pricing", color_discrete_sequence=[INDIGO, ORANGE], hole=0.45)
    elif chart == "Rating by category":
        fig = px.box(data, x="category_grp", y="rating", color="category_grp")
    else:
        counts = data["category_grp"].value_counts().reset_index()
        counts.columns = ["category", "apps"]
        fig = px.bar(counts, x="category", y="apps", color_discrete_sequence=[VIOLET])
    fig.update_layout(template="plotly_white", height=430, margin=dict(l=20, r=20, t=40, b=20),
                      title=f"{chart}  |  {category}  (n = {len(d)})")
    return fig


def model_report_fig():
    coefs = pd.Series(model.coef_, index=X.columns)
    top = coefs.reindex(coefs.abs().sort_values(ascending=False).head(10).index)[::-1]
    resid = S["y_test"] - S["test_pred"]

    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        "Top 10 coefficients", "Actual vs predicted (test)", "Residual distribution"))
    fig.add_trace(go.Bar(x=top.values, y=top.index, orientation="h",
                         marker_color=[INDIGO if v > 0 else "#ef4444" for v in top.values]), row=1, col=1)
    fig.add_trace(go.Scatter(x=S["y_test"], y=S["test_pred"], mode="markers",
                             marker=dict(color=GREEN, opacity=0.6)), row=1, col=2)
    fig.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()], mode="lines",
                             line=dict(color="red", dash="dash")), row=1, col=2)
    fig.add_trace(go.Histogram(x=resid, nbinsx=20, marker_color="#94a3b8"), row=1, col=3)
    fig.update_layout(template="plotly_white", height=420, showlegend=False,
                      margin=dict(l=20, r=20, t=50, b=20))
    return fig


# ------------------------------------------------------------------ page chrome
st.markdown("""
<style>
.block-container {max-width: 1200px; padding-top: 2rem;}
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed); border: none; font-weight: 700;}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="background:linear-gradient(135deg,#4f46e5,#7c3aed);padding:30px;border-radius:18px;color:white">
  <h1 style="margin:0;font-size:30px;color:white;padding:0">📱 Play Store App Popularity Predictor</h1>
  <p style="margin:8px 0 0;opacity:.92">Regression-based prediction of app installs from reviews, rating,
  pricing, category and content rating.</p>
</div>
<div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:14px;margin-bottom:18px">
  {stat_card("APPS SCRAPED", f"{len(data):,}", INDIGO)}
  {stat_card("CATEGORIES", len(cat_options), VIOLET)}
  {stat_card("MODEL R² (TEST)", f"{r2:.2f}", GREEN)}
  {stat_card("MEDIAN INSTALLS", f"{data['min_installs'].median():,.0f}", ORANGE)}
</div>
""", unsafe_allow_html=True)

tab_predict, tab_explore, tab_report, tab_search = st.tabs(
    ["🔮 Predict", "📊 Data Explorer", "📈 Model Report", "🔎 Search Apps"])

# ------------------------------------------------------------------ Predict
with tab_predict:
    left, right = st.columns([2, 3], gap="large")

    with left:
        st.markdown("### Describe your app")
        reviews = st.number_input("Number of reviews", min_value=0, value=15000, step=100)
        rating = st.slider("Average rating", 1.0, 5.0, 4.3, 0.1)
        pricing = st.radio("Pricing", ["Free", "Paid"], horizontal=True)
        category = st.selectbox("Category", cat_options)
        content_rating = st.selectbox("Content rating", cr_options)

    is_free = int(pricing == "Free")
    p = predict_log(reviews, rating, is_free, category, content_rating)
    est = max(np.expm1(p), 0)
    low = max(np.expm1(p - 1.96 * resid_std), 0)
    high = np.expm1(p + 1.96 * resid_std)
    pct = (data["log_installs"] < p).mean() * 100

    with right:
        st.markdown(result_card(est, low, high, pct), unsafe_allow_html=True)

        fig1 = go.Figure(go.Histogram(x=data["log_installs"], nbinsx=25, marker_color="#c7d2fe"))
        fig1.add_vrect(x0=p - 1.96 * resid_std, x1=p + 1.96 * resid_std,
                       fillcolor=INDIGO, opacity=0.12, line_width=0)
        fig1.add_vline(x=p, line_color=INDIGO, line_width=3,
                       annotation_text="Your app", annotation_position="top")
        fig1.update_layout(title="Where your app sits among scraped apps", template="plotly_white",
                           xaxis_title="log(installs)", yaxis_title="Number of apps",
                           height=320, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
        st.plotly_chart(fig1, width="stretch")

    # What-if analysis
    ratings = np.arange(1, 5.01, 0.25)
    rev_grid = np.logspace(1, 7, 25)
    est_vs_rating = [np.expm1(predict_log(reviews, r_, is_free, category, content_rating)) for r_ in ratings]
    est_vs_reviews = [np.expm1(predict_log(rv, rating, is_free, category, content_rating)) for rv in rev_grid]

    fig2 = make_subplots(rows=1, cols=2, subplot_titles=("If the rating changes", "If the reviews change"))
    fig2.add_trace(go.Scatter(x=ratings, y=est_vs_rating, mode="lines+markers",
                              line=dict(color=ORANGE, width=3)), row=1, col=1)
    fig2.add_trace(go.Scatter(x=rev_grid, y=est_vs_reviews, mode="lines",
                              line=dict(color=GREEN, width=3)), row=1, col=2)
    fig2.update_xaxes(title_text="Rating", row=1, col=1)
    fig2.update_xaxes(title_text="Reviews", type="log", row=1, col=2)
    fig2.update_yaxes(title_text="Predicted installs", type="log")
    fig2.update_layout(template="plotly_white", height=320, showlegend=False,
                       margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig2, width="stretch")

    # Similar apps
    st.markdown("### Similar apps in the same category")
    pool = data[data["category_grp"] == category].copy()
    pool["gap"] = (pool["log_reviews"] - np.log1p(max(reviews or 0, 0))).abs()
    similar = pool.sort_values("gap").head(6)[["name", "rating", "reviews", "min_installs"]]
    similar.columns = ["App", "Rating", "Reviews", "Installs (lower bound)"]
    st.dataframe(similar, width="stretch", hide_index=True)

# ------------------------------------------------------------------ Data Explorer
with tab_explore:
    c1, c2 = st.columns(2)
    chart = c1.selectbox("Chart", ["Rating distribution", "Installs distribution (log)",
                                   "Reviews vs installs", "Free vs paid",
                                   "Rating by category", "Apps per category"])
    cat_f = c2.selectbox("Filter by category", ["All"] + cat_options)
    st.plotly_chart(explore_fig(chart, cat_f), width="stretch")

# ------------------------------------------------------------------ Model Report
with tab_report:
    st.markdown(
        '<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:12px">'
        f'{stat_card("R² (TEST)", f"{r2:.3f}", GREEN)}'
        f'{stat_card("RMSE", f"{rmse:.3f}", ORANGE)}'
        f'{stat_card("MAE", f"{mae:.3f}", VIOLET)}'
        f'{stat_card("TRAIN / TEST", f"{len(S["X_train"])} / {len(S["X_test"])}", INDIGO)}'
        "</div>", unsafe_allow_html=True)
    st.plotly_chart(model_report_fig(), width="stretch")
    st.markdown("Errors are on the **log(installs)** scale. Predictions are approximate because "
                "Google Play only reports install *ranges* (for example '10,000+').")

# ------------------------------------------------------------------ Search Apps
with tab_search:
    c1, c2, c3, c4 = st.columns(4)
    q = c1.text_input("App name contains", placeholder="e.g. whatsapp")
    s_cat = c2.selectbox("Category", ["All"] + cat_options, key="search_cat")
    s_min = c3.slider("Minimum rating", 1.0, 5.0, 1.0, 0.1)
    s_sort = c4.selectbox("Sort by", ["min_installs", "reviews", "rating"])

    t = data.copy()
    if q:
        t = t[t["name"].str.contains(q, case=False, na=False)]
    if s_cat != "All":
        t = t[t["category_grp"] == s_cat]
    t = t[t["rating"] >= s_min].sort_values(s_sort, ascending=False)
    out = t[["name", "category", "rating", "reviews", "min_installs"]].head(100)
    out.columns = ["App", "Category", "Rating", "Reviews", "Installs (lower bound)"]
    st.caption(f"Showing {len(out)} of {len(t)} matching apps")
    st.dataframe(out, width="stretch", hide_index=True)
