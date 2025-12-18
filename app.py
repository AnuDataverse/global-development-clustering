import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Global Development Decision System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌍 Global Development Decision System")
st.markdown("""
**A Machine Learning Decision Support System.**  
This tool analyzes socio-economic indicators to classify development levels,
visualize global trends, and generate actionable policy recommendations.
""")

# ---------------------------------------------------------
# 2. Load Model & Data (Safe Caching)
# ---------------------------------------------------------
@st.cache_resource
def load_model_bundle():
    return joblib.load("development_cluster_pipeline_kmeans.joblib")

@st.cache_data
def load_dataset():
    return pd.read_csv("clustered_data.csv")

bundle = load_model_bundle()
df = load_dataset().copy()

pipeline = bundle["pipeline"]
model = bundle.get("model", pipeline.steps[-1][1])
feature_cols = bundle["feature_cols"]

# ---------------------------------------------------------
# 3. Detect Country Column Automatically
# ---------------------------------------------------------
def detect_country_column(df):
    for c in ["Country", "country", "Country_Name", "country_name"]:
        if c in df.columns:
            return c
    return None

country_col = detect_country_column(df)

# ---------------------------------------------------------
# 4. Auto-Label Clusters (Poor → Developed)
# ---------------------------------------------------------
def get_status_map(df):
    cluster_col = next(
        (c for c in ["Cluster", "kmeans", "cluster_label"] if c in df.columns),
        None
    )
    if cluster_col is None:
        return {}, None

    wealth_col = next(
        (c for c in ["GDP", "GDP_per_capita", "Log_GDP"] if c in df.columns),
        None
    )
    if wealth_col is None:
        return {}, cluster_col

    ranked = (
        df.groupby(cluster_col)[wealth_col]
        .mean()
        .sort_values()
        .index.tolist()
    )

    mapping = {
        ranked[0]: "Poor Country",
        ranked[1]: "Developing Country",
        ranked[2]: "Developed Country"
    }

    return mapping, cluster_col

status_map, cluster_col = get_status_map(df)

if cluster_col:
    df["Status"] = df[cluster_col].map(status_map)
else:
    df["Status"] = "Unknown"

# ---------------------------------------------------------
# 5. Sidebar Prediction
# ---------------------------------------------------------
st.sidebar.header("📝 Predict Development Status")

input_data = {}
for col in feature_cols:
    min_v = float(df[col].min())
    max_v = float(df[col].max())
    mean_v = float(df[col].mean())
    step = max((max_v - min_v) / 100, 0.01)

    input_data[col] = st.sidebar.number_input(
        col,
        min_value=min_v,
        max_value=max_v,
        value=mean_v,
        step=step
    )

if st.sidebar.button("Predict Status"):
    input_df = pd.DataFrame([input_data])
    processed = pipeline.transform(input_df)

    pred_id = model.predict(processed)[0]
    pred_label = status_map.get(pred_id, f"Cluster {pred_id}")

    distances = model.transform(processed)
    confidence = 1 / (1 + distances.min())

    if "Developed" in pred_label:
        st.sidebar.success(pred_label)
    elif "Developing" in pred_label:
        st.sidebar.warning(pred_label)
    else:
        st.sidebar.error(pred_label)

    st.sidebar.metric("Confidence", f"{confidence:.2f}")

    st.session_state["new_pred"] = {
        "coords": processed[0][:2],
        "label": pred_label
    }

# ---------------------------------------------------------
# 6. Dashboard Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Analysis & Policy",
    "🗺️ World Map",
    "📊 PCA Visualization",
    "📈 Feature Importance"
])

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("🌍 Country Comparison")

    if country_col:
        country = st.selectbox(
            "Select Country",
            sorted(df[country_col].unique())
        )

        row = df[df[country_col] == country].iloc[0]
        developed = df[df["Status"] == "Developed Country"]

        target_avg = developed[feature_cols].mean()
        current = row[feature_cols]

        st.info(f"**Current Status:** {row['Status']}")

        comp = pd.DataFrame({
            "Current": current,
            "Developed Avg": target_avg
        })

        norm = (comp - comp.min()) / (comp.max() - comp.min() + 1e-9)
        st.bar_chart(norm.head(10))

        st.subheader("🧭 Policy Recommendations")

        HIGHER = ["GDP", "Life", "Internet", "Phone"]
        LOWER = ["Mortality", "Birth", "Inflation"]

        recs = []
        for f in feature_cols:
            if any(x in f for x in HIGHER) and current[f] < target_avg[f] * 0.8:
                recs.append(f"📈 Increase **{f}**")
            if any(x in f for x in LOWER) and current[f] > target_avg[f] * 1.2:
                recs.append(f"🚨 Reduce **{f}**")

        if recs:
            for r in recs[:5]:
                st.write(r)
        else:
            st.success("Country meets most developed benchmarks.")

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("🗺️ Global Development Map")

    if country_col:
        fig = px.choropleth(
            df,
            locations=country_col,
            locationmode="country names",
            color="Status",
            hover_name=country_col,
            template="plotly_white",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Country column not found.")

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("📊 PCA Cluster Visualization")

    X_pca = pipeline.transform(df[feature_cols])
    plot_df = pd.DataFrame(X_pca[:, :2], columns=["PC1", "PC2"])
    plot_df["Status"] = df["Status"]
    plot_df["Country"] = df[country_col] if country_col else df.index

    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="Status",
        hover_name="Country",
        height=600
    )

    if "new_pred" in st.session_state:
        p = st.session_state["new_pred"]
        fig.add_scatter(
            x=[p["coords"][0]],




