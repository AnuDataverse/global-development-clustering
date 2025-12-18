import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Global Development Decision System",
    layout="wide"
)

st.title("🌍 Global Development Decision System")
st.markdown(
    "A Machine Learning Decision Support System for classifying country development levels."
)

# ---------------------------------------------------------
# Load Resources
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("development_cluster_pipeline_kmeans.joblib")

@st.cache_data
def load_data():
    return pd.read_csv("clustered_data.csv")

bundle = load_model()
df = load_data().copy()

pipeline = bundle["pipeline"]
model = bundle.get("model", pipeline.steps[-1][1])
feature_cols = bundle["feature_cols"]

# ---------------------------------------------------------
# Detect Country Column
# ---------------------------------------------------------
def detect_country_column(df):
    for c in ["Country", "country", "Country_Name", "country_name"]:
        if c in df.columns:
            return c
    return None

country_col = detect_country_column(df)

# ---------------------------------------------------------
# Auto-label clusters
# ---------------------------------------------------------
cluster_col = next(
    (c for c in ["Cluster", "kmeans", "cluster_label"] if c in df.columns),
    None
)

if cluster_col and "GDP" in df.columns:
    ranked = (
        df.groupby(cluster_col)["GDP"]
        .mean()
        .sort_values()
        .index.tolist()
    )

    status_map = {
        ranked[0]: "Poor Country",
        ranked[1]: "Developing Country",
        ranked[2]: "Developed Country",
    }

    df["Status"] = df[cluster_col].map(status_map)
else:
    df["Status"] = "Unknown"

# ---------------------------------------------------------
# Sidebar Prediction
# ---------------------------------------------------------
st.sidebar.header("📝 Predict Development Status")

input_data = {}
for col in feature_cols:
    input_data[col] = st.sidebar.number_input(
        col,
        float(df[col].min()),
        float(df[col].max()),
        float(df[col].mean())
    )

if st.sidebar.button("Predict"):
    input_df = pd.DataFrame([input_data])
    X = pipeline.transform(input_df)
    pred = model.predict(X)[0]

    label = status_map.get(pred, f"Cluster {pred}")
    st.sidebar.success(f"Prediction: {label}")

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Analysis", "World Map", "Clusters", "Feature Importance"]
)

# ---------------------------------------------------------
# TAB 1: Country Analysis
# ---------------------------------------------------------
with tab1:
    if country_col:
        country = st.selectbox("Select Country", df[country_col].unique())
        row = df[df[country_col] == country].iloc[0]

        st.write("**Status:**", row["Status"])

# ---------------------------------------------------------
# TAB 2: World Map
# ---------------------------------------------------------
with tab2:
    if country_col:
        fig = px.choropleth(
            df,
            locations=country_col,
            locationmode="country names",
            color="Status",
            hover_name=country_col,
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Country column not found.")

# ---------------------------------------------------------
# TAB 3: PCA Clusters (NO DOUBLE PCA)
# ---------------------------------------------------------
with tab3:
    X_pca = pipeline.transform(df[feature_cols])
    plot_df = pd.DataFrame(X_pca[:, :2], columns=["PC1", "PC2"])
    plot_df["Status"] = df["Status"]
    plot_df["Country"] = df[country_col] if country_col else plot_df.index

    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="Status",
        hover_name="Country"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TAB 4: Feature Importance (NO matplotlib)
# ---------------------------------------------------------
with tab4:
    importance = df.groupby("Status")[feature_cols].mean().T
    st.dataframe(importance.round(2))

st.caption("Unsupervised ML – Global Development Classification")
