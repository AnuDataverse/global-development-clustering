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
**A Machine Learning Decision Support System.** This tool analyzes socio-economic indicators to classify development levels, 
visualize global trends, and generate actionable policy recommendations.
""")

# ---------------------------------------------------------
# 2. Robust Resource Loading (Split Caching)
# ---------------------------------------------------------
@st.cache_resource
def load_model_bundle():
    try:
        return joblib.load("development_cluster_pipeline_kmeans.joblib")
    except FileNotFoundError:
        st.error("❌ Model file missing. Upload 'development_cluster_pipeline_kmeans.joblib'.")
        st.stop()

@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("clustered_data.csv")
    except FileNotFoundError:
        try:
            return pd.read_csv("clustered_data (1).csv")
        except FileNotFoundError:
            st.error("❌ Data file missing. Upload 'clustered_data.csv'.")
            st.stop()

try:
    bundle = load_model_bundle()
    df_orig = load_dataset()
    df = df_orig.copy()

    pipeline = bundle["pipeline"]
    # Handle cases where model might be nested or direct
    # If "model" key is missing, try to get it from the last step of the pipeline
    model = bundle.get("model", pipeline.steps[-1][1] if hasattr(pipeline, "steps") else None)
    feature_cols = bundle["feature_cols"]
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# ---------------------------------------------------------
# 3. Auto-Labeling Logic (Poor / Developing / Developed)
# ---------------------------------------------------------
def get_development_status_map(df):
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
        return {i: f"Cluster {i}" for i in df[cluster_col].unique()}, cluster_col

    # Sort clusters by average wealth (Low -> High)
    ranked = (
        df.groupby(cluster_col)[wealth_col]
        .mean()
        .sort_values()
        .index.tolist()
    )

    mapping = {}
    if len(ranked) == 3:
        mapping[ranked[0]] = "Poor Country"
        mapping[ranked[1]] = "Developing Country"
        mapping[ranked[2]] = "Developed Country"
    elif len(ranked) == 2:
        mapping[ranked[0]] = "Developing"
        mapping[ranked[1]] = "Developed"
    else:
         for rank, cid in enumerate(ranked):
            mapping[cid] = f"Cluster {cid} (Rank {rank+1})"

    return mapping, cluster_col

status_map, cluster_col = get_development_status_map(df)

if cluster_col:
    df["Status"] = df[cluster_col].map(status_map)
else:
    df["Status"] = "Unknown"

# ---------------------------------------------------------
# 4. Sidebar: Prediction
# ---------------------------------------------------------
st.sidebar.header("📝 Predict Country Development")

input_data = {}

for col in feature_cols:
    if col in df.columns:
        min_v = float(df[col].min())
        max_v = float(df[col].max())
        mean_v = float(df[col].mean())
    else:
        min_v, max_v, mean_v = 0.0, 100.0, 50.0
    
    step = max((max_v - min_v) / 100, 0.01)

    input_data[col] = st.sidebar.number_input(
        col,
        min_value=min_v,
        max_value=max_v,
        value=mean_v,
        step=step
    )

if st.sidebar.button("Predict Status"):
    try:
        input_df = pd.DataFrame([input_data])
        
        processed = pipeline.transform(input_df)
        pred_id = model.predict(processed)[0]
        pred_label = status_map.get(pred_id, f"Cluster {pred_id}")

        distances = model.transform(processed)
        confidence = 1 / (1 + distances.min())

        if "Developed" in pred_label:
            st.sidebar.success(f"🎉 {pred_label}")
        elif "Developing" in pred_label:
            st.sidebar.warning(f"⚠️ {pred_label}")
        else:
            st.sidebar.error(f"🛑 {pred_label}")
            
        st.sidebar.metric("Confidence Score", f"{confidence:.2f}")

        st.session_state["new_pred"] = {
            "label": pred_label,
            "coords": processed[0][:2], 
            "confidence": confidence
        }

    except Exception as e:
        st.sidebar.error(f"Prediction Error: {e}")

# ---------------------------------------------------------
# 5. Main Dashboard (Tabs)
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Analysis & Policy", 
    "🗺️ World Map", 
    "📊 Cluster Visualization", 
    "📈 Feature Importance"
])

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("🌍 Country Development Comparison")
    
    if "Country" in df.columns:
        selected_country = st.selectbox(
            "Select a Country", 
            sorted(df["Country"].unique())
        )
        
        country_row = df[df["Country"] == selected_country]
        
        if not country_row.empty:
            current_status = country_row["Status"].values[0]
            
            developed_df = df[df["Status"] == "Developed Country"]
            if developed_df.empty: 
                developed_df = df
                
            target_avg = developed_df[feature_cols].mean()
            current_vals = country_row[feature_cols].iloc[0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Current Status:** {current_status}")
            with col2:
                if "GDP" in current_vals:
                    gap = current_vals["GDP"] - target_avg["GDP"]
                    st.metric("GDP Gap vs Developed", f"${gap:,.0f}")
                elif "Log_GDP" in current_vals:
                     st.metric("Log GDP Gap", f"{current_vals['Log_GDP'] - target_avg['Log_GDP']:.2f}")

            comp_df = pd.DataFrame({
                "Current": current_vals,
                "Developed Avg": target_avg
            })
            
            norm_comp = (comp_df - comp_df.min()) / (comp_df.max() - comp_df.min() + 1e-9)
            st.bar_chart(norm_comp.head(10)) 
            
            st.subheader("🧭 Policy Recommendations")
            
            HIGHER_BETTER = ["GDP", "Life", "Internet", "Phone", "Tourism"]
            LOWER_BETTER = ["Mortality", "Birth", "Tax", "Inflation"]
            
            recommendations = []
            for feat in feature_cols:
                val, tgt = current_vals[feat], target_avg[feat]
                
                if any(x in feat for x in HIGHER_BETTER) and val < tgt * 0.8:
                    recommendations.append(f"📈 **{feat}**: Increase investment.")
                
                if any(x in feat for x in LOWER_BETTER) and val > tgt * 1.2:
                    recommendations.append(f"🚨 **{feat}**: Reduce through policy reform.")
            
            if recommendations:
                for r in recommendations[:5]:
                    st.write(r)
            else:
                st.success("✅ Country meets most developed benchmarks.")

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("🗺️ Global Development Status Map")
    
    if "Country" in df.columns:
        fig_map = px.choropleth(
            df,
            locations="Country",
            locationmode="country names",
            color="Status",
            hover_name="Country",
            color_discrete_map={
                "Poor Country": "#d62728",
                "Developing Country": "#ff7f0e",
                "Developed Country": "#2ca02c"
            },
            template="plotly_white",
            height=600
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("Country column not found for map visualization.")

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("📊 PCA Cluster Visualization")
    
    X_pca = pipeline.transform(df[feature_cols])
    plot_df = pd.DataFrame(X_pca[:, :2], columns=["PC1", "PC2"])
    plot_df["Status"] = df["Status"]
    plot_df["Country"] = df["Country"] if "Country" in df.columns else df.index
    
    fig_pca = px.scatter(
        plot_df, 
        x="PC1", 
        y="PC2", 
        color="Status", 
        hover_name="Country",
        template="plotly_white",
        height=600
    )
    
    if "new_pred" in st.session_state:
        p = st.session_state["new_pred"]
        fig_pca.add_scatter(
            x=[p["coords"][0]], 
            y=[p["coords"][1]], 
            mode="markers+text", 
            marker=dict(size=25, color="blue", symbol="star"), 
            text=["NEW"], 
            name=p["label"]
        )
        
    st.plotly_chart(fig_pca, use_container_width=True)

# ---------------- TAB 4 ----------------
with tab4:
    st.subheader("📈 Feature Importance by Development Level")
    
    importance = df.groupby("Status")[feature_cols].mean().T
    cols = ["Poor Country", "Developing Country", "Developed Country"]
    importance = importance[[c for c in cols if c in importance.columns]]
    
    st.dataframe(importance.style.background_gradient(axis=1))

st.markdown("---")
st.caption("📌 Unsupervised ML-based Global Development Decision Support System")
