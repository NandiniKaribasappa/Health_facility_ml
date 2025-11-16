# app.py - Health Facility clusters
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

st.set_page_config(page_title="Health Facility Clustering", layout="wide")
st.title("Health Facility Clustering")
st.write("View cluster outputs, PCA plot and KPIs for the project.")

@st.cache_data
def load_data():
    df = pd.read_csv("health_facility_clusters.csv")
    return df

@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")
    kmeans = joblib.load("kmeans_health_model.pkl")
    return scaler, encoder, kmeans

try:
    df = load_data()
    scaler, encoder, kmeans = load_models()
    st.success("Data and model loaded.")
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

st.subheader("Dataset preview")
st.dataframe(df.head(50))

st.subheader("Key KPIs")
col1, col2, col3 = st.columns(3)
col1.metric("Total Facilities", df.shape[0])
col2.metric("Unique States", df["State Name"].nunique())
col3.metric("Clusters", df["cluster"].nunique())

st.subheader("Cluster distribution")
st.bar_chart(df["cluster"].value_counts().sort_index())

st.subheader("Dominant facility type per cluster")
ft = df.groupby("cluster")["Facility Type"].agg(lambda x: x.mode()[0] if len(x)>0 else "")
st.table(ft.to_frame("Dominant Facility Type"))

st.subheader("PCA Visualization")
features = ["Latitude", "Longitude"]
X = df[features].fillna(0).values
pca = PCA(n_components=2)
pca_points = pca.fit_transform(X)
fig, ax = plt.subplots(figsize=(8,6))
scatter = ax.scatter(pca_points[:,0], pca_points[:,1], c=df["cluster"], s=10, cmap="tab10")
ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2"); ax.set_title("PCA 2D - Clusters")
st.pyplot(fig)

st.subheader("Interactive filters")
state = st.selectbox("State", ["All"] + sorted(df["State Name"].unique()))
if state != "All":
    df = df[df["State Name"] == state]
clusters = st.multiselect("Choose clusters", sorted(df["cluster"].unique()), default=sorted(df["cluster"].unique()))
df_filtered = df[df["cluster"].isin(clusters)]
st.write("Filtered results")
st.dataframe(df_filtered.head(200))
csv = df_filtered.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered data", csv, "filtered_health_facilities.csv")
