import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Used Oil Pilot Dashboard – Maharashtra", layout="wide")

st.title("Used Oil Collection Pilot – Maharashtra")
st.caption("Layer 1: India map + filters (Annual/Monthly/Weekly thresholds + name search)")

uploaded = st.sidebar.file_uploader("Upload MH geocoded Excel", type=["xlsx"])

if uploaded is None:
    st.info("Upload MH_geocoded.xlsx from the sidebar to start.")
    st.stop()

@st.cache_data
def load_excel(file) -> pd.DataFrame:
    df = pd.read_excel(file)

    for c in ["Generation_MT", "Monthly_Generation_MT", "Weekly_Generation_MT", "Latitude", "Longitude"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")

    for c in ["Customer_Code", "Customer_Name", "Address", "City", "Pin_Code", "Geocode_Status"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    df = df[df["Latitude"].notna() & df["Longitude"].notna()].copy()
    return df

df = load_excel(uploaded)

depot_mask = df["Customer_Code"].astype(str).str.contains("DEPOT_ABC", na=False)
if not depot_mask.any():
    st.error("Depot row not found. Expecting Customer_Code containing 'DEPOT_ABC'.")
    st.stop()

depot = df[depot_mask].iloc[0]
sites = df[~depot_mask].copy()

st.sidebar.header("Filters")
name_q = st.sidebar.text_input("Search by name", value="", placeholder="type workshop name")

annual_min = st.sidebar.number_input("Annual ≥ (MT)", min_value=0.0, value=0.0, step=0.5)
monthly_min = st.sidebar.number_input("Monthly ≥ (MT)", min_value=0.0, value=0.0, step=0.1)
weekly_min  = st.sidebar.number_input("Weekly ≥ (MT)", min_value=0.0, value=0.0, step=0.05)

top_n = st.sidebar.number_input("Top N by Annual MT (0 = all)", min_value=0, max_value=1000, value=0, step=5)

f = sites.copy()
if name_q.strip():
    q = name_q.strip().lower()
    f = f[f["Customer_Name"].str.lower().str.contains(q)]

f = f[(f["Generation_MT"] >= annual_min) &
      (f["Monthly_Generation_MT"] >= monthly_min) &
      (f["Weekly_Generation_MT"] >= weekly_min)]

if top_n and top_n > 0:
    f = f.sort_values("Generation_MT", ascending=False).head(int(top_n))

total_annual = float(f["Generation_MT"].sum())
total_monthly = float(f["Monthly_Generation_MT"].sum())
total_weekly = float(f["Weekly_Generation_MT"].sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Workshops (filtered)", f"{len(f)}")
k2.metric("Annual total (MT)", f"{total_annual:,.2f}")
k3.metric("Monthly total (MT)", f"{total_monthly:,.2f}")
k4.metric("Weekly total (MT)", f"{total_weekly:,.2f}")

plot_df = pd.concat([pd.DataFrame([depot]), f], ignore_index=True)

wk = plot_df["Weekly_Generation_MT"].fillna(0).astype(float)
size = 10 + (wk / (wk.max() if wk.max() > 0 else 1)) * 25
plot_df["_marker_size"] = size.clip(10, 35)

fig = px.scatter_mapbox(
    plot_df,
    lat="Latitude",
    lon="Longitude",
    hover_name="Customer_Name",
    hover_data={
        "City": True,
        "Pin_Code": True,
        "Generation_MT": ":.2f",
        "Monthly_Generation_MT": ":.2f",
        "Weekly_Generation_MT": ":.2f",
        "Geocode_Status": True,
        "_marker_size": False,
    },
    size="_marker_size",
    zoom=6,
    height=650,
)

fig.update_layout(
    mapbox_style="open-street-map",
    mapbox=dict(center=dict(lat=float(depot["Latitude"]), lon=float(depot["Longitude"]))),
    margin=dict(l=0, r=0, t=0, b=0),
    showlegend=False,
)

fig.add_trace(go.Scattermapbox(
    lat=[float(depot["Latitude"])],
    lon=[float(depot["Longitude"])],
    mode="markers",
    marker=dict(size=18),
    hovertext=[f"DEPOT: {depot['Customer_Name']}"],
    hoverinfo="text",
    name="Depot",
))

left, right = st.columns([2, 1], vertical_alignment="top")

with left:
    st.subheader("Map (India basemap, Maharashtra focus)")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Filtered list")
    show_cols = [
        "Customer_Code", "Customer_Name", "City", "Pin_Code",
        "Generation_MT", "Monthly_Generation_MT", "Weekly_Generation_MT", "Geocode_Status"
    ]
    st.dataframe(
        f[show_cols].sort_values("Generation_MT", ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=560
    )
    csv = f[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", data=csv, file_name="filtered_workshops.csv", mime="text/csv")

st.info("Next layer: selection + route optimization + 5 MT capacity logic.")
