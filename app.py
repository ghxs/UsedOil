import math
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

st.set_page_config(page_title="Used Oil Pilot Dashboard – Maharashtra", layout="wide")
st.title("Used Oil Collection Pilot – Maharashtra")

uploaded = st.sidebar.file_uploader("Upload MH geocoded Excel", type=["xlsx"])
if uploaded is None:
    st.info("Upload MH_geocoded.xlsx from the sidebar to start.")
    st.stop()


@st.cache_data
def load_excel(file) -> pd.DataFrame:
    df = pd.read_excel(file)

    for c in ["Generation_MT", "Monthly_Generation_MT", "Weekly_Generation_MT", "Latitude", "Longitude"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["Customer_Code", "Customer_Name", "City", "Pin_Code", "Geocode_Status"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    df = df[df["Latitude"].notna() & df["Longitude"].notna()].copy()
    return df


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def build_distance_matrix(coords):
    n = len(coords)
    mat = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 0
            else:
                km = haversine_km(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
                mat[i, j] = int(km * 1000)
    return mat


def solve_open_route_from_depot(coords, seconds=2):
    n = len(coords)
    dummy_end = n
    coords2 = coords + [coords[0]]
    dist = build_distance_matrix(coords2)
    BIG = 10**9

    for i in range(n + 1):
        dist[i][dummy_end] = 0
    for j in range(n + 1):
        dist[dummy_end][j] = BIG
    dist[dummy_end][dummy_end] = 0

    manager = pywrapcp.RoutingIndexManager(n + 1, 1, [0], [dummy_end])
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(from_index, to_index):
        f = manager.IndexToNode(from_index)
        t = manager.IndexToNode(to_index)
        return int(dist[f][t])

    transit = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(int(seconds))

    sol = routing.SolveWithParameters(params)
    if not sol:
        return None, None, None

    route_nodes = []
    idx = routing.Start(0)
    while not routing.IsEnd(idx):
        route_nodes.append(manager.IndexToNode(idx))
        idx = sol.Value(routing.NextVar(idx))
    route_nodes.append(manager.IndexToNode(idx))

    return route_nodes, sol.ObjectiveValue(), dummy_end


def _stable_u01(seed_text: str) -> float:
    h = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    return (int(h[:16], 16) % 10**12) / 10**12


def apply_jitter(df_points: pd.DataFrame, jitter_m=0.0) -> pd.DataFrame:
    out = df_points.copy()
    if jitter_m <= 0:
        out["Lat_plot"] = out["Latitude"].astype(float)
        out["Lon_plot"] = out["Longitude"].astype(float)
        return out

    lat_plot, lon_plot = [], []
    for _, row in out.iterrows():
        code = str(row["Customer_Code"])
        u1 = _stable_u01(code + "|u1")
        u2 = _stable_u01(code + "|u2")
        angle = 2 * math.pi * u1
        radius = jitter_m * (0.25 + 0.75 * u2)

        lat0 = float(row["Latitude"])
        lon0 = float(row["Longitude"])

        dlat = (radius * math.cos(angle)) / 111_320.0
        denom = 111_320.0 * max(math.cos(math.radians(lat0)), 0.2)
        dlon = (radius * math.sin(angle)) / denom

        lat_plot.append(lat0 + dlat)
        lon_plot.append(lon0 + dlon)

    out["Lat_plot"] = lat_plot
    out["Lon_plot"] = lon_plot
    return out


def scale_sizes(values, min_size=12, max_size=28):
    v = pd.to_numeric(values, errors="coerce").fillna(0).astype(float)
    vmin, vmax = float(v.min()), float(v.max())
    if vmax <= vmin + 1e-9:
        return np.full(len(v), (min_size + max_size) / 2.0)
    x = (v - vmin) / (vmax - vmin)
    return min_size + x * (max_size - min_size)


df = load_excel(uploaded)

depot_mask = df["Customer_Code"].astype(str).str.contains("DEPOT_ABC", na=False)
if not depot_mask.any():
    st.error("Depot row not found.")
    st.stop()

depot = df[depot_mask].iloc[0]
sites = df[~depot_mask].copy()

# Sidebar filters
st.sidebar.header("Filters")
name_q = st.sidebar.text_input("Search by name", "")

annual_min = st.sidebar.number_input("Annual ≥", 0.0, value=0.0)
monthly_min = st.sidebar.number_input("Monthly ≥", 0.0, value=0.0)
weekly_min = st.sidebar.number_input("Weekly ≥", 0.0, value=0.0)

top_n = st.sidebar.number_input("Top N (0 = all)", 0, value=0, step=5)

st.sidebar.header("Map display")
use_jitter = st.sidebar.checkbox("Jitter overlapping pins", True)
jitter_m = st.sidebar.slider("Jitter meters", 0, 800, 140) if use_jitter else 0

# Apply filters
f = sites.copy()
if name_q:
    f = f[f["Customer_Name"].str.lower().str.contains(name_q.lower())]

f = f[
    (f["Generation_MT"] >= annual_min) &
    (f["Monthly_Generation_MT"] >= monthly_min) &
    (f["Weekly_Generation_MT"] >= weekly_min)
]

if top_n > 0:
    f = f.sort_values("Generation_MT", ascending=False).head(int(top_n))

# KPIs
k1, k2, k3 = st.columns(3)
k1.metric("Workshops", len(f))
k2.metric("Weekly MT", round(f["Weekly_Generation_MT"].sum(), 2))
k3.metric("Annual MT", round(f["Generation_MT"].sum(), 2))

f_plot = apply_jitter(f, jitter_m)
sizes = scale_sizes(f_plot["Weekly_Generation_MT"])

# Map
fig = go.Figure()

fig.add_trace(go.Scattermapbox(
    lat=f_plot["Lat_plot"],
    lon=f_plot["Lon_plot"],
    mode="markers+text",
    marker=dict(size=sizes, color="blue"),
    text=["📍"] * len(f_plot),
    textposition="top center",
    textfont=dict(size=14, color="blue"),
    hovertext=f_plot["Customer_Name"],
    customdata=np.stack([
        f_plot["City"],
        f_plot["Pin_Code"],
        f_plot["Generation_MT"],
        f_plot["Monthly_Generation_MT"],
        f_plot["Weekly_Generation_MT"]
    ], axis=1),
    hovertemplate=(
        "<b>%{hovertext}</b><br>"
        "City: %{customdata[0]}<br>"
        "Pin: %{customdata[1]}<br>"
        "Annual: %{customdata[2]:.2f}<br>"
        "Monthly: %{customdata[3]:.2f}<br>"
        "Weekly: %{customdata[4]:.2f}<extra></extra>"
    ),
))

fig.add_trace(go.Scattermapbox(
    lat=[depot["Latitude"]],
    lon=[depot["Longitude"]],
    mode="markers+text",
    marker=dict(size=26, color="red"),
    text=["📌"],
    textposition="top center",
    textfont=dict(size=16, color="red"),
    hovertext=[depot["Customer_Name"]],
    hovertemplate="<b>Depot: %{hovertext}</b><extra></extra>",
))

fig.update_layout(
    mapbox=dict(
        style="open-street-map",
        center=dict(lat=float(depot["Latitude"]) - 0.4, lon=float(depot["Longitude"]) + 0.1),
        zoom=6.8
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    height=780
)

st.plotly_chart(fig, use_container_width=True)

# Route button
if st.button("Optimize route"):
    ordered = pd.concat([pd.DataFrame([depot]), f], ignore_index=True)
    coords = list(zip(ordered["Latitude"], ordered["Longitude"]))
    route_nodes, obj, dummy = solve_open_route_from_depot(coords)
    if route_nodes:
        st.success(f"One-way distance ≈ {obj/1000:.1f} km")
