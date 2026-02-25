import math
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from sklearn.cluster import DBSCAN, KMeans

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
    mat = np.zeros((n, n), dtype=int)  # meters
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 0
            else:
                km = haversine_km(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
                mat[i, j] = int(km * 1000)
    return mat


def solve_open_route_from_depot(coords, seconds=2):
    """
    Open path: start at depot (index 0), visit all points, end anywhere (one-way milk run).
    Implemented by adding a dummy end node with zero inbound cost.
    """
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


def scale_sizes(values: pd.Series, min_size=10, max_size=28) -> np.ndarray:
    v = pd.to_numeric(values, errors="coerce").fillna(0).astype(float)
    vmin, vmax = float(v.min()), float(v.max())
    if vmax <= vmin + 1e-9:
        return np.full(len(v), (min_size + max_size) / 2.0)
    x = (v - vmin) / (vmax - vmin)
    return (min_size + x * (max_size - min_size)).to_numpy()


def dbscan_clusters_haversine(lat, lon, radius_km=100.0, min_samples=2):
    # sklearn DBSCAN expects radians for haversine metric
    coords = np.radians(np.c_[lat.astype(float).to_numpy(), lon.astype(float).to_numpy()])
    eps = float(radius_km) / 6371.0  # km -> radians
    model = DBSCAN(eps=eps, min_samples=int(min_samples), metric="haversine")
    labels = model.fit_predict(coords)
    return labels


def kmeans_geo(lat, lon, k=2, random_state=42):
    X = np.c_[lat.astype(float).to_numpy(), lon.astype(float).to_numpy()]
    k = max(1, int(k))
    if len(X) <= k:
        # each point its own cluster (degenerate)
        return np.arange(len(X))
    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    return km.fit_predict(X)


df = load_excel(uploaded)

# Depot + sites
depot_mask = df["Customer_Code"].astype(str).str.contains("DEPOT_ABC", na=False)
if not depot_mask.any():
    st.error("Depot row not found. Expecting Customer_Code containing 'DEPOT_ABC'.")
    st.stop()

depot = df[depot_mask].iloc[0]
sites = df[~depot_mask].copy()

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")
name_q = st.sidebar.text_input("Search by name", value="", placeholder="type workshop name")

annual_min = st.sidebar.number_input("Annual ≥ (MT)", min_value=0.0, value=0.0, step=0.5)
monthly_min = st.sidebar.number_input("Monthly ≥ (MT)", min_value=0.0, value=0.0, step=0.1)
weekly_min  = st.sidebar.number_input("Weekly ≥ (MT)", min_value=0.0, value=0.0, step=0.05)

top_n = st.sidebar.number_input("Top N by Annual MT (0 = all)", min_value=0, max_value=1000, value=0, step=5)

st.sidebar.header("Map display")
use_jitter = st.sidebar.checkbox("Jitter overlapping pins", value=True)
jitter_m = st.sidebar.slider("Jitter strength (meters)", 0, 800, 140, 20) if use_jitter else 0

# Visual clustering
st.sidebar.subheader("Visual clustering")
use_visual_cluster = st.sidebar.checkbox("Enable clustering on map", value=True)
cluster_radius_km = st.sidebar.slider("Cluster radius (km)", 10, 300, 100, 10)
cluster_min_samples = st.sidebar.slider("Min points per cluster", 2, 10, 2, 1)
show_individual_points = st.sidebar.checkbox("Show individual pins also", value=False)

# Operational clustering (routes in parallel)
st.sidebar.subheader("Operational clustering")
routes_parallel = st.sidebar.number_input("Routes in parallel", min_value=1, max_value=10, value=2, step=1)

# Apply filters
f = sites.copy()
if name_q.strip():
    q = name_q.strip().lower()
    f = f[f["Customer_Name"].str.lower().str.contains(q)]

f = f[
    (f["Generation_MT"] >= annual_min) &
    (f["Monthly_Generation_MT"] >= monthly_min) &
    (f["Weekly_Generation_MT"] >= weekly_min)
]

if top_n and top_n > 0:
    f = f.sort_values("Generation_MT", ascending=False).head(int(top_n))

# KPIs
total_annual = float(f["Generation_MT"].sum())
total_monthly = float(f["Monthly_Generation_MT"].sum())
total_weekly = float(f["Weekly_Generation_MT"].sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Workshops (filtered)", f"{len(f)}")
k2.metric("Annual total (MT)", f"{total_annual:,.2f}")
k3.metric("Monthly total (MT)", f"{total_monthly:,.2f}")
k4.metric("Weekly total (MT)", f"{total_weekly:,.2f}")

# Plot coords with jitter (visual only)
f_plot = apply_jitter(f, jitter_m=float(jitter_m))
pin_sizes = scale_sizes(f_plot["Weekly_Generation_MT"], min_size=12, max_size=28)  # min size preserved

# Center / zoom to Maharashtra (slightly adjusted)
center_lat = float(depot["Latitude"]) - 0.4
center_lon = float(depot["Longitude"]) + 0.1
zoom_level = 6.8

# ---------------- Map Figure ----------------
fig = go.Figure()

# Visual clusters layer
if use_visual_cluster and len(f_plot) > 0:
    labels = dbscan_clusters_haversine(
        f_plot["Latitude"], f_plot["Longitude"],
        radius_km=float(cluster_radius_km),
        min_samples=int(cluster_min_samples)
    )
    f_plot = f_plot.copy()
    f_plot["Cluster_Label"] = labels

    # Cluster aggregates (labels >= 0)
    clustered = f_plot[f_plot["Cluster_Label"] >= 0].copy()
    if len(clustered) > 0:
        agg = clustered.groupby("Cluster_Label").agg(
            Lat=("Latitude", "mean"),
            Lon=("Longitude", "mean"),
            Count=("Customer_Code", "count"),
            Weekly_MT=("Weekly_Generation_MT", "sum"),
            Monthly_MT=("Monthly_Generation_MT", "sum"),
            Annual_MT=("Generation_MT", "sum"),
        ).reset_index()

        # size cluster markers by count (bounded)
        csize = (10 + np.sqrt(agg["Count"]) * 6).clip(14, 40)

        fig.add_trace(go.Scattermapbox(
            lat=agg["Lat"],
            lon=agg["Lon"],
            mode="markers+text",
            marker=dict(size=csize, color="purple", opacity=0.85),
            text=agg["Count"].astype(int).astype(str),
            textfont=dict(size=14, color="white"),
            textposition="middle center",
            name="Clusters",
            customdata=np.stack([agg["Count"], agg["Weekly_MT"], agg["Monthly_MT"], agg["Annual_MT"]], axis=1),
            hovertemplate=(
                "<b>Cluster</b><br>"
                "Points: %{customdata[0]:.0f}<br>"
                "Weekly total: %{customdata[1]:.2f} MT<br>"
                "Monthly total: %{customdata[2]:.2f} MT<br>"
                "Annual total: %{customdata[3]:.2f} MT<br>"
                "<extra></extra>"
            ),
        ))

# Individual pins layer (optional or if clustering off)
if len(f_plot) > 0 and (not use_visual_cluster or show_individual_points):
    customdata = np.stack([
        f_plot["City"].astype(str),
        f_plot["Pin_Code"].astype(str),
        f_plot["Generation_MT"].astype(float),
        f_plot["Monthly_Generation_MT"].astype(float),
        f_plot["Weekly_Generation_MT"].astype(float),
    ], axis=1)

    fig.add_trace(go.Scattermapbox(
        lat=f_plot["Lat_plot"],
        lon=f_plot["Lon_plot"],
        mode="markers+text",
        marker=dict(size=pin_sizes, color="blue", opacity=0.92),
        text=["📍"] * len(f_plot),
        textposition="top center",
        textfont=dict(size=14, color="blue"),
        name="Workshops",
        hovertext=f_plot["Customer_Name"],
        customdata=customdata,
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "City: %{customdata[0]}<br>"
            "Pincode: %{customdata[1]}<br>"
            "Annual: %{customdata[2]:.2f} MT<br>"
            "Monthly: %{customdata[3]:.2f} MT<br>"
            "Weekly: %{customdata[4]:.2f} MT<br>"
            "<extra></extra>"
        ),
    ))

# Depot
fig.add_trace(go.Scattermapbox(
    lat=[float(depot["Latitude"])],
    lon=[float(depot["Longitude"])],
    mode="markers+text",
    marker=dict(size=26, color="red", opacity=1),
    text=["📌"],
    textposition="top center",
    textfont=dict(size=16, color="red"),
    name="Depot (ABC Petrochem)",
    hovertext=[depot["Customer_Name"]],
    hovertemplate="<b>Depot: %{hovertext}</b><extra></extra>",
))

fig.update_layout(
    mapbox=dict(
        style="open-street-map",
        center=dict(lat=center_lat, lon=center_lon),
        zoom=zoom_level
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    height=780,
    legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01)
)

st.subheader("Map")
st.plotly_chart(fig, use_container_width=True)

# ---------------- Route Optimization ----------------
st.divider()
st.subheader("Route optimization (one-way milk run from depot)")
st.caption("Distance uses straight-line (Haversine). Next upgrade: road distance via Google Distance Matrix API.")

run_route = st.button("Optimize routes for current filtered set")

if run_route:
    if len(f) < 1:
        st.warning("No workshops in the filtered list. Adjust filters first.")
    else:
        # Operational clustering into K parallel routes
        K = int(routes_parallel)
        labels = kmeans_geo(f["Latitude"], f["Longitude"], k=K)
        f2 = f.copy()
        f2["Route_Group"] = labels

        # Per-group solve + draw lines
        # Use plotted coords for nicer visuals if jitter is on
        f2_plot = f_plot.copy()
        lut_plot = dict(zip(f2_plot["Customer_Code"].astype(str), zip(f2_plot["Lat_plot"], f2_plot["Lon_plot"])))

        fig2 = fig  # reuse current map and add route lines

        summary_rows = []
        for g in sorted(f2["Route_Group"].unique()):
            gdf = f2[f2["Route_Group"] == g].copy()
            ordered = pd.concat([pd.DataFrame([depot]), gdf], ignore_index=True).reset_index(drop=True)
            coords = list(zip(ordered["Latitude"].astype(float), ordered["Longitude"].astype(float)))

            route_nodes, objective_m, dummy_end = solve_open_route_from_depot(coords, seconds=2)
            if route_nodes is None:
                summary_rows.append([g, len(gdf), np.nan, gdf["Weekly_Generation_MT"].sum()])
                continue

            route_nodes = [n for n in route_nodes if n != dummy_end]
            route_df = ordered.iloc[route_nodes].reset_index(drop=True)
            km = objective_m / 1000.0

            # Build line coords (prefer jittered for visuals)
            route_lat, route_lon = [], []
            for _, row in route_df.iterrows():
                code = str(row["Customer_Code"])
                if "DEPOT_ABC" in code:
                    route_lat.append(float(depot["Latitude"]))
                    route_lon.append(float(depot["Longitude"]))
                else:
                    latlon = lut_plot.get(code, (float(row["Latitude"]), float(row["Longitude"])))
                    route_lat.append(latlon[0])
                    route_lon.append(latlon[1])

            fig2.add_trace(go.Scattermapbox(
                lat=route_lat,
                lon=route_lon,
                mode="lines",
                line=dict(width=4),  # let plotly pick default colors
                name=f"Route {g+1}"
            ))

            summary_rows.append([g + 1, len(gdf), km, float(gdf["Weekly_Generation_MT"].sum())])

        st.plotly_chart(fig2, use_container_width=True)

        summary = pd.DataFrame(summary_rows, columns=["Route", "Stops", "One-way km (approx)", "Weekly MT (sum)"])
        st.subheader("Route summary")
        st.dataframe(summary, use_container_width=True)

# ---------------- Table + Download ----------------
st.subheader("Filtered list")
show_cols = [
    "Customer_Code", "Customer_Name", "City", "Pin_Code",
    "Generation_MT", "Monthly_Generation_MT", "Weekly_Generation_MT", "Geocode_Status"
]
st.dataframe(
    f[show_cols].sort_values("Generation_MT", ascending=False).reset_index(drop=True),
    use_container_width=True,
    height=420
)

csv = f[show_cols].to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", data=csv, file_name="filtered_workshops.csv", mime="text/csv")
