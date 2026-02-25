import math
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

    # numeric
    for c in ["Generation_MT", "Monthly_Generation_MT", "Weekly_Generation_MT", "Latitude", "Longitude"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # strings
    for c in ["Customer_Code", "Customer_Name", "Address", "City", "Pin_Code", "Geocode_Status"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    # keep only geocoded
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
    Open path: start at depot (index 0), visit all points, END ANYWHERE (one-way milk run).
    Trick: add a dummy end node with zero cost from any node -> end at dummy.
    """
    # coords: depot + sites
    n = len(coords)
    dummy_end = n  # new node index
    coords2 = coords + [coords[0]]  # dummy coords (same as depot; not used)
    dist = build_distance_matrix(coords2)

    BIG = 10**9

    # Set distance TO dummy end as 0 from any node (lets route end there freely)
    for i in range(n + 1):
        dist[i][dummy_end] = 0

    # Prevent starting from dummy or going out of dummy meaningfully
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

    # Extract route
    route_nodes = []
    idx = routing.Start(0)
    while not routing.IsEnd(idx):
        route_nodes.append(manager.IndexToNode(idx))
        idx = sol.Value(routing.NextVar(idx))
    route_nodes.append(manager.IndexToNode(idx))  # dummy_end

    # Objective is in "meters" because we built it that way
    objective_m = sol.ObjectiveValue()

    return route_nodes, objective_m, dummy_end


df = load_excel(uploaded)

# Identify depot
depot_mask = df["Customer_Code"].astype(str).str.contains("DEPOT_ABC", na=False)
if not depot_mask.any():
    st.error("Depot row not found. Expecting Customer_Code containing 'DEPOT_ABC'.")
    st.stop()

depot = df[depot_mask].iloc[0]
sites = df[~depot_mask].copy()

# Sidebar filters
st.sidebar.header("Filters")

name_q = st.sidebar.text_input("Search by name", value="", placeholder="type workshop name")

annual_min = st.sidebar.number_input("Annual ≥ (MT)", min_value=0.0, value=0.0, step=0.5)
monthly_min = st.sidebar.number_input("Monthly ≥ (MT)", min_value=0.0, value=0.0, step=0.1)
weekly_min = st.sidebar.number_input("Weekly ≥ (MT)", min_value=0.0, value=0.0, step=0.05)

top_n = st.sidebar.number_input("Top N by Annual MT (0 = all)", min_value=0, max_value=1000, value=0, step=5)

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

# ---------- Map (Pins) ----------
# Plotly scattermapbox supports pin-like symbols via marker.symbol="marker"
# We'll use go.Scattermapbox for full control and hovertemplate.

center_lat = float(depot["Latitude"])
center_lon = float(depot["Longitude"])

fig = go.Figure()

# Workshop pins (blue)
if len(f) > 0:
    customdata = np.stack([
        f["Address"].astype(str),
        f["Generation_MT"].astype(float),
        f["Monthly_Generation_MT"].astype(float),
        f["Weekly_Generation_MT"].astype(float),
    ], axis=1)

    fig.add_trace(go.Scattermapbox(
        lat=f["Latitude"],
        lon=f["Longitude"],
        mode="markers",
        marker=go.scattermapbox.Marker(
            size=14,
            color="blue",
            symbol="marker"  # pin-like
        ),
        name="Workshops",
        customdata=customdata,
        hovertemplate=(
            "<b>%{text}</b><br>"
            "%{customdata[0]}<br>"
            "Annual: %{customdata[1]:.2f} MT<br>"
            "Monthly: %{customdata[2]:.2f} MT<br>"
            "Weekly: %{customdata[3]:.2f} MT<br>"
            "<extra></extra>"
        ),
        text=f["Customer_Name"]
    ))

# Depot pin (red)
fig.add_trace(go.Scattermapbox(
    lat=[center_lat],
    lon=[center_lon],
    mode="markers",
    marker=go.scattermapbox.Marker(
        size=18,
        color="red",
        symbol="marker"
    ),
    name="Depot (ABC Petrochem)",
    hovertemplate=(
        "<b>Depot: %{text}</b><br>"
        "%{customdata}<br>"
        "<extra></extra>"
    ),
    text=[depot["Customer_Name"]],
    customdata=[depot["Address"] if "Address" in depot else ""]
))

# Layout
fig.update_layout(
    mapbox=dict(
        style="open-street-map",
        center=dict(lat=center_lat, lon=center_lon),
        zoom=6
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    height=650,
    legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01)
)

# ---------- Route Optimization ----------
st.divider()
st.subheader("Route optimization (one-way milk run from depot)")
st.caption("Currently uses straight-line distance (Haversine). Next upgrade can be road distance using Google Distance Matrix API.")

route_col1, route_col2 = st.columns([1, 2], vertical_alignment="top")

with route_col1:
    run_route = st.button("Optimize route for filtered workshops")

route_info = None
route_df = None
route_km = None

if run_route:
    if len(f) < 1:
        st.warning("No workshops in the filtered list. Adjust filters first.")
    else:
        ordered = pd.concat([pd.DataFrame([depot]), f], ignore_index=True).reset_index(drop=True)
        coords = list(zip(ordered["Latitude"].astype(float), ordered["Longitude"].astype(float)))

        route_nodes, objective_m, dummy_end = solve_open_route_from_depot(coords, seconds=2)

        if route_nodes is None:
            st.error("Could not compute a route. Try reducing Top N or widening filters.")
        else:
            # Remove dummy end node
            route_nodes_no_dummy = [n for n in route_nodes if n != dummy_end]

            route_df = ordered.iloc[route_nodes_no_dummy].reset_index(drop=True)
            route_km = objective_m / 1000.0  # already one-way because dummy end costs 0

            route_info = {
                "one_way_km": route_km,
                "stops": len(route_df),
            }

            # Add route line to map
            fig.add_trace(go.Scattermapbox(
                lat=route_df["Latitude"],
                lon=route_df["Longitude"],
                mode="lines",
                line=dict(width=4, color="black"),
                name="Optimized route"
            ))

with route_col2:
    st.subheader("Map (pins + hover details)")
    st.plotly_chart(fig, use_container_width=True)

# Show route outputs if available
if route_info:
    st.success(f"One-way distance (approx): {route_info['one_way_km']:.1f} km | Stops (incl. depot start): {route_info['stops']}")

    st.subheader("Route order")
    st.dataframe(
        route_df[["Customer_Name", "Address", "City", "Pin_Code", "Weekly_Generation_MT", "Monthly_Generation_MT", "Generation_MT"]]
        .reset_index(drop=True),
        use_container_width=True
    )

# Table + download
st.subheader("Filtered list")
show_cols = [
    "Customer_Code", "Customer_Name", "Address", "City", "Pin_Code",
    "Generation_MT", "Monthly_Generation_MT", "Weekly_Generation_MT", "Geocode_Status"
]
st.dataframe(
    f[show_cols].sort_values("Generation_MT", ascending=False).reset_index(drop=True),
    use_container_width=True,
    height=420
)

csv = f[show_cols].to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", data=csv, file_name="filtered_workshops.csv", mime="text/csv")
