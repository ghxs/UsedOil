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

    # numeric
    for c in ["Generation_MT", "Monthly_Generation_MT", "Weekly_Generation_MT", "Latitude", "Longitude"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # strings
    for c in ["Customer_Code", "Customer_Name", "City", "Pin_Code", "Geocode_Status"]:
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
    """
    Deterministic jitter in meters (visual-only). Adds Lat_plot/Lon_plot.
    """
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


def scale_sizes(values: pd.Series, min_size=12, max_size=30) -> np.ndarray:
    """
    Scale marker sizes by weekly volume with a minimum size.
    """
    v = pd.to_numeric(values, errors="coerce").fillna(0).astype(float)
    vmin, vmax = float(v.min()), float(v.max())
    if vmax <= vmin + 1e-9:
        return np.full(len(v), (min_size + max_size) / 2.0)
    x = (v - vmin) / (vmax - vmin)
    return (min_size + x * (max_size - min_size)).to_numpy()


def build_map(depot_row, points_df_plot, route_df_plot=None):
    # Maharashtra focus
    center_lat = float(depot_row["Latitude"]) - 0.4
    center_lon = float(depot_row["Longitude"]) + 0.1

    fig = go.Figure()

    # Workshops
    if len(points_df_plot) > 0:
        sizes = scale_sizes(points_df_plot["Weekly_Generation_MT"], min_size=12, max_size=30)

        customdata = np.stack([
            points_df_plot["City"].astype(str),
            points_df_plot["Pin_Code"].astype(str),
            points_df_plot["Generation_MT"].astype(float),
            points_df_plot["Monthly_Generation_MT"].astype(float),
            points_df_plot["Weekly_Generation_MT"].astype(float),
        ], axis=1)

        fig.add_trace(go.Scattermapbox(
            lat=points_df_plot["Lat_plot"],
            lon=points_df_plot["Lon_plot"],
            mode="markers+text",
            marker=dict(size=sizes, color="blue", opacity=0.95),
            text=["📍"] * len(points_df_plot),
            textposition="top center",
            textfont=dict(size=14, color="blue"),
            name="Workshops",
            hovertext=points_df_plot["Customer_Name"],
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
        lat=[float(depot_row["Latitude"])],
        lon=[float(depot_row["Longitude"])],
        mode="markers+text",
        marker=dict(size=26, color="red", opacity=1),
        text=["📌"],
        textposition="top center",
        textfont=dict(size=16, color="red"),
        name="Depot",
        hovertext=[depot_row["Customer_Name"]],
        hovertemplate="<b>Depot: %{hovertext}</b><extra></extra>",
    ))

    # Route line
    if route_df_plot is not None and len(route_df_plot) >= 2:
        fig.add_trace(go.Scattermapbox(
            lat=route_df_plot["Lat_plot"],
            lon=route_df_plot["Lon_plot"],
            mode="lines",
            line=dict(width=4, color="black"),
            name="Optimized route",
        ))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=6.8,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=800,
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01),
    )
    return fig


# ---------------- Load data ----------------
df = load_excel(uploaded)

# Identify depot row
depot_mask = df["Customer_Code"].astype(str).str.contains("DEPOT_ABC", na=False)
if not depot_mask.any():
    st.error("Depot row not found. Expecting Customer_Code containing 'DEPOT_ABC'.")
    st.stop()

depot = df[depot_mask].iloc[0]
sites = df[~depot_mask].copy()

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")

annual_min = st.sidebar.number_input("Annual ≥ (MT)", min_value=0.0, value=0.0, step=0.5)
monthly_min = st.sidebar.number_input("Monthly ≥ (MT)", min_value=0.0, value=0.0, step=0.1)
weekly_min  = st.sidebar.number_input("Weekly ≥ (MT)", min_value=0.0, value=0.0, step=0.05)

top_n = st.sidebar.number_input("Top N by Annual (0 = all)", min_value=0, max_value=1000, value=0, step=5)

st.sidebar.header("Map display")
use_jitter = st.sidebar.checkbox("Jitter overlapping pins", value=True)
jitter_m = st.sidebar.slider("Jitter meters", 0, 800, 140, 20) if use_jitter else 0

# Apply numeric filters first (as before)
filtered = sites.copy()
filtered = filtered[
    (filtered["Generation_MT"] >= annual_min) &
    (filtered["Monthly_Generation_MT"] >= monthly_min) &
    (filtered["Weekly_Generation_MT"] >= weekly_min)
].copy()

if top_n and top_n > 0:
    filtered = filtered.sort_values("Generation_MT", ascending=False).head(int(top_n)).copy()

# Excel-style select-by-name in sidebar (all selected by default)
st.sidebar.subheader("Workshops (tick/untick)")

options = list(zip(filtered["Customer_Code"].astype(str), filtered["Customer_Name"].astype(str)))

if "name_select_defaulted" not in st.session_state:
    st.session_state.name_select_defaulted = False

# If first time OR filter list changed drastically, default to "all selected"
# (simple rule: if any selected code not in current options -> reset to all)
option_codes = [c for c, _ in options]
if "selected_codes" not in st.session_state or any(c not in option_codes for c in st.session_state.selected_codes):
    st.session_state.selected_codes = option_codes

selected_options = st.sidebar.multiselect(
    "Select workshops (type to search)",
    options=options,
    default=[opt for opt in options if opt[0] in st.session_state.selected_codes],
    format_func=lambda x: f"{x[1]} ({x[0]})"
)

selected_codes = [c for c, _ in selected_options]
st.session_state.selected_codes = selected_codes

# Apply manual selection (this replaces the old name search)
f = filtered[filtered["Customer_Code"].astype(str).isin(set(selected_codes))].copy()

# ---------------- KPIs (as before) ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Workshops (selected)", len(f))
c2.metric("Annual total (MT)", f"{f['Generation_MT'].sum():.2f}")
c3.metric("Monthly total (MT)", f"{f['Monthly_Generation_MT'].sum():.2f}")
c4.metric("Weekly total (MT)", f"{f['Weekly_Generation_MT'].sum():.2f}")

# ---------------- Route state ----------------
if "route_result" not in st.session_state:
    st.session_state.route_result = None

btn1, btn2 = st.columns([1, 6])
with btn1:
    run_route = st.button("Optimize route")
with btn2:
    clear_route = st.button("Clear route")

if clear_route:
    st.session_state.route_result = None

route_df_plot = None
route_km = None
route_order_df = None

if run_route:
    if len(f) < 1:
        st.warning("Select at least 1 workshop in the left panel.")
    else:
        ordered = pd.concat([pd.DataFrame([depot]), f], ignore_index=True).reset_index(drop=True)
        coords = list(zip(ordered["Latitude"].astype(float), ordered["Longitude"].astype(float)))

        route_nodes, objective_m, dummy_end = solve_open_route_from_depot(coords, seconds=2)

        if route_nodes is None:
            st.error("Route solve failed. Try selecting fewer points.")
        else:
            route_nodes = [n for n in route_nodes if n != dummy_end]
            route_order_df = ordered.iloc[route_nodes].reset_index(drop=True)
            route_km = objective_m / 1000.0

            # Plot line using jittered coords for workshops
            f_plot = apply_jitter(f, jitter_m=float(jitter_m))
            lut = dict(zip(f_plot["Customer_Code"].astype(str), zip(f_plot["Lat_plot"], f_plot["Lon_plot"])))

            route_lat_plot, route_lon_plot = [], []
            for _, row in route_order_df.iterrows():
                code = str(row["Customer_Code"])
                if "DEPOT_ABC" in code:
                    route_lat_plot.append(float(depot["Latitude"]))
                    route_lon_plot.append(float(depot["Longitude"]))
                else:
                    latlon = lut.get(code, (float(row["Latitude"]), float(row["Longitude"])))
                    route_lat_plot.append(latlon[0])
                    route_lon_plot.append(latlon[1])

            route_df_plot = pd.DataFrame({"Lat_plot": route_lat_plot, "Lon_plot": route_lon_plot})

            st.session_state.route_result = {
                "route_km": route_km,
                "route_order_df": route_order_df,
                "route_df_plot": route_df_plot,
            }

# reuse saved route if available
if st.session_state.route_result is not None and route_df_plot is None:
    route_km = st.session_state.route_result["route_km"]
    route_order_df = st.session_state.route_result["route_order_df"]
    route_df_plot = st.session_state.route_result["route_df_plot"]

# ---------------- Map (center) ----------------
f_plot = apply_jitter(f, jitter_m=float(jitter_m))
fig = build_map(depot, f_plot, route_df_plot=route_df_plot)
st.subheader("Map")
st.plotly_chart(fig, use_container_width=True)

# ---------------- Route outputs (below map, as before) ----------------
if route_km is not None:
    st.success(f"One-way distance (approx, straight-line): {route_km:.1f} km")
    st.subheader("Route order")
    st.dataframe(
        route_order_df[["Customer_Name", "City", "Pin_Code", "Weekly_Generation_MT", "Monthly_Generation_MT", "Generation_MT"]],
        use_container_width=True,
        hide_index=True
    )

# ---------------- Table + Download (below) ----------------
st.subheader("Selected workshops")
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
st.download_button("Download selected workshops CSV", data=csv, file_name="selected_workshops.csv", mime="text/csv")
