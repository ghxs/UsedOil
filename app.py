import math
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# ---------------- Brand colors ----------------
ORANGE = "#F26522"
DARK_BLUE = "#0A2365"
BLUE = "#0085C8"

st.set_page_config(page_title="Used Oil Pilot Dashboard – Maharashtra", layout="wide")

# ---------------- Global UI styling ----------------
st.markdown(
    f"""
    <style>
      .stApp {{ background: #ffffff; }}

      /* Larger base font */
      html, body, [class*="css"] {{ font-size: 17px !important; }}

      .title {{
        font-size: 34px;
        font-weight: 800;
        color: {DARK_BLUE};
        margin-bottom: 0.2rem;
      }}
      .subtitle {{
        font-size: 16px;
        color: #334155;
        margin-top: 0;
        margin-bottom: 1rem;
      }}

      /* KPI cards */
      .kpi-wrap {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin: 6px 0 16px 0;
      }}
      .kpi {{
        border: 1px solid rgba(10,35,101,0.12);
        border-left: 6px solid {BLUE};
        border-radius: 14px;
        padding: 14px 14px;
        background: #ffffff;
        box-shadow: 0 8px 22px rgba(10,35,101,0.06);
      }}
      .kpi.orange {{ border-left-color: {ORANGE}; }}
      .kpi.dark {{ border-left-color: {DARK_BLUE}; }}
      .kpi .label {{
        font-size: 14px;
        color: #475569;
        margin-bottom: 6px;
        font-weight: 600;
      }}
      .kpi .value {{
        font-size: 26px;
        color: #0f172a;
        font-weight: 800;
        line-height: 1.1;
      }}

      /* Buttons */
      .stButton > button {{
        background: {ORANGE};
        color: white;
        border: none;
        padding: 0.65rem 1.0rem;
        border-radius: 10px;
        font-weight: 800;
        font-size: 16px;
      }}
      .stButton > button:hover {{
        background: #d9551e;
        color: white;
      }}

      /* Sidebar headings */
      section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{
        color: {DARK_BLUE};
      }}

      /* Dataframe header */
      thead tr th {{ font-weight: 700 !important; }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Header ----------------
st.markdown('<div class="title">Used Oil Collection Pilot — Maharashtra</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Filter workshops by volume, visualize on map, and generate a one-way milk-run route from the depot.</div>',
    unsafe_allow_html=True
)

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
    """
    One-way milk run:
    Start at depot (index 0), visit all points, end anywhere.
    Implement with dummy end node with zero inbound cost.
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
    route_nodes.append(manager.IndexToNode(idx))  # dummy end

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


def scale_sizes(values, min_size=12, max_size=30):
    v = pd.to_numeric(values, errors="coerce").fillna(0).astype(float)
    vmin, vmax = float(v.min()), float(v.max())
    if vmax <= vmin + 1e-9:
        return np.full(len(v), (min_size + max_size) / 2.0)
    x = (v - vmin) / (vmax - vmin)
    return (min_size + x * (max_size - min_size)).to_numpy()


def kpi_cards(workshops_count, weekly_mt, monthly_mt, annual_mt):
    st.markdown(
        f"""
        <div class="kpi-wrap">
          <div class="kpi dark">
            <div class="label">Workshops (filtered)</div>
            <div class="value">{workshops_count}</div>
          </div>
          <div class="kpi">
            <div class="label">Weekly total (MT)</div>
            <div class="value">{weekly_mt:,.2f}</div>
          </div>
          <div class="kpi orange">
            <div class="label">Monthly total (MT)</div>
            <div class="value">{monthly_mt:,.2f}</div>
          </div>
          <div class="kpi">
            <div class="label">Annual total (MT)</div>
            <div class="value">{annual_mt:,.2f}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------- Load and filter ----------------
df = load_excel(uploaded)

depot_mask = df["Customer_Code"].astype(str).str.contains("DEPOT_ABC", na=False)
if not depot_mask.any():
    st.error("Depot row not found (Customer_Code must contain 'DEPOT_ABC').")
    st.stop()

depot = df[depot_mask].iloc[0]
sites = df[~depot_mask].copy()

st.sidebar.header("Filters")
name_q = st.sidebar.text_input("Search by name", "")

annual_min = st.sidebar.number_input("Annual ≥ (MT)", 0.0, value=0.0)
monthly_min = st.sidebar.number_input("Monthly ≥ (MT)", 0.0, value=0.0)
weekly_min = st.sidebar.number_input("Weekly ≥ (MT)", 0.0, value=0.0)

top_n = st.sidebar.number_input("Top N by Annual (0 = all)", 0, value=0, step=5)

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

weekly_total = float(f["Weekly_Generation_MT"].sum())
monthly_total = float(f["Monthly_Generation_MT"].sum())
annual_total = float(f["Generation_MT"].sum())

kpi_cards(len(f), weekly_total, monthly_total, annual_total)

# Plot data (with jitter for visualization)
f_plot = apply_jitter(f, jitter_m)
sizes = scale_sizes(f_plot["Weekly_Generation_MT"], min_size=12, max_size=30)

# Lookup for jittered plotting coordinates (for route line)
plot_lut = dict(zip(
    f_plot["Customer_Code"].astype(str),
    zip(f_plot["Lat_plot"].astype(float), f_plot["Lon_plot"].astype(float))
))

# ---------------- Build base map ----------------
fig = go.Figure()

# Workshops pins
if len(f_plot) > 0:
    fig.add_trace(go.Scattermapbox(
        lat=f_plot["Lat_plot"],
        lon=f_plot["Lon_plot"],
        mode="markers+text",
        marker=dict(size=sizes, color=BLUE, opacity=0.95),
        text=["📍"] * len(f_plot),
        textposition="top center",
        textfont=dict(size=15, color=BLUE),
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
            "Annual: %{customdata[2]:.2f} MT<br>"
            "Monthly: %{customdata[3]:.2f} MT<br>"
            "Weekly: %{customdata[4]:.2f} MT<br>"
            "<extra></extra>"
        ),
        name="Workshops"
    ))

# Depot pin
fig.add_trace(go.Scattermapbox(
    lat=[float(depot["Latitude"])],
    lon=[float(depot["Longitude"])],
    mode="markers+text",
    marker=dict(size=30, color=ORANGE, opacity=1),
    text=["📌"],
    textposition="top center",
    textfont=dict(size=17, color=ORANGE),
    hovertext=[depot["Customer_Name"]],
    hovertemplate="<b>Depot: %{hovertext}</b><extra></extra>",
    name="Depot"
))

fig.update_layout(
    mapbox=dict(
        style="open-street-map",
        center=dict(lat=float(depot["Latitude"]) - 0.4, lon=float(depot["Longitude"]) + 0.1),
        zoom=6.8
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    height=620  # ✅ smaller map height
)

# ---------------- Route optimization + draw line ----------------
c1, c2 = st.columns([1, 3], vertical_alignment="center")
with c1:
    run_route = st.button("Optimize route (draw line)")

route_df = None
route_km = None

if run_route:
    if len(f) < 1:
        st.warning("No workshops selected by filters.")
    elif len(f) == 1:
        st.info("Only 1 workshop selected. No route line needed.")
    else:
        ordered = pd.concat([pd.DataFrame([depot]), f], ignore_index=True).reset_index(drop=True)
        coords = list(zip(ordered["Latitude"].astype(float), ordered["Longitude"].astype(float)))

        route_nodes, obj_m, dummy_end = solve_open_route_from_depot(coords, seconds=2)

        if route_nodes is None:
            st.error("Route solve failed. Try reducing Top N.")
        else:
            route_nodes = [n for n in route_nodes if n != dummy_end]
            route_df = ordered.iloc[route_nodes].reset_index(drop=True)
            route_km = obj_m / 1000.0

            line_lat, line_lon = [], []
            for _, r in route_df.iterrows():
                code = str(r["Customer_Code"])
                if "DEPOT_ABC" in code:
                    line_lat.append(float(depot["Latitude"]))
                    line_lon.append(float(depot["Longitude"]))
                else:
                    latlon = plot_lut.get(code, (float(r["Latitude"]), float(r["Longitude"])))
                    line_lat.append(latlon[0])
                    line_lon.append(latlon[1])

            fig.add_trace(go.Scattermapbox(
                lat=line_lat,
                lon=line_lon,
                mode="lines",
                line=dict(width=5, color=DARK_BLUE),
                name="Optimized route"
            ))

# ✅ framed map container (presentation look)
st.markdown(
    """
    <div style="
        border:1px solid rgba(10,35,101,0.12);
        border-radius:14px;
        padding:6px;
        box-shadow:0 6px 18px rgba(10,35,101,0.06);
        margin-bottom:8px;
    ">
    """,
    unsafe_allow_html=True
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

if route_km is not None:
    st.success(f"One-way distance (approx): {route_km:.1f} km")

    st.subheader("Route order")
    st.dataframe(
        route_df[["Customer_Name", "City", "Pin_Code",
                  "Weekly_Generation_MT", "Monthly_Generation_MT", "Generation_MT"]]
        .reset_index(drop=True),
        use_container_width=True,
        height=320
    )

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
