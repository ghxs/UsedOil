import math
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from streamlit_plotly_events import plotly_events

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
      html, body, [class*="css"] {{ font-size: 19px !important; }}

      .title {{
        font-size: 38px;
        font-weight: 900;
        color: {DARK_BLUE};
        margin-bottom: 0.25rem;
      }}
      .subtitle {{
        font-size: 18px;
        color: #334155;
        margin-top: 0;
        margin-bottom: 1rem;
      }}

      section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p {{
        font-size: 16.5px !important;
      }}

      .kpi-wrap {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin: 6px 0 16px 0;
      }}
      .kpi {{
        border: 1px solid rgba(10,35,101,0.12);
        border-left: 7px solid {BLUE};
        border-radius: 14px;
        padding: 14px 14px;
        background: #ffffff;
        box-shadow: 0 8px 22px rgba(10,35,101,0.06);
      }}
      .kpi.orange {{ border-left-color: {ORANGE}; }}
      .kpi.dark {{ border-left-color: {DARK_BLUE}; }}
      .kpi .label {{
        font-size: 15px;
        color: #475569;
        margin-bottom: 6px;
        font-weight: 700;
      }}
      .kpi .value {{
        font-size: 30px;
        color: #0f172a;
        font-weight: 900;
        line-height: 1.1;
      }}

      .stButton > button {{
        background: {ORANGE};
        color: white;
        border: none;
        padding: 0.70rem 1.05rem;
        border-radius: 10px;
        font-weight: 900;
        font-size: 16.5px;
      }}
      .stButton > button:hover {{
        background: #d9551e;
        color: white;
      }}

      thead tr th {{ font-weight: 900 !important; }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- App title ----------------
st.markdown('<div class="title">Used Oil Collection Pilot — Maharashtra</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload → view all workshops → click Select Cluster → box/lasso select → see totals. One single map only.</div>',
    unsafe_allow_html=True
)

# ---------------- Session state ----------------
if "selected_names" not in st.session_state:
    st.session_state.selected_names = None

if "cluster_mode" not in st.session_state:
    st.session_state.cluster_mode = False

if "map_selected_codes" not in st.session_state:
    st.session_state.map_selected_codes = []

# ---------------- Upload ----------------
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


# ---------------- Load and split depot ----------------
df = load_excel(uploaded)

depot_mask = df["Customer_Code"].astype(str).str.contains("DEPOT_ABC", na=False)
if not depot_mask.any():
    st.error("Depot row not found (Customer_Code must contain 'DEPOT_ABC').")
    st.stop()

depot = df[depot_mask].iloc[0]
sites = df[~depot_mask].copy()

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")

annual_min = st.sidebar.number_input("Annual ≥ (MT)", 0.0, value=0.0)
monthly_min = st.sidebar.number_input("Monthly ≥ (MT)", 0.0, value=0.0)
weekly_min  = st.sidebar.number_input("Weekly ≥ (MT)", 0.0, value=0.0)
top_n = st.sidebar.number_input("Top N by Annual (0 = all)", 0, value=0, step=5)

st.sidebar.subheader("Search / Select workshops")
all_names = sorted(sites["Customer_Name"].dropna().astype(str).unique().tolist())

# Init default selection once (all)
if st.session_state.selected_names is None:
    st.session_state.selected_names = all_names

b1, b2 = st.sidebar.columns(2)
with b1:
    if st.button("Select all", key="btn_select_all"):
        st.session_state.selected_names = all_names
with b2:
    if st.button("Clear all", key="btn_clear_all"):
        st.session_state.selected_names = []

selected_names = st.sidebar.multiselect(
    "Tick/Untick names",
    options=all_names,
    default=st.session_state.selected_names
)
st.session_state.selected_names = selected_names

st.sidebar.header("Map display")
use_jitter = st.sidebar.checkbox("Jitter overlapping pins", True)
jitter_m = st.sidebar.slider("Jitter meters", 0, 800, 140) if use_jitter else 0

# ---------------- Apply filters ----------------
f = sites.copy()

if selected_names:
    f = f[f["Customer_Name"].isin(selected_names)]
else:
    f = f.iloc[0:0]

f = f[
    (f["Generation_MT"] >= annual_min) &
    (f["Monthly_Generation_MT"] >= monthly_min) &
    (f["Weekly_Generation_MT"] >= weekly_min)
]

if top_n > 0:
    f = f.sort_values("Generation_MT", ascending=False).head(int(top_n))

weekly_total = float(f["Weekly_Generation_MT"].sum()) if len(f) else 0.0
monthly_total = float(f["Monthly_Generation_MT"].sum()) if len(f) else 0.0
annual_total = float(f["Generation_MT"].sum()) if len(f) else 0.0
kpi_cards(len(f), weekly_total, monthly_total, annual_total)

# For map plotting
f_plot = apply_jitter(f, jitter_m)
sizes = scale_sizes(f_plot["Weekly_Generation_MT"], min_size=12, max_size=30)

# ---------------- Cluster controls (buttons) ----------------
btn1, btn2 = st.columns([1, 1])

with btn1:
    if st.button("Select Cluster"):
        st.session_state.cluster_mode = True

with btn2:
    if st.button("Clear Selection"):
        st.session_state.map_selected_codes = []
        st.session_state.cluster_mode = False

# ---------------- Build ONE map ----------------
fig = go.Figure()

# Workshops trace (must be trace 0 for selection indexing)
if len(f_plot) > 0:
    fig.add_trace(go.Scattermapbox(
        lat=f_plot["Lat_plot"],
        lon=f_plot["Lon_plot"],
        mode="markers",
        marker=dict(size=sizes, color=BLUE, opacity=0.9),
        customdata=f_plot["Customer_Code"].astype(str),
        hovertext=f_plot["Customer_Name"],
        hovertemplate="<b>%{hovertext}</b><extra></extra>",
        name="Workshops"
    ))

# Depot trace
fig.add_trace(go.Scattermapbox(
    lat=[float(depot["Latitude"])],
    lon=[float(depot["Longitude"])],
    mode="markers",
    marker=dict(size=30, color=ORANGE, opacity=1),
    hovertext=["Depot"],
    hovertemplate="<b>Depot</b><extra></extra>",
    name="Depot"
))

fig.update_layout(
    mapbox=dict(
        style="open-street-map",
        center=dict(lat=float(depot["Latitude"]) - 0.4, lon=float(depot["Longitude"]) + 0.1),
        zoom=6.8
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    height=620,
    dragmode="lasso" if st.session_state.cluster_mode else "pan"
)

# ---------------- Single render container ----------------
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

if st.session_state.cluster_mode:
    st.info("Cluster mode ON — use toolbar (top-right) → Box Select or Lasso Select, then drag on the map.")

    selected = plotly_events(
        fig,
        select_event=True,
        click_event=False,
        hover_event=False,
        override_height=620
    )

    # Persist selection
    if selected:
        idxs = []
        for p in selected:
            if p.get("curveNumber", 0) == 0 and "pointIndex" in p:
                idxs.append(int(p["pointIndex"]))
        idxs = sorted(set(idxs))
        if idxs:
            st.session_state.map_selected_codes = f_plot.iloc[idxs]["Customer_Code"].astype(str).tolist()

# Always show the same single map exactly once
st.plotly_chart(fig, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Selected cluster summary ----------------
st.subheader("Cluster Summary")

selected_codes = st.session_state.map_selected_codes
selected_df = f[f["Customer_Code"].astype(str).isin(selected_codes)].copy() if selected_codes else f.iloc[0:0].copy()

if len(selected_df) == 0:
    st.caption("No cluster selected. Click Select Cluster and draw a box/lasso on the map.")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Workshops", len(selected_df))
    c2.metric("Weekly MT", round(float(selected_df["Weekly_Generation_MT"].sum()), 2))
    c3.metric("Monthly MT", round(float(selected_df["Monthly_Generation_MT"].sum()), 2))
    c4.metric("Annual MT", round(float(selected_df["Generation_MT"].sum()), 2))

    st.dataframe(
        selected_df[[
            "Customer_Name",
            "City",
            "Pin_Code",
            "Weekly_Generation_MT",
            "Monthly_Generation_MT",
            "Generation_MT"
        ]].sort_values("Generation_MT", ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=360
    )

# ---------------- Full filtered list + download ----------------
st.divider()
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
