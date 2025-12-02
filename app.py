import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="PlayStation Sales Dashboard",
    layout="wide"
)

# -------------------------
# Load data
# -------------------------
@st.cache_data
def load_data():
    # Make sure this filename matches EXACTLY the CSV name in your GitHub repo
    df = pd.read_csv("PlayStation Sales and Metadata (PS3PS4PS5) (Oct 2025).csv")

    # Parse Release Date -> Year (for filtering)
    if "Release Date" in df.columns:
        df["Release Date"] = pd.to_datetime(df["Release Date"], errors="coerce")
        df["Year"] = df["Release Date"].dt.year
    else:
        # Fallback if Year already exists or Release Date missing
        if "Year" not in df.columns:
            df["Year"] = np.nan

    return df


df = load_data()

# -------------------------
# Sidebar Filters
# -------------------------
st.sidebar.title("Filters")

# Console filter
if "Console" in df.columns:
    console_options = sorted(df["Console"].dropna().unique().tolist())
    selected_consoles = st.sidebar.multiselect(
        "Console",
        console_options,
        default=console_options
    )
else:
    selected_consoles = []

# Genre filter (we extract unique genres from the comma-separated list)
if "genres" in df.columns:
    all_genres = set()
    for g in df["genres"].dropna():
        for item in str(g).split(","):
            item = item.strip()
            if item:
                all_genres.add(item)
    genre_options = ["All genres"] + sorted(all_genres)
    selected_genre = st.sidebar.selectbox("Genre", genre_options, index=0)
else:
    selected_genre = "All genres"

# Year filter
if "Year" in df.columns and df["Year"].notna().any():
    min_year = int(df["Year"].min())
    max_year = int(df["Year"].max())
    year_range = st.sidebar.slider(
        "Year range",
        min_year,
        max_year,
        (min_year, max_year)
    )
else:
    year_range = (None, None)

# Minimum Total Sales filter
if "Total Sales" in df.columns:
    max_sales = float(df["Total Sales"].max())
    min_sales = st.sidebar.slider(
        "Minimum Global Sales",
        0.0,
        max_sales,
        0.0,
        step=max_sales / 50 if max_sales > 0 else 1.0
    )
else:
    min_sales = 0.0

# -------------------------
# Apply Filters
# -------------------------
filtered = df.copy()

# Filter by console
if selected_consoles:
    filtered = filtered[filtered["Console"].isin(selected_consoles)]

# Filter by genre
if selected_genre != "All genres" and "genres" in filtered.columns:
    filtered = filtered[
        filtered["genres"]
        .fillna("")
        .str.contains(selected_genre, case=False, na=False)
    ]

# Filter by year
if "Year" in filtered.columns and year_range[0] is not None:
    start_year, end_year = year_range
    filtered = filtered[
        (filtered["Year"].notna()) &
        (filtered["Year"] >= start_year) &
        (filtered["Year"] <= end_year)
    ]

# Filter by minimum Total Sales
if "Total Sales" in filtered.columns:
    filtered = filtered[filtered["Total Sales"] >= min_sales]

# If nothing left after filters
if filtered.empty:
    st.warning("No games match the selected filters. Try relaxing your filters.")
    st.stop()

# -------------------------
# Main Title & Description
# -------------------------
st.title("Interactive PlayStation Sales Dashboard")

st.markdown(
    """
This dashboard explores **PlayStation 3 / 4 / 5 game sales and metadata**.
Use the filters in the left sidebar to slice the data by console, genre, year,
and minimum global sales. The visualizations and KPIs below update dynamically
based on your selections.
"""
)

# -------------------------
# KPIs
# -------------------------
st.subheader("Key Metrics (Filtered)")

n_games = len(filtered)

total_sales = (
    filtered["Total Sales"].sum()
    if "Total Sales" in filtered.columns
    else None
)
avg_rating = (
    filtered["rating"].mean()
    if "rating" in filtered.columns
    else None
)

top_publisher = None
top_publisher_sales = None
if "Publisher" in filtered.columns and "Total Sales" in filtered.columns:
    publisher_sales = (
        filtered.groupby("Publisher")["Total Sales"]
        .sum()
        .sort_values(ascending=False)
    )
    if not publisher_sales.empty:
        top_publisher = publisher_sales.index[0]
        top_publisher_sales = publisher_sales.iloc[0]

# Display KPIs in columns
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.metric("Number of Games", n_games)

with kpi_col2:
    if total_sales is not None:
        st.metric("Total Global Sales", f"{total_sales:,.0f} units")
    else:
        st.metric("Total Global Sales", "N/A")

with kpi_col3:
    if avg_rating is not None and not np.isnan(avg_rating):
        st.metric("Average Rating", f"{avg_rating:.2f}")
    else:
        st.metric("Average Rating", "N/A")

with kpi_col4:
    if top_publisher is not None:
        st.metric(
            "Top Publisher by Sales",
            f"{top_publisher} ({top_publisher_sales:,.0f} units)"
        )
    else:
        st.metric("Top Publisher by Sales", "N/A")

st.markdown("---")

# -------------------------
# Chart 1 – Top 10 Games by Total Sales
# -------------------------
if "Total Sales" in filtered.columns and "Name" in filtered.columns:
    st.subheader("Top 10 Games by Total Sales (Filtered)")

    top_games = (
        filtered.sort_values("Total Sales", ascending=False)
        .head(10)
        .copy()
    )

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.barh(top_games["Name"], top_games["Total Sales"])
    ax1.set_xlabel("Total Sales (units)")
    ax1.set_ylabel("Game")
    ax1.set_title("Top 10 Games by Total Sales (Filtered)")
    ax1.invert_yaxis()  # largest on top
    plt.tight_layout()
    st.pyplot(fig1)

    st.markdown("---")

# -------------------------
# Chart 2 – Total Sales by Console
# -------------------------
if "Total Sales" in filtered.columns and "Console" in filtered.columns:
    st.subheader("Total Sales by Console (Filtered)")

    sales_by_console = (
        filtered.groupby("Console")["Total Sales"]
        .sum()
        .reset_index()
        .sort_values("Total Sales", ascending=False)
    )

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(sales_by_console["Console"], sales_by_console["Total Sales"])
    ax2.set_xlabel("Console")
    ax2.set_ylabel("Total Sales (units)")
    ax2.set_title("Total Sales by Console (Filtered)")
    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown("---")

# ==========================================================
# Regional Sales Analysis (NA vs PAL vs Japan)
# ==========================================================

st.subheader("Regional Sales Comparison: NA vs PAL vs Japan")

# 1) Side-by-side bar chart for top 10 games
needed_cols = {"NA Sales", "PAL Sales", "Japan Sales", "Name"}
if needed_cols.issubset(filtered.columns):

    st.markdown(
        """
**Chart A – Side-by-side bar chart**  
Shows **NA vs PAL vs Japan sales** for the top 10 games by Total Sales (after filters).
"""
    )

    # Use the same top 10 games as above (by Total Sales)
    regional_top = (
        filtered.sort_values("Total Sales", ascending=False)
        .head(10)
        .copy()
    )

    regional_top = regional_top.set_index("Name")[
        ["NA Sales", "PAL Sales", "Japan Sales"]
    ]

    x = np.arange(len(regional_top))  # game positions
    width = 0.25

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    ax3.bar(x - width, regional_top["NA Sales"], width, label="NA")
    ax3.bar(x, regional_top["PAL Sales"], width, label="PAL")
    ax3.bar(x + width, regional_top["Japan Sales"], width, label="Japan")

    ax3.set_xticks(x)
    ax3.set_xticklabels(regional_top.index, rotation=45, ha="right")
    ax3.set_ylabel("Sales (units)")
    ax3.set_title("NA vs PAL vs Japan Sales for Top 10 Games (Filtered)")
    ax3.legend()
    plt.tight_layout()
    st.pyplot(fig3)

else:
    st.info("Regional columns (NA Sales, PAL Sales, Japan Sales) not found in data.")

st.markdown("---")

# 2) Map visualization of regional totals
if {"NA Sales", "PAL Sales", "Japan Sales"}.issubset(filtered.columns):

st.subheader("Chart B – Regional sales map (pydeck)")
st.caption(
    "Each bubble represents the total sales, aggregated over the filtered games, "
    "for North America (NA), PAL (Europe), and Japan."
)

# Make sure the region columns exist
region_cols = ["NA Sales", "PAL Sales", "Japan Sales"]
available_region_cols = [c for c in region_cols if c in filtered.columns]

if not available_region_cols:
    st.info("No regional sales columns found in the data.")
else:
    # 1. Aggregate regional sales over the filtered games
    na_total = filtered["NA Sales"].sum() if "NA Sales" in filtered.columns else 0
    pal_total = filtered["PAL Sales"].sum() if "PAL Sales" in filtered.columns else 0
    jp_total = filtered["Japan Sales"].sum() if "Japan Sales" in filtered.columns else 0

    # 2. Build a small DataFrame with SAFE column names (no spaces)
    region_sales = pd.DataFrame(
        [
            {"region": "North America", "lat": 40.0, "lon": -100.0, "total_sales": na_total},
            {"region": "PAL (Europe)", "lat": 50.0, "lon": 10.0, "total_sales": pal_total},
            {"region": "Japan", "lat": 36.0, "lon": 138.0, "total_sales": jp_total},
        ]
    )

    # scale radius so bubbles aren't too huge or tiny
    max_sales = region_sales["total_sales"].max()
    if max_sales > 0:
        region_sales["radius"] = region_sales["total_sales"] / max_sales * 2_000_000
    else:
        region_sales["radius"] = 0

    # 3. Define pydeck layer
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=region_sales,
        get_position="[lon, lat]",
        get_radius="radius",
        get_fill_color="[50, 100, 200, 180]",
        pickable=True,
    )

    # 4. View state (center roughly on the Atlantic so all regions are visible)
    view_state = pdk.ViewState(
        latitude=40,
        longitude=0,
        zoom=1.2,
        bearing=0,
        pitch=30,
    )

    # 5. Tooltip – uses the **safe** column names: region, total_sales
    tooltip = {
        "html": "<b>{region}</b><br/>Total Sales: {total_sales}",
        "style": {"backgroundColor": "#222", "color": "white"},
    }

    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=view_state,
            layers=[layer],
            tooltip=tooltip,
        )
    )

# -------------------------
# Show a sample of filtered data
# -------------------------
st.subheader("Sample of Filtered Data")
cols_to_show = [
    c
    for c in ["Name", "Console", "Publisher", "Total Sales",
              "rating", "genres", "Year", "NA Sales", "PAL Sales", "Japan Sales"]
    if c in filtered.columns
]
st.dataframe(filtered[cols_to_show].head(15))
