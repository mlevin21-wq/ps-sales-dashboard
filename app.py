import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from rapidfuzz import process   # pip install rapidfuzz

# -----------------------
# Load and prepare data
# -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("PlayStation Sales and Metadata (PS3PS4PS5) (Oct 2025).csv")

    # Keep only PlayStation consoles
    df = df[df["Console"].str.contains("PS", na=False)]

    # Parse date + make Year column
    df["Release Date"] = pd.to_datetime(df["Release Date"], errors="coerce")
    df["Year"] = df["Release Date"].dt.year

    return df

df = load_data()

# Convenience: drop rows with no Total Sales
df = df[df["Total Sales"].notna()]

# -----------------------
# Sidebar filters
# -----------------------
st.sidebar.title("Filters")

# Console multi-select
all_consoles = sorted(df["Console"].dropna().unique().tolist())
selected_consoles = st.sidebar.multiselect(
    "Console",
    options=all_consoles,
    default=all_consoles,
)

# Genre dropdown (we’ll flatten all unique genres)
all_genres = set()
for g in df["genres"].dropna():
    for genre in str(g).split(","):
        all_genres.add(genre.strip())
all_genres = sorted(all_genres)

selected_genre = st.sidebar.selectbox(
    "Genre",
    options=["All genres"] + all_genres,
    index=0,
)

# Year range slider (ignore NaN years)
year_min = int(df["Year"].dropna().min())
year_max = int(df["Year"].dropna().max())
year_range = st.sidebar.slider(
    "Year range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max),
)

# Minimum total sales (in units)
max_sales = float(df["Total Sales"].max())
min_sales = st.sidebar.slider(
    "Min Total Sales (units)",
    min_value=0.0,
    max_value=max_sales,
    value=0.0,
    step=max_sales / 20,
)

# -----------------------
# Filter data based on controls
# -----------------------
filtered = df.copy()

# 1) Console filter
if selected_consoles:
    filtered = filtered[filtered["Console"].isin(selected_consoles)]

# 2) Genre filter
if selected_genre != "All genres":
    filtered = filtered[
        filtered["genres"]
        .fillna("")
        .str.contains(selected_genre, na=False)
    ]

# 3) Year filter
start_year, end_year = year_range
filtered = filtered[
    (filtered["Year"].notna())
    & (filtered["Year"].between(start_year, end_year))
]

# 4) Min sales filter
filtered = filtered[filtered["Total Sales"] >= min_sales]

# If no data left, show message and stop
if filtered.empty:
    st.title("PlayStation Sales Dashboard & Recommender")
    st.warning("No games match the selected filters. Try relaxing your filters.")
    st.stop()

# -----------------------
# Page title & description
# -----------------------
st.title("PlayStation Sales Dashboard & Game Recommender")

st.write(
    """
This interactive dashboard uses PlayStation 3, 4, and 5 game sales and metadata to help
users explore performance across consoles, years, and genres. It also includes a
**game recommender** that suggests similar titles based on what you like.
"""
)

# -----------------------
# KPIs
# -----------------------
n_games = len(filtered)
total_sales = filtered["Total Sales"].sum()
avg_rating = filtered["rating"].mean() if "rating" in filtered.columns else None

# Top publisher by total sales
if "Publisher" in filtered.columns:
    publisher_sales = (
        filtered.groupby("Publisher")["Total Sales"]
        .sum()
        .sort_values(ascending=False)
    )
    top_publisher = publisher_sales.index[0]
    top_publisher_sales = publisher_sales.iloc[0]
else:
    top_publisher, top_publisher_sales = None, None

st.subheader("Key Metrics (Filtered)")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Number of games", n_games)

with col2:
    st.metric("Total Sales (units)", f"{int(total_sales):,}")

with col3:
    if avg_rating is not None and not pd.isna(avg_rating):
        st.metric("Average rating", f"{avg_rating:.2f}")
    else:
        st.metric("Average rating", "N/A")

with col4:
    if top_publisher is not None:
        st.metric(
            "Top publisher (by sales)",
            f"{top_publisher} ({int(top_publisher_sales):,} units)",
        )
    else:
        st.metric("Top publisher (by sales)", "N/A")

st.markdown("---")

# -----------------------
# Chart 1 – Top 10 games by Total Sales
# -----------------------
st.subheader("Top 10 Games by Total Sales (Filtered)")

top_games = (
    filtered.sort_values("Total Sales", ascending=False)[
        ["Name", "Total Sales"]
    ]
    .head(10)
)

fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.barh(top_games["Name"], top_games["Total Sales"])
ax1.set_xlabel("Total Sales (units)")
ax1.set_ylabel("Game")
ax1.invert_yaxis()  # largest on top
plt.tight_layout()
st.pyplot(fig1)

st.markdown("---")

# -----------------------
# Chart 2 – Total Sales by Console
# -----------------------
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
plt.tight_layout()
st.pyplot(fig2)

st.markdown("---")

# -----------------------
# Sample of filtered data
# -----------------------
st.subheader("Sample of Filtered Data")
cols_to_show = [
    c
    for c in ["Name", "Console", "Publisher", "Total Sales", "rating", "genres", "Year"]
    if c in filtered.columns
]
st.dataframe(filtered[cols_to_show].head(20))

st.markdown("---")

# =========================================================
# 🎮 GAME RECOMMENDER SECTION
# =========================================================
st.header("🎮 PlayStation Game Recommender")

st.write(
    """
Type in the name of a PlayStation game you enjoy.  
The app will find the closest match in the dataset and recommend similar games
(based on console, overlapping genres, and higher ratings / sales).
"""
)

user_title = st.text_input("Enter a PlayStation game you like:")

if user_title:
    # All unique game names
    choices = df["Name"].dropna().unique().tolist()

    match = process.extractOne(user_title, choices)
    if match is None or match[1] < 60:
        st.warning(
            "I couldn't find a close match. Try a simpler title or check the spelling."
        )
    else:
        best_name, score, _ = match
        st.success(f"Closest match: **{best_name}** (confidence {score:.1f}%)")

        game_row = df[df["Name"] == best_name].iloc[0]
        game_console = game_row["Console"]
        raw_genres = str(game_row["genres"])
        base_genres = {g.strip() for g in raw_genres.split(",") if g.strip()}

        st.write(
            f"Console: **{game_console}**, Genres: **{', '.join(base_genres) if base_genres else 'N/A'}**"
        )

        # Candidate pool: same console
        candidates = df[df["Console"] == game_console].copy()
        candidates = candidates[candidates["Name"] != best_name]

        # Compute genre overlap
        def genre_overlap(g):
            gset = {x.strip() for x in str(g).split(",") if x.strip()}
            return len(base_genres & gset)

        candidates["genre_overlap"] = candidates["genres"].apply(genre_overlap)

        # Keep only games sharing at least one genre, if possible
        recs = candidates[candidates["genre_overlap"] > 0]
        if recs.empty:
            recs = candidates.copy()

        # Sort by: overlap (desc), rating (desc), sales (desc)
        sort_cols = ["genre_overlap"]
        ascending = [False]

        if "rating" in recs.columns:
            sort_cols.append("rating")
            ascending.append(False)
        if "Total Sales" in recs.columns:
            sort_cols.append("Total Sales")
            ascending.append(False)

        recs = recs.sort_values(sort_cols, ascending=ascending)

        top_recs = recs[
            ["Name", "Console", "genres", "rating", "Total Sales", "Publisher"]
        ].head(5)

        st.subheader("Recommended Games for You")
        st.dataframe(top_recs.reset_index(drop=True))
