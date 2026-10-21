from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
# =========================================================
# 1. APP CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AeroFare | Flight Price Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# 2. LOAD MODEL AND DATASET FROM HUGGING FACE
# =========================================================

HF_REPO_ID = "shaswat44/flight-price-prediction-files"

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="flight_price_model.pkl",
        repo_type="model",
        token=st.secrets["HF_TOKEN"],
    )
    return joblib.load(model_path)


@st.cache_data
def load_data():
    data_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="Freshly_cleaned.csv",
        repo_type="model",
        token=st.secrets["HF_TOKEN"],
    )
    return pd.read_csv(data_path)
# =========================================================
# 3. PREMIUM AEROFARE STYLING
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --navy: #10264B;
        --blue: #2864F0;
        --blue-dark: #194AC0;
        --muted: #71809A;
        --line: #E7ECF4;
        --surface: #FFFFFF;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 5% 0%, #EAF2FF 0%, transparent 34%),
            #F7F9FC;
        color: var(--navy);
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }

    .block-container {
        max-width: 1240px;
        padding: 1.6rem 2rem 3.5rem;
    }

    /* Brand navigation */
    .topbar {
        background: rgba(255,255,255,.92);
        border: 1px solid rgba(224,232,244,.95);
        border-radius: 17px;
        padding: 16px 23px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 23px;
        box-shadow: 0 6px 25px rgba(16,38,75,.045);
        animation: fadeUp .55s ease both;
    }

    .brand {
        font-family: 'Manrope', sans-serif;
        font-size: 25px;
        font-weight: 800;
        letter-spacing: -1.2px;
        color: var(--navy);
    }

    .brand span {
        color: var(--blue);
    }

    .nav-label {
        color: #71809A;
        font-size: 11px;
        letter-spacing: 1.5px;
        font-weight: 700;
    }

    /* Hero */
    .hero {
        position: relative;
        isolation: isolate;
        overflow: hidden;
        border-radius: 25px;
        padding: clamp(27px, 5vw, 49px);
        margin-bottom: 25px;
        color: white;
        background: linear-gradient(115deg, #10264B 0%, #164887 58%, #2864F0 100%);
        box-shadow: 0 16px 40px rgba(25,74,160,.15);
        animation: fadeUp .65s ease both;
    }

    .hero:before, .hero:after {
        content: "";
        position: absolute;
        z-index: -1;
        border: 1px solid rgba(255,255,255,.13);
        border-radius: 50%;
        pointer-events: none;
    }

    .hero:before {
        width: 260px;
        height: 260px;
        right: -35px;
        top: -150px;
        box-shadow: 0 0 0 35px rgba(255,255,255,.035),
                    0 0 0 75px rgba(255,255,255,.025);
    }

    .hero:after {
        width: 140px;
        height: 140px;
        right: 17%;
        bottom: -105px;
    }

    .eyebrow {
        color: #C7DDFF;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 2.2px;
        margin-bottom: 14px;
    }

    .hero h1 {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(31px, 4.4vw, 47px);
        font-weight: 800;
        line-height: 1.13;
        letter-spacing: -2px;
        color: #FFFFFF;
        margin: 0 0 15px;
    }

    .hero p {
        max-width: 580px;
        color: #D9E7FF;
        font-size: 14px;
        line-height: 1.85;
        margin: 0;
    }

    /* Section headings */
    .section-heading {
        font-family: 'Manrope', sans-serif;
        color: var(--navy);
        font-size: 23px;
        font-weight: 800;
        letter-spacing: -.8px;
        margin: 8px 0 4px;
    }

    .section-sub {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.7;
        margin-bottom: 19px;
    }

    /* Main card surfaces */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,.96);
        border: 1px solid var(--line);
        border-radius: 19px;
        box-shadow: 0 5px 22px rgba(16,38,75,.035);
        transition: box-shadow .22s ease, border-color .22s ease;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #D4E1F7;
        box-shadow: 0 9px 27px rgba(16,38,75,.065);
    }

    /* KPI cards */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 19px 20px;
        box-shadow: 0 4px 16px rgba(16,38,75,.025);
        transition: transform .2s ease, box-shadow .2s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 9px 22px rgba(16,38,75,.07);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: .5px;
    }

    div[data-testid="stMetricValue"] {
        color: var(--navy);
        font-family: 'Manrope', sans-serif;
        font-size: 25px;
        font-weight: 800;
        letter-spacing: -.8px;
    }

    /* Fare result */
    .fare-card {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, #FFFFFF 0%, #F0F5FF 100%);
        border: 1px solid #DCE7FA;
        border-radius: 21px;
        padding: 29px;
        margin: 8px 0 19px;
        box-shadow: 0 9px 27px rgba(40,100,240,.055);
        animation: fadeUp .45s ease both;
    }

    .fare-card:after {
        content: "₹";
        position: absolute;
        right: 20px;
        top: -35px;
        font-family: 'Manrope', sans-serif;
        font-size: 145px;
        font-weight: 800;
        color: rgba(40,100,240,.045);
        pointer-events: none;
    }

    .fare-label {
        color: #617596;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .fare-price {
        position: relative;
        z-index: 1;
        font-family: 'Manrope', sans-serif;
        color: var(--navy);
        font-size: clamp(38px, 5vw, 53px);
        font-weight: 800;
        letter-spacing: -2.2px;
        margin: 10px 0;
    }

    .fare-note {
        max-width: 440px;
        color: #687994;
        font-size: 12px;
        line-height: 1.8;
    }

    .route-pill {
        display: inline-block;
        background: #E9F0FF;
        color: #2455B2;
        border: 1px solid #DDE8FF;
        padding: 7px 12px;
        border-radius: 30px;
        font-size: 11px;
        font-weight: 700;
        margin: 4px 5px 3px 0;
    }

    /* Inputs */
    label {
        color: #334968 !important;
        font-weight: 600 !important;
        font-size: 12px !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 10px;
        border-color: #DCE5F1;
        background: #FFFFFF;
        transition: border-color .2s ease, box-shadow .2s ease;
    }

    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:focus-within {
        border-color: #6B96FF;
        box-shadow: 0 0 0 3px rgba(40,100,240,.09);
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(105deg, #2864F0, #3977FF);
        color: #FFFFFF;
        border: 1px solid transparent;
        border-radius: 11px;
        min-height: 47px;
        padding: 12px 18px;
        font-weight: 700;
        box-shadow: 0 5px 13px rgba(40,100,240,.13);
        transition: transform .18s ease, box-shadow .18s ease,
                    background .18s ease;
    }

    div.stButton > button:hover {
        background: linear-gradient(105deg, #194AC0, #2864F0);
        color: #FFFFFF;
        transform: translateY(-2px);
        box-shadow: 0 8px 19px rgba(40,100,240,.22);
    }

    div.stButton > button:active {
        transform: translateY(0);
    }

    /* Charts and separators */
    div[data-testid="stVegaLiteChart"],
    div[data-testid="stArrowVegaLiteChart"],
    div[data-testid="stPlotlyChart"] {
        border-radius: 12px;
    }

    hr {
        border-color: #E9EEF6;
        margin: 22px 0;
    }

    .footnote {
        color: #8996A9;
        font-size: 11px;
        line-height: 1.9;
        text-align: center;
        padding-top: 25px;
    }

    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(9px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: .01ms !important;
            transition-duration: .01ms !important;
        }
    }

    @media (max-width: 700px) {
        .block-container {
            padding: 1rem 1rem 2rem;
        }

        .topbar {
            padding: 14px 16px;
        }

        .nav-label {
            font-size: 9px;
            letter-spacing: .5px;
            text-align: right;
        }

        .hero {
            padding: 27px 22px;
            border-radius: 19px;
        }

        .fare-card {
            padding: 23px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# =========================================================
# 4. LOAD EXISTING MODEL AND DATASET
# =========================================================

try:
    model = load_model()
    df = load_data()
except Exception as error:
    st.error(f"Could not load the model or dataset: {error}")
    st.stop()


FEATURES = [
    "airline",
    "from",
    "to",
    "class_category",
    "class",
    "day",
    "month",
    "route",
    "dep_hour",
    "arr_hour",
    "dep_period",
    "arr_period",
    "duration_in_min",
    "stops",
    "stops_category",
    "arr_daytime",
    "arr_daytime_category",
    "dep_daytime",
    "dep_daytime_category",
    "month_category",
]

required_columns = FEATURES + ["price"]

missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    st.error(f"Required dataset columns are missing: {missing_columns}")
    st.stop()


# =========================================================
# 5. HELPER FUNCTIONS
# =========================================================

def get_period(hour):
    if hour < 7:
        return "Early_morning"
    elif hour < 12:
        return "Morning"
    elif hour < 17:
        return "Afternoon"
    return "Night"


def most_common_value(frame, column, fallback):
    if column in frame.columns and not frame.empty:
        values = frame[column].dropna()
        if not values.empty:
            return values.mode().iloc[0]
    return fallback


def hour_based_value(frame, hour_column, target_column, selected_hour, fallback):
    """Use the dataset's observed mapping for the selected hour."""
    if frame.empty or hour_column not in frame.columns:
        return fallback

    exact = frame[frame[hour_column] == selected_hour]
    return most_common_value(exact, target_column, fallback)


def format_rupees(value):
    return f"₹{value:,.0f}"


def _lookup_key(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    return value


def _most_common_map(frame, group_col, target_col):
    """Most common target value for each group, with stable Python keys."""
    mapping = {}
    subset = frame[[group_col, target_col]].dropna()
    if subset.empty:
        return mapping

    for key, values in subset.groupby(group_col, sort=False)[target_col]:
        modes = values.mode()
        if modes.empty:
            continue
        mapping[_lookup_key(key)] = modes.iloc[0]
    return mapping


@st.cache_data
def get_select_options():
    data = load_data()
    airlines = sorted(data["airline"].dropna().astype(str).unique().tolist())
    cities = sorted(
        set(data["from"].dropna().astype(str).unique())
        | set(data["to"].dropna().astype(str).unique())
    )
    classes = sorted(data["class_category"].dropna().astype(str).unique().tolist())
    months = sorted(data["month"].dropna().astype(int).unique().tolist())
    return airlines, cities, classes, months


@st.cache_data
def get_overview_metrics():
    data = load_data()
    return {
        "total_flights": len(data),
        "airline_count": int(data["airline"].nunique()),
        "route_count": int(data[["from", "to"]].drop_duplicates().shape[0]),
        "average_fare": float(data["price"].mean()),
    }


@st.cache_data
def get_fare_context_stats():
    prices = load_data()["price"].dropna()
    return {
        "low_fare": float(prices.quantile(0.25)),
        "median_fare": float(prices.median()),
        "high_fare": float(prices.quantile(0.75)),
    }


@st.cache_data
def get_airline_fares():
    data = load_data()
    return (
        data.groupby("airline", as_index=False)["price"]
        .mean()
        .sort_values("price", ascending=False)
        .set_index("airline")
    )


@st.cache_data
def get_class_fares():
    data = load_data()
    return (
        data.groupby("class_category", as_index=False)["price"]
        .mean()
        .sort_values("price", ascending=False)
        .set_index("class_category")
    )


@st.cache_data
def get_route_summary():
    data = load_data()
    return (
        data.assign(route_label=data["from"].astype(str) + " → " + data["to"].astype(str))
        .groupby("route_label")
        .agg(
            flights=("price", "size"),
            average_fare=("price", "mean"),
        )
        .sort_values("flights", ascending=False)
        .head(8)
    )


@st.cache_data
def get_fare_distribution_chart():
    """Representative fare histogram with ~160 bins instead of 300k raw points."""
    prices = load_data()["price"].dropna()
    if prices.empty:
        return pd.DataFrame({"flights": pd.Series(dtype="int64")})

    clipped = prices.clip(upper=float(prices.quantile(0.95)))
    n_bins = 160
    hist = clipped.groupby(pd.cut(clipped, bins=n_bins), observed=False).size()
    bin_mids = [float(interval.mid) for interval in hist.index]
    return pd.DataFrame(
        {"flights": hist.to_numpy(dtype="int64")},
        index=pd.Index(bin_mids, name="Fare (₹)"),
    )


@st.cache_data
def get_feature_lookups():
    """Reusable dataset mappings used while building a prediction row."""
    data = load_data()
    return {
        "dep_daytime": _most_common_map(data, "dep_hour", "dep_daytime"),
        "dep_daytime_category": _most_common_map(
            data, "dep_hour", "dep_daytime_category"
        ),
        "arr_daytime": _most_common_map(data, "arr_hour", "arr_daytime"),
        "arr_daytime_category": _most_common_map(
            data, "arr_hour", "arr_daytime_category"
        ),
        "month_category": _most_common_map(data, "month", "month_category"),
    }


airlines, cities, classes, available_months = get_select_options()
overview = get_overview_metrics()
fare_context_stats = get_fare_context_stats()
airline_fares = get_airline_fares()
class_fares = get_class_fares()
route_summary = get_route_summary()
fare_distribution = get_fare_distribution_chart()
feature_lookups = get_feature_lookups()

if not airlines or not cities or not classes:
    st.error("The dataset does not contain the expected airline, city, or class values.")
    st.stop()


# =========================================================
# 6. TOP NAVIGATION + HERO
# =========================================================

st.markdown(
    """
    <div class="topbar">
        <div class="brand">AERO<span>FARE</span></div>
        <div class="nav-label">FLIGHT PRICE INTELLIGENCE &nbsp; · &nbsp; ML POWERED</div>
    </div>

    <div class="hero">
        <div class="eyebrow">SMARTER TRAVEL STARTS HERE</div>
        <h1>Find your flight's<br>estimated fare.</h1>
        <p>
            Explore your journey details and get a machine-learning-based
            flight fare estimate using our trained prediction model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 7. DATASET OVERVIEW
# =========================================================

total_flights = overview["total_flights"]
route_count = overview["route_count"]
airline_count = overview["airline_count"]
average_fare = overview["average_fare"]

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric("FLIGHT RECORDS", f"{total_flights:,}")
metric2.metric("AIRLINES", f"{airline_count}")
metric3.metric("ROUTE PAIRS", f"{route_count}")
metric4.metric("AVERAGE FARE", format_rupees(average_fare))


st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<div class="section-heading">Plan your journey</div>'
    '<div class="section-sub">Choose your travel details to estimate the fare.</div>',
    unsafe_allow_html=True,
)


# =========================================================
# 8. INPUT FORM
# =========================================================

with st.container(border=True):
    st.markdown("#### ✈️ Route & airline")

    col_from, col_swap, col_to = st.columns([5, 1, 5])

    with col_from:
        source = st.selectbox("From", cities, index=0)

    with col_swap:
        st.markdown("<br>", unsafe_allow_html=True)
        swap = st.button("⇄", help="Swap origin and destination")

    with col_to:
        default_dest_index = 1 if len(cities) > 1 else 0
        destination = st.selectbox(
            "To",
            cities,
            index=default_dest_index,
        )

    if swap:
        st.session_state["source_value"] = destination
        st.session_state["destination_value"] = source
        st.rerun()

    # The selectboxes above use session-state keys below on subsequent reruns.
    # These keys allow the swap button to retain its chosen route.
    if "source_value" in st.session_state:
        source = st.session_state["source_value"]
    if "destination_value" in st.session_state:
        destination = st.session_state["destination_value"]

    airline_col, class_col, month_col, day_col = st.columns(4)

    with airline_col:
        airline = st.selectbox("Airline", airlines)

    with class_col:
        travel_class = st.selectbox("Travel class", classes)

    with month_col:
        month = st.selectbox("Travel month", available_months)

    with day_col:
        max_day = 28 if month == 2 else 31
        day = st.number_input("Travel day", min_value=1, max_value=max_day, value=15)

    st.markdown("---")
    st.markdown("#### 🕒 Schedule & stops")

    time_col1, time_col2, duration_col, stops_col = st.columns(4)

    with time_col1:
        dep_hour = st.selectbox(
            "Departure time",
            list(range(24)),
            index=9,
            format_func=lambda h: f"{h:02d}:00",
        )

    with time_col2:
        arr_hour = st.selectbox(
            "Arrival time",
            list(range(24)),
            index=12,
            format_func=lambda h: f"{h:02d}:00",
        )

    with duration_col:
        duration = st.number_input(
            "Duration (minutes)",
            min_value=50,
            max_value=2990,
            value=150,
            step=10,
        )

    with stops_col:
        stops_label = st.selectbox(
            "Stops",
            ["Non-stop", "1-Stop", "Multiple-Stops"],
        )

    stop_map = {
        "Non-stop": 0,
        "1-Stop": 1,
        "Multiple-Stops": 2,
    }
    stops = stop_map[stops_label]

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button(
        "✦  Predict Flight Fare",
        type="primary",
        use_container_width=True,
    )


# =========================================================
# 9. PREDICTION
# =========================================================

if predict_clicked:
    if source == destination:
        st.error("Origin and destination must be different.")
        st.stop()

    # Match the selected route, airline, and travel class to the dataset.
    route_rows = df[
        (df["from"].astype(str) == source)
        & (df["to"].astype(str) == destination)
    ]

    if route_rows.empty:
        st.warning(
            "This direction is not present in the training dataset. "
            "The estimate may be less reliable."
        )
        template_rows = df[
            (df["airline"].astype(str) == airline)
            & (df["class_category"].astype(str) == travel_class)
        ]
    else:
        template_rows = route_rows[
            (route_rows["airline"].astype(str) == airline)
            & (route_rows["class_category"].astype(str) == travel_class)
        ]

        if template_rows.empty:
            template_rows = route_rows

    if template_rows.empty:
        template_rows = df[
            (df["airline"].astype(str) == airline)
            & (df["class_category"].astype(str) == travel_class)
        ]

    if template_rows.empty:
        template_rows = df

    # Prefer a row with the selected stop count when available.
    matching_stops = template_rows[template_rows["stops"] == stops]
    if not matching_stops.empty:
        template_rows = matching_stops

    template = template_rows.iloc[0]

    # Find common dataset values associated with the selected month.
    month_category = feature_lookups["month_category"].get(_lookup_key(month))
    if month_category is None:
        month_rows = df[df["month"] == month]
        month_category = most_common_value(
            month_rows, "month_category", template["month_category"]
        )

    # Get daytime feature values using observed values for selected hours.
    dep_daytime = feature_lookups["dep_daytime"].get(_lookup_key(dep_hour))
    if dep_daytime is None:
        dep_daytime = hour_based_value(
            df, "dep_hour", "dep_daytime", dep_hour, template["dep_daytime"]
        )

    dep_daytime_category = feature_lookups["dep_daytime_category"].get(
        _lookup_key(dep_hour)
    )
    if dep_daytime_category is None:
        dep_daytime_category = hour_based_value(
            df, "dep_hour", "dep_daytime_category",
            dep_hour, template["dep_daytime_category"]
        )

    arr_daytime = feature_lookups["arr_daytime"].get(_lookup_key(arr_hour))
    if arr_daytime is None:
        arr_daytime = hour_based_value(
            df, "arr_hour", "arr_daytime", arr_hour, template["arr_daytime"]
        )

    arr_daytime_category = feature_lookups["arr_daytime_category"].get(
        _lookup_key(arr_hour)
    )
    if arr_daytime_category is None:
        arr_daytime_category = hour_based_value(
            df, "arr_hour", "arr_daytime_category",
            arr_hour, template["arr_daytime_category"]
        )

    # Route value: use the observed route value for the selected direction.
    selected_route = most_common_value(
        route_rows, "route", template["route"]
    )

    input_data = {
        "airline": airline,
        "from": source,
        "to": destination,
        "class_category": travel_class,
        "class": template["class"],
        "day": int(day),
        "month": int(month),
        "route": selected_route,
        "dep_hour": int(dep_hour),
        "arr_hour": int(arr_hour),
        "dep_period": get_period(dep_hour),
        "arr_period": get_period(arr_hour),
        "duration_in_min": int(duration),
        "stops": int(stops),
        "stops_category": stops_label,
        "arr_daytime": arr_daytime,
        "arr_daytime_category": arr_daytime_category,
        "dep_daytime": dep_daytime,
        "dep_daytime_category": dep_daytime_category,
        "month_category": month_category,
    }

    input_df = pd.DataFrame([input_data], columns=FEATURES)

    try:
        prediction = float(model.predict(input_df)[0])
    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.stop()

    prediction = max(0, prediction)

    st.session_state["last_prediction"] = {
        "fare": prediction,
        "input": input_data,
        "source": source,
        "destination": destination,
        "airline": airline,
        "travel_class": travel_class,
        "day": int(day),
        "month": int(month),
        "duration": int(duration),
        "stops_label": stops_label,
        "dep_hour": int(dep_hour),
        "arr_hour": int(arr_hour),
    }


# =========================================================
# 10. PREDICTION RESULT
# =========================================================

result = st.session_state.get("last_prediction")

if result:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-heading">Your fare estimate</div>'
        '<div class="section-sub">Based on the details you selected.</div>',
        unsafe_allow_html=True,
    )

    fare = result["fare"]

    left, right = st.columns([1.25, 1])

    with left:
        st.markdown(
            f"""
            <div class="fare-card">
                <div class="fare-label">Estimated flight fare</div>
                <div class="fare-price">{format_rupees(fare)}</div>
                <div class="fare-note">
                    This is a machine-learning estimate, not a live airline quote.
                    Actual fares can differ based on availability, booking time,
                    taxes, and airline pricing.
                </div>
                <br>
                <span class="route-pill">{result['airline']}</span>
                <span class="route-pill">{result['travel_class']}</span>
                <span class="route-pill">{result['stops_label']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Journey summary")
        st.markdown(
            f"**{result['source']}**  →  **{result['destination']}**"
        )

        detail1, detail2, detail3 = st.columns(3)
        detail1.metric("Departure", f"{result['dep_hour']:02d}:00")
        detail2.metric("Arrival", f"{result['arr_hour']:02d}:00")
        detail3.metric("Duration", f"{result['duration']} min")

    with right:
        st.markdown("#### Fare context")

        low_fare = fare_context_stats["low_fare"]
        median_fare = fare_context_stats["median_fare"]
        high_fare = fare_context_stats["high_fare"]

        context = pd.DataFrame({
            "Fare reference": ["25th percentile", "Median fare", "75th percentile", "Your prediction"],
            "Price (₹)": [low_fare, median_fare, high_fare, fare],
        })

        st.bar_chart(
            context.set_index("Fare reference"),
            y="Price (₹)",
        )

        if fare < low_fare:
            st.info("The estimate is below the dataset's 25th-percentile fare.")
        elif fare > high_fare:
            st.warning("The estimate is above the dataset's 75th-percentile fare.")
        else:
            st.success("The estimate is within the dataset's middle 50% fare range.")

    st.caption(
        "Fare context uses the full training dataset, not live fares for this route. "
        "The comparison is for general context only."
    )


# =========================================================
# 11. DATA INSIGHTS DASHBOARD
# =========================================================

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    '<div class="section-heading">Explore flight fare insights</div>'
    '<div class="section-sub">Understand the dataset used by the prediction model.</div>',
    unsafe_allow_html=True,
)

chart_left, chart_right = st.columns(2)

with chart_left:
    with st.container(border=True):
        st.markdown("#### Average fare by airline")

        st.bar_chart(airline_fares, y="price")
        st.caption("Average observed price in the dataset, not a current quote.")

with chart_right:
    with st.container(border=True):
        st.markdown("#### Fare distribution")

        st.area_chart(fare_distribution, y="flights")
        st.caption("Prices are capped at the 95th percentile for readability.")

chart_bottom_left, chart_bottom_right = st.columns(2)

with chart_bottom_left:
    with st.container(border=True):
        st.markdown("#### Average fare by class")

        st.bar_chart(class_fares, y="price")

with chart_bottom_right:
    with st.container(border=True):
        st.markdown("#### Most common routes")

        st.bar_chart(route_summary[["flights"]])
        st.caption(
            "Average fare for the most common routes: "
            + " · ".join(
                f"{route}: {format_rupees(price)}"
                for route, price in route_summary["average_fare"].head(3).items()
            )
        )


# =========================================================
# 12. ABOUT MODEL + FOOTER
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

with st.expander("About this prediction model"):
    st.write(
        """
        AeroFare uses a previously trained Random Forest regression pipeline
        to estimate a flight fare from the selected journey details.

        **Model evaluation from the project training run:**
        - Mean Absolute Error (MAE): approximately ₹1,434
        - Root Mean Squared Error (RMSE): approximately ₹3,127
        - R² score: approximately 0.981

        These are test-set metrics from the existing project model. They do
        not mean every prediction is accurate to 98.1%. Actual prediction
        error can vary by route, airline, travel class, and the data available.
        """
    )

st.markdown(
    """
    <div class="footnote">
        AEROFARE · FLIGHT PRICE INTELLIGENCE<br>
        Built with Python, Streamlit, Pandas and Scikit-learn.<br>
        Estimates are for educational purposes and are not live booking prices.
    </div>
    """,
    unsafe_allow_html=True,
)


