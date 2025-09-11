import os
import math
import requests
from typing import Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from dotenv import load_dotenv

# Optional browser geolocation component (install streamlit-js-eval to enable)
try:
    from streamlit_js_eval import get_geolocation
    GEO_AVAILABLE = True
except Exception:
    GEO_AVAILABLE = False

# ----------------------------
# Load environment (API keys should be set in .env or deployment secrets)
# ----------------------------
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")  # required for live weather
# SoilGrids is public; no key required for basic queries
SOILGRIDS_BASE = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# ----------------------------
# Sample dataset (districts + crop stats)
# ----------------------------
DATA = {
    "District": ["Dhamtari", "Kurud", "Baloda bazar", "Bilaspur", "Mahasamund",
                 "Gariyaband", "Durg", "Raipur", "Rajnandgao", "Korba"],
    "Rainfall": [880, 900, 870, 850, 1100, 2400, 950, 970, 1200, 1600],
    "Temperature": [25, 24, 26, 27, 30, 32, 28, 23, 29, 26],
    "Soil_pH": [6.7, 6.8, 6.5, 6.9, 6.2, 5.8, 6.4, 6.6, 6.3, 6.5],
    "Crop": ["Wheat", "Wheat", "Rice", "Rice", "Cotton",
             "Sugarcane", "Maize", "Millets", "Pulses", "Jute"],
    "Yield": [42, 44, 39, 38, 25, 80, 32, 28, 22, 30],  # q/ha
    "Profit": [45000, 46000, 41000, 40000, 35000, 90000, 30000, 27000, 20000, 32000],  # ₹/ha
    "SuccessRate": [78, 82, 75, 73, 65, 90, 70, 68, 60, 72]  # %
}
DF = pd.DataFrame(DATA)

# ----------------------------
# Sustainability dataset
# ----------------------------
SUSTAINABILITY_DATA = {
    "Wheat":    {"Water": 1200, "Fertilizer": "Medium", "Soil": "Neutral", "Carbon": "Medium"},
    "Rice":     {"Water": 5000, "Fertilizer": "High",   "Soil": "Depleting", "Carbon": "High"},
    "Millets":  {"Water": 400,  "Fertilizer": "Low",    "Soil": "Neutral",   "Carbon": "Low"},
    "Pulses":   {"Water": 500,  "Fertilizer": "Low",    "Soil": "Enriching", "Carbon": "Low"},
    "Cotton":   {"Water": 1500, "Fertilizer": "High",   "Soil": "Depleting", "Carbon": "High"},
    "Sugarcane":{"Water": 2200, "Fertilizer": "High",   "Soil": "Depleting", "Carbon": "High"},
    "Maize":    {"Water": 1000, "Fertilizer": "Medium", "Soil": "Neutral",   "Carbon": "Medium"},
    "Jute":     {"Water": 1200, "Fertilizer": "Medium", "Soil": "Neutral",   "Carbon": "Medium"},
}

# ----------------------------
# Utilities: sustainability scoring
# ----------------------------
def calculate_sustainability(crop: str) -> int:
    c = SUSTAINABILITY_DATA.get(crop)
    if not c:
        return 0
    score = 0
    # Water score (1-5)
    w = c["Water"]
    if w < 600:
        score += 5
    elif w < 1200:
        score += 4
    elif w < 2000:
        score += 3
    elif w < 3000:
        score += 2
    else:
        score += 1
    # Fertilizer
    fert_map = {"Low":5, "Medium":3, "High":1}
    score += fert_map.get(c["Fertilizer"], 0)
    # Soil
    soil_map = {"Enriching":5, "Neutral":3, "Depleting":1}
    score += soil_map.get(c["Soil"], 0)
    # Carbon
    carb_map = {"Low":5, "Medium":3, "High":1}
    score += carb_map.get(c["Carbon"], 0)
    return score  # 4 factors, each 1-5 -> total 4-20

# ----------------------------
# Model training: encode & scale
# ----------------------------
LE = LabelEncoder()
DF = DF.copy()
DF["Crop_encoded"] = LE.fit_transform(DF["Crop"])
FEATURES = ["Rainfall", "Temperature", "Soil_pH"]
X = DF[FEATURES].values.astype(float)
y = DF["Crop_encoded"].values
SCALER = StandardScaler().fit(X)
X_SCALED = SCALER.transform(X)
KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(X_SCALED, y)

# ----------------------------
# Caching external calls (weather & soil)
# ----------------------------
@st.cache_data(ttl=60*30)
def fetch_weather(lat: float, lon: float) -> Optional[Tuple[float, float]]:
    """Return (temp_celsius, recent_24h_rainfall_mm) or None on failure"""
    if not OPENWEATHER_API_KEY:
        return None
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        )
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json()
        temp = data.get("current", {}).get("temp")
        rainfall = 0.0
        for h in data.get("hourly", [])[:24]:
            rainfall += h.get("rain", {}).get("1h", 0.0)
        return (temp, rainfall)
    except Exception:
        return None

@st.cache_data(ttl=60*60)
def fetch_soil_ph(lat: float, lon: float) -> Optional[float]:
    """Query SoilGrids for phh2o top layer (approx)"""
    try:
        params = {"lat": lat, "lon": lon, "property": "phh2o"}
        r = requests.get(SOILGRIDS_BASE, params=params, timeout=8)
        r.raise_for_status()
        j = r.json()
        layers = j.get("properties", {}).get("phh2o", {}).get("layers")
        if layers and len(layers) > 0:
            # each layer has values list; pick the first value safely
            values = layers[0].get("values")
            if values and len(values) > 0:
                val = values[0].get("value")
                if val is not None:
                    return float(val)
    except Exception:
        return None
    return None

# ----------------------------
# Helper: district fallback database (small dataframe)
# ----------------------------
DISTRICT_DB = DF[["District","Rainfall","Temperature","Soil_pH"]].copy()

# ----------------------------
# App UI
# ----------------------------
st.set_page_config(page_title="🌱 Smart Crop Advisor", page_icon="🌾", layout="wide")
st.title("🌱 Smart Crop Advisor — Yield, Profit & Sustainability")
st.markdown("Use location or enter values to get crop recommendations with sustainability insights.")

# Sidebar: settings
with st.sidebar:
    st.header("Settings")
    n_neighbors = st.slider("Nearest neighbors (K)", 1, 7, 3)
    weight_profit = st.slider("Weight: Profit", 0.0, 1.0, 0.6)
    weight_sust = st.slider("Weight: Sustainability", 0.0, 1.0, 0.4)
    if abs((weight_profit + weight_sust) - 1.0) > 1e-6:
        st.caption("Weights are applied and normalized internally; adjust sliders to express preference.")
    st.markdown("---")
    st.markdown("*API status*")
    st.write(f"OpenWeather key: {'✔️' if OPENWEATHER_API_KEY else '❌ (not set)'}")
    st.write(f"Geolocation available: {'✔️' if GEO_AVAILABLE else '❌ (component missing)'}")

# main input area
col1, col2 = st.columns([1,2])
with col1:
    input_mode = st.radio("Choose input type:", ["📍 Use My Location (Browser)", "📌 Select District", "✍️ Manual Input"]) 
    lat = lon = None
    rainfall = None
    temp = None
    soil_ph = None

    if input_mode == "📍 Use My Location (Browser)":
        if not GEO_AVAILABLE:
            st.warning("Browser geolocation component not available. Install streamlit-js-eval for GPS support.")
        else:
            loc = get_geolocation()
            if loc:
                # streamlit-js-eval returns a dict like {'coords': {'latitude':..., 'longitude':...}}
                coords = loc.get("coords", {}) if isinstance(loc, dict) else {}
                lat = coords.get("latitude")
                lon = coords.get("longitude")
                if lat is not None and lon is not None:
                    st.success(f"Detected location: {lat:.5f}, {lon:.5f}")
                    weather = fetch_weather(lat, lon)
                    soil = fetch_soil_ph(lat, lon)
                    if weather:
                        temp, rainfall = weather
                    if soil is not None:
                        soil_ph = soil
                    # if any missing, fallback to nearest district
                    if any(v is None for v in (rainfall, temp, soil_ph)):
                        st.info("Some live data missing; falling back to nearest district averages.")
                        st.caption("Please also choose a district below as a fallback (used only if needed).")
                else:
                    st.info("Could not parse coordinates from browser geolocation.")
            else:
                st.info("Allow location access in the browser to use this feature.")

    if input_mode == "📌 Select District":
        district = st.selectbox("Choose your District", DISTRICT_DB["District"].unique())
        row = DISTRICT_DB[DISTRICT_DB["District"] == district].iloc[0]
        rainfall = float(row["Rainfall"])
        temp = float(row["Temperature"])
        soil_ph = float(row["Soil_pH"])

    if input_mode == "✍️ Manual Input":
        rainfall = float(st.number_input("🌧️ Average Rainfall (mm)", min_value=0.0, max_value=10000.0, value=1000.0, step=1.0))
        temp = float(st.number_input("🌡️ Average Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0, step=0.1))
        soil_ph = float(st.number_input("🧪 Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1))

with col2:
    st.write("### Preview: Inputs used for recommendation")
    placeholder = st.empty()
    def show_preview(rf, t, ph):
        placeholder.markdown(f"""- 🌧️ Rainfall: *{rf if rf is not None else '—'} mm*  
- 🌡️ Temperature: *{t if t is not None else '—'} °C*  
- 🧪 Soil pH: *{ph if ph is not None else '—'}*""")

    show_preview(rainfall, temp, soil_ph)

# If using location and some values missing, allow district fallback
if input_mode == "📍 Use My Location (Browser)":
    if (rainfall is None or temp is None or soil_ph is None):
        st.info("Please select a fallback district (used only if live data missing).")
        district_fb = st.selectbox("Fallback district", DISTRICT_DB["District"].unique(), key="fb")
        row = DISTRICT_DB[DISTRICT_DB["District"] == district_fb].iloc[0]
        if rainfall is None:
            rainfall = float(row["Rainfall"]) if not pd.isna(row["Rainfall"]) else rainfall
        if temp is None:
            temp = float(row["Temperature"]) if not pd.isna(row["Temperature"]) else temp
        if soil_ph is None:
            soil_ph = float(row["Soil_pH"]) if not pd.isna(row["Soil_pH"]) else soil_ph

# Final guard: require inputs
if any(v is None for v in (rainfall, temp, soil_ph)):
    st.warning("Please provide or fetch all three inputs (Rainfall, Temperature, Soil pH) to get a recommendation.")
else:
    if st.button("✅ Get Crop Recommendation"):
        # prepare input, scale and predict
        input_arr = np.array([[float(rainfall), float(temp), float(soil_ph)]], dtype=float)
        input_scaled = SCALER.transform(input_arr)

        # retrain knn with chosen neighbors
        knn_local = KNeighborsClassifier(n_neighbors=max(1, int(n_neighbors)))
        knn_local.fit(X_SCALED, y)

        pred_encoded = knn_local.predict(input_scaled)[0]
        best_crop = LE.inverse_transform([pred_encoded])[0]

        # find representative row for best crop (first occurrence)
        best_row = DF[DF["Crop"] == best_crop].iloc[0]

        # sustainability
        best_sust = calculate_sustainability(best_crop)

        # Display main recommendation
        st.markdown("---")
        header_col, metric_col = st.columns([3,2])
        with header_col:
            st.success(f"✅ Recommended Crop: {best_crop}")
            st.write(f"*Yield:* {best_row['Yield']} q/ha  ")
            st.write(f"*Profit:* ₹{int(best_row['Profit']):,}/ha  ")
            st.write(f"*Success Rate:* {best_row['SuccessRate']}%  ")
        with metric_col:
            st.metric("Sustainability Score", f"{best_sust}/20")
            sdata = SUSTAINABILITY_DATA.get(best_crop, {})
            st.write(f"💧 {sdata.get('Water','—')} L/kg")
            st.write(f"🧪 {sdata.get('Fertilizer','—')}")

        # Neighbors and alternatives
        neigh_idx = knn_local.kneighbors(input_scaled, return_distance=False)[0]
        neighbor_rows = DF.iloc[neigh_idx].reset_index(drop=True)

        alt_candidates = neighbor_rows[neighbor_rows['Crop'] != best_crop].drop_duplicates('Crop')
        # build alternatives info
        alternatives_info = []
        for _, r in alt_candidates.iterrows():
            alt = r['Crop']
            alt_sust = calculate_sustainability(alt)
            alternatives_info.append({
                'crop': alt,
                'Yield_qph': r['Yield'],
                'Profit_inr': r['Profit'],
                'Sustainability': alt_sust
                   })

        if alternatives_info:
            st.subheader("🌱 Alternative Options")
            for i, ai in enumerate(alternatives_info, start=1):
                st.info(f"{i}. {ai['crop']} — Yield: {ai['Yield_qph']} q/ha | Profit: ₹{int(ai['Profit_inr']):,}/ha | Sustainability: {ai['Sustainability']}/20")

        # Combine profit & sustainability ranking
        # Build dataframe of all unique crops in dataset and score them
        crops_unique = DF[['Crop','Yield','Profit']].drop_duplicates('Crop').reset_index(drop=True)
        crops_unique['Sustainability'] = crops_unique['Crop'].apply(calculate_sustainability)
        # normalize and compute combined score
        prof = crops_unique['Profit'].astype(float).values
        sust = crops_unique['Sustainability'].astype(float).values
        prof_n = (prof - prof.min()) / (prof.max() - prof.min() + 1e-9)
        sust_n = (sust - sust.min()) / (sust.max() - sust.min() + 1e-9)
        # normalize weights to sum 1
        wsum = weight_profit + weight_sust
        if wsum <= 0:
            w_profit_norm = 0.6
            w_sust_norm = 0.4
        else:
            w_profit_norm = weight_profit / wsum
            w_sust_norm = weight_sust / wsum
        combined = w_profit_norm * prof_n + w_sust_norm * sust_n
        crops_unique['CombinedScore'] = combined
        crops_sorted = crops_unique.sort_values('CombinedScore', ascending=False).reset_index(drop=True)

        st.subheader("🔀 Combined Ranking (Profit vs Sustainability)")
        st.table(crops_sorted[['Crop','Yield','Profit','Sustainability','CombinedScore']].assign(Profit=lambda df: df['Profit'].apply(lambda x: f"₹{int(x):,}")))

        # Visualization: bar chart comparing Best + Alternatives
        viz_crops = [{'crop': best_crop,'Yield_qph': best_row['Yield'],'Profit_inr': best_row['Profit'], 'Sustainability': best_sust}]
        viz_crops.extend(alternatives_info[:3])  # up to 3 alternatives
        viz_df = pd.DataFrame(viz_crops)

        st.subheader("📊 Comparison Chart")
        # melt for plotting
        plot_df = viz_df.melt(id_vars=['crop'], value_vars=['Yield_qph','Profit_inr','Sustainability'], var_name='metric', value_name='value')
        # scale Profit for visualization: show in thousands to keep bars comparable
        plot_df.loc[plot_df['metric']=='Profit_inr','value'] = plot_df.loc[plot_df['metric']=='Profit_inr','value'] / 1000.0
        # annotate axis label
        chart = alt.Chart(plot_df).mark_bar().encode(
            x=alt.X('crop:N', title='Crop'),
            y=alt.Y('value:Q', title='Metric (Yield q/ha | Profit ₹000s | Sustainability score)'),
            color='metric:N',
            tooltip=['crop','metric','value']
        ).properties(width=700, height=400)
        st.altair_chart(chart, use_container_width=True)

        # Offer JSON export of recommendation
        st.download_button("📥 Download recommendation (JSON)", data=viz_df.to_json(orient='records', indent=2), file_name='recommendation.json', mime='application/json')

        st.success("Recommendation generated — good luck! 🌾")

# Footer / help
st.markdown("---")
st.caption("Notes: 1) Make sure OPENWEATHER_API_KEY is set in your .env or deployment secrets to enable live weather. 2) If geolocation isn't working, select district or use manual input.")


# End of file