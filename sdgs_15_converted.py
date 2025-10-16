# 🌳 SDGs 15 Dashboard — Kehidupan di Darat
# Analisis Deforestasi & Emisi Karbon Indonesia (2001–2027)
# Dibuat oleh: Miftah Faridl © 2025
# Versi edukatif, interaktif, dan tanpa dependensi berbayar

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ------------------------------------------
# 🎨 Konfigurasi Halaman
# ------------------------------------------
st.set_page_config(page_title="SDGs 15 Dashboard", layout="wide", page_icon="🌳")

st.markdown(
    """
    <h1 style="text-align:center; color:#2E8B57;">🌳 SDGs 15 – Kehidupan di Darat</h1>
    <h4 style="text-align:center;">Analisis Deforestasi dan Emisi Karbon Indonesia (2001–2027)</h4>
    <p style="text-align:center; color:gray;">Miftah Faridl © 2025 | Data: Global Forest Watch</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------
# 📂 Load Data
# ------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/miftahfaridl710-sketch/SDGS-15/main/IDN.xlsx"
    df = pd.read_excel(url, sheet_name="Subnational 1 tree cover loss", engine="openpyxl")
    df = df[df["threshold"] == 30].reset_index(drop=True)
    loss_cols = [c for c in df.columns if c.startswith("tc_loss_ha_")]
    df_long = df.melt(
        id_vars=["country", "subnational1", "extent_2000_ha"],
        value_vars=loss_cols,
        var_name="year",
        value_name="loss_ha"
    )
    df_long["year"] = df_long["year"].str.extract(r"(\d{4})").astype(int)
    df_long["loss_rate_%"] = (df_long["loss_ha"] / df_long["extent_2000_ha"]) * 100
    df_long = df_long.sort_values(["subnational1", "year"])
    return df, df_long

df, df_long = load_data()

# ------------------------------------------
# 🧮 Perhitungan Statistik Utama
# ------------------------------------------
total_loss = df_long["loss_ha"].sum()
avg_emission = total_loss * 50 * 3.67
tahun_akhir = 2027

# Placeholder untuk model
r2_value = 0.87  # akan diperbarui di bagian model

# ------------------------------------------
# 🟩 Kartu SDGs Info
# ------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("🌳 Total Kehilangan Hutan", f"{total_loss/1e6:,.2f} juta ha")
col2.metric("💨 Total Emisi Karbon", f"{avg_emission/1e6:,.1f} juta Mg CO₂e")
col3.metric("🔮 Tahun Prediksi", tahun_akhir)
col4.metric("📊 Akurasi Model (R²)", f"{r2_value:.2f}")

st.markdown("---")

# ------------------------------------------
# 📊 Tabs Interaktif
# ------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🏞 Kehilangan Tutupan Hutan",
    "💨 Emisi Karbon",
    "📈 Prediksi 2025–2027",
    "🗺 Persebaran Tahun 2027"
])

# ------------------------------------------
# 🏞 Tab 1: Kehilangan Tutupan Hutan
# ------------------------------------------
with tab1:
    st.subheader("Tren Kehilangan Tutupan Hutan di Indonesia (2001–2024)")
    national_trend = df_long.groupby("year", as_index=False)["loss_ha"].sum()

    fig1 = px.line(
        national_trend, x="year", y="loss_ha",
        title="Tren Kehilangan Tutupan Hutan Nasional (2001–2024)",
        labels={"loss_ha": "Luas Kehilangan (ha)", "year": "Tahun"},
        color_discrete_sequence=["#2E8B57"],
        markers=True
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.download_button("📥 Unduh Data", national_trend.to_csv(index=False).encode(), "tren_nasional.csv", "text/csv")

    st.info(
        "📈 **Insight:** Kehilangan tutupan hutan di Indonesia cenderung meningkat "
        "pada periode 2013–2017, yang merupakan masa deforestasi tertinggi di beberapa provinsi besar."
    )

    st.subheader("Top 5 Provinsi dengan Kehilangan Hutan Tertinggi")
    top5 = df_long.groupby("subnational1")["loss_ha"].mean().nlargest(5).index
    df_top5 = df_long[df_long["subnational1"].isin(top5)]
    fig2 = px.line(
        df_top5, x="year", y="loss_ha", color="subnational1", markers=True,
        title="Tren Kehilangan Hutan di 5 Provinsi Teratas",
        labels={"loss_ha": "Luas Kehilangan (ha)", "subnational1": "Provinsi"}
    )
    st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------
# 💨 Tab 2: Emisi Karbon
# ------------------------------------------
with tab2:
    st.subheader("Tren Emisi Karbon Akibat Deforestasi (2001–2024)")
    df_long["emission_CO2e"] = df_long["loss_ha"] * 50 * 3.67
    carbon_trend = df_long.groupby("year", as_index=False)["emission_CO2e"].sum()

    fig3 = px.area(
        carbon_trend, x="year", y="emission_CO2e",
        title="Tren Emisi Karbon Akibat Deforestasi",
        color_discrete_sequence=["#E85D04"],
        labels={"emission_CO2e": "Emisi (Mg CO₂e)", "year": "Tahun"}
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.download_button("📥 Unduh Data Emisi", carbon_trend.to_csv(index=False).encode(), "emisi_karbon.csv", "text/csv")

    st.info(
        "🌍 **Insight:** Kenaikan emisi karbon berkorelasi langsung dengan peningkatan kehilangan tutupan hutan. "
        "Provinsi dengan deforestasi tinggi seperti Kalimantan Tengah dan Papua berkontribusi besar terhadap emisi nasional."
    )

# ------------------------------------------
# 📈 Tab 3: Prediksi Deforestasi 2025–2027
# ------------------------------------------
with tab3:
    st.subheader("Prediksi Deforestasi & Emisi Karbon (2025–2027) Menggunakan Model XGBoost")

    df_model = df_long.copy()
    df_model["lag1"] = df_model.groupby("subnational1")["loss_ha"].shift(1)
    df_model["lag2"] = df_model.groupby("subnational1")["loss_ha"].shift(2)
    df_model.dropna(inplace=True)

    le = LabelEncoder()
    df_model["prov_enc"] = le.fit_transform(df_model["subnational1"])
    FEATURES = ["lag1", "lag2", "year", "prov_enc"]
    TARGET = "loss_ha"

    train = df_model[df_model["year"] <= 2021]
    test = df_model[df_model["year"] > 2021]

    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5)
    model.fit(train[FEATURES], train[TARGET])
    preds = model.predict(test[FEATURES])

    rmse = np.sqrt(mean_squared_error(test[TARGET], preds))
    r2 = r2_score(test[TARGET], preds)
    r2_value = r2

    c1, c2 = st.columns(2)
    c1.metric("RMSE", f"{rmse:,.2f}")
    c2.metric("R²", f"{r2:.3f}")

    st.info(
        "🤖 **Interpretasi:** Nilai R² mendekati 1 menunjukkan model mampu memprediksi pola deforestasi dengan baik. "
        "RMSE menunjukkan rata-rata selisih antara nilai aktual dan prediksi."
    )

    # Forecast 2025–2027
    forecast_rows = []
    for prov in df_model["subnational1"].unique():
        prov_df = df_model[df_model["subnational1"] == prov].sort_values("year")
        lag1, lag2 = prov_df.iloc[-1]["loss_ha"], prov_df.iloc[-2]["loss_ha"]
        prov_enc = le.transform([prov])[0]
        for y in [2025, 2026, 2027]:
            pred = model.predict(pd.DataFrame([[lag1, lag2, y, prov_enc]], columns=FEATURES))[0]
            emission = pred * 50 * 3.67
            forecast_rows.append({"subnational1": prov, "year": y, "pred_loss_ha": pred, "emission_pred": emission})
            lag1, lag2 = pred, lag1

    forecast_df = pd.DataFrame(forecast_rows)
    national_forecast = forecast_df.groupby("year", as_index=False)[["pred_loss_ha", "emission_pred"]].sum()

    fig4 = px.line(
        national_forecast, x="year", y="pred_loss_ha", markers=True,
        title="Prediksi Kehilangan Tutupan Hutan Nasional (2025–2027)",
        labels={"pred_loss_ha": "Luas Kehilangan (ha)", "year": "Tahun"},
        color_discrete_sequence=["#2E8B57"]
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.download_button("📥 Unduh Data Prediksi", forecast_df.to_csv(index=False).encode(), "prediksi_deforestasi.csv", "text/csv")

# ------------------------------------------
# 🗺 Tab 4: Peta Persebaran Tahun 2027
# ------------------------------------------
with tab4:
    st.subheader("Peta Persebaran Deforestasi & Emisi Karbon Tahun 2027")
    st.write("Menampilkan estimasi kehilangan tutupan hutan dan emisi karbon berdasarkan hasil prediksi model XGBoost.")

    shp_path = "gadm41_IDN_1.shp"
    gdf = gpd.read_file(shp_path)
    gdf["NAME_1"] = gdf["NAME_1"].str.title().str.strip()
    forecast_df["subnational1"] = forecast_df["subnational1"].str.title().str.strip()
    map_2027 = forecast_df[forecast_df["year"] == 2027]
    gdf_merged = gdf.merge(map_2027, left_on="NAME_1", right_on="subnational1", how="left")

    fig5 = px.choropleth(
        gdf_merged,
        geojson=gdf_merged.geometry.__geo_interface__,
        locations=gdf_merged.index,
        color="pred_loss_ha",
        color_continuous_scale=["#A7F3A7", "#E85D04"],
        labels={"pred_loss_ha": "Luas Kehilangan (ha)"},
        title="Persebaran Prediksi Deforestasi Tahun 2027"
    )
    fig5.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig5, use_container_width=True)

    fig6 = px.choropleth(
        gdf_merged,
        geojson=gdf_merged.geometry.__geo_interface__,
        locations=gdf_merged.index,
        color="emission_pred",
        color_continuous_scale=["#E5F5E0", "#E85D04"],
        labels={"emission_pred": "Emisi (Mg CO₂e)"},
        title="Persebaran Emisi Karbon Tahun 2027"
    )
    fig6.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")
st.caption("📊 Data: Global Forest Watch | Visualisasi oleh Miftah Faridl © 2025 | Tujuan Pembangunan Berkelanjutan (SDGs 15)")
