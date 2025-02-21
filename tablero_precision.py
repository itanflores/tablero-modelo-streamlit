import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from google.cloud import storage
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import os
import json

# 📌 Configuración del cliente de Google Cloud Storage
BUCKET_NAME = "monitoreo_gcp_bucket"
ARCHIVO_DATOS = "dataset_monitoreo_servers.csv"
# Inicializar cliente de Google Cloud Storage (usando autenticación predeterminada de GCP)
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)
bucket = storage_client.bucket(BUCKET_NAME)

# 📌 Descargar el dataset desde Google Cloud Storage
@st.cache_data
def cargar_datos():
    try:
        blob = bucket.blob(ARCHIVO_DATOS)
        contenido = blob.download_as_text()
        df = pd.read_csv(pd.compat.StringIO(contenido))
        return df
    except Exception as e:
        st.error(f"❌ Error al descargar el archivo desde GCP: {e}")
        return None

df = cargar_datos()
if df is None:
    st.stop()

# 📌 Convertir fechas y manejar valores nulos
df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
df.dropna(inplace=True)

# 📌 Dashboard de Streamlit
st.title("📊 Monitoreo de Servidores - GCP")
st.sidebar.header("Filtros")

# Filtros
estados_seleccionados = st.sidebar.multiselect("Selecciona Estados:", df["Estado del Sistema"].unique(), default=df["Estado del Sistema"].unique())
df_filtrado = df[df["Estado del Sistema"].isin(estados_seleccionados)]

if df_filtrado.empty:
    st.warning("⚠ No hay datos disponibles con los filtros seleccionados.")
    st.stop()

# 📌 Agrupar y suavizar datos
df_grouped = df_filtrado.groupby(["Fecha", "Estado del Sistema"]).size().reset_index(name="Cantidad")
try:
    df_grouped["Cantidad_Suavizada"] = df_grouped.groupby("Estado del Sistema")["Cantidad"].transform(lambda x: x.rolling(7, min_periods=1).mean())
except Exception as e:
    st.error(f"❌ Error al calcular 'Cantidad_Suavizada': {e}")
    st.stop()

# 📊 Gráfico de Evolución en el Tiempo
st.plotly_chart(
    px.line(df_grouped, x="Fecha", y="Cantidad_Suavizada", color="Estado del Sistema", title="📈 Evolución en el Tiempo", markers=True),
    use_container_width=True
)

# 📌 Predicción de Temperatura Crítica
st.subheader("🌡 Predicción de Temperatura Crítica")

if {"Uso CPU (%)", "Carga de Red (MB/s)", "Temperatura (°C)"}.issubset(df_filtrado.columns):
    df_temp = df_filtrado[["Fecha", "Uso CPU (%)", "Carga de Red (MB/s)", "Temperatura (°C)"]].dropna()
    
    if len(df_temp) >= 10:
        X = df_temp[["Uso CPU (%)", "Carga de Red (MB/s)"]]
        y = df_temp["Temperatura (°C)"]

        # 🔹 Normalizar datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 🔹 Entrenar modelo
        model_temp = RandomForestRegressor(n_estimators=100, random_state=42)
        model_temp.fit(X_scaled, y)

        # 🔹 Generar predicciones
        future_data = pd.DataFrame({
            "Uso CPU (%)": np.linspace(X["Uso CPU (%)"].min(), X["Uso CPU (%)"].max(), num=12),
            "Carga de Red (MB/s)": np.linspace(X["Carga de Red (MB/s)"].min(), X["Carga de Red (MB/s)"].max(), num=12)
        })

        future_data_scaled = scaler.transform(future_data)
        future_temp_pred = model_temp.predict(future_data_scaled)

        df_future_temp = pd.DataFrame({
            "Fecha": pd.date_range(start=df_temp["Fecha"].max(), periods=12, freq="M"),
            "Temperatura Predicha (°C)": future_temp_pred
        })

        st.plotly_chart(
            px.line(df_future_temp, x="Fecha", y="Temperatura Predicha (°C)", title="📈 Predicción de Temperatura Crítica", markers=True),
            use_container_width=True
        )
    else:
        st.warning("⚠ No hay suficientes datos para predecir la temperatura crítica.")
else:
    st.warning("⚠ No se encontraron las columnas necesarias para realizar la predicción.")

st.success("✅ Datos cargados correctamente desde GCP")
