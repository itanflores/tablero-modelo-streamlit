import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from google.cloud import storage
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import boto3  # 📌 Agregamos la librería para S3

# 📌 Configuración del Cliente de Google Cloud Storage
BUCKET_NAME = "monitoreo_gcp_bucket"
ARCHIVO_DATOS = "dataset_monitoreo_servers.csv"

# 📌 Configuración de AWS S3
S3_BUCKET_NAME = "tfm-monitoring-data"

# Diccionario con los nombres de los datasets procesados para cada modelo
ARCHIVOS_PROCESADOS = {
    "Árbol de Decisión": "dataset_procesado_arbol_decision.csv",
    "Regresión Logística": "dataset_procesado_regresion_logistica.csv",
    "Random Forest": "dataset_procesado_random_forest.csv"
}

# Inicializar cliente de Google Cloud Storage
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

# 📌 Función para cargar los datos desde GCP Storage
@st.cache_data
def cargar_datos():
    try:
        blob = bucket.blob(ARCHIVO_DATOS)
        contenido = blob.download_as_text()
        df = pd.read_csv(StringIO(contenido))
        return df
    except Exception as e:
        st.error(f"❌ Error al descargar el archivo desde GCP: {e}")
        return None

df = cargar_datos()
if df is None:
    st.stop()

# 📌 Función para procesar los datos (ahora se hace por modelo)
def procesar_datos(df, modelo):
    df_procesado = df.copy()

    # Convertir fecha
    df_procesado["Fecha"] = pd.to_datetime(df_procesado["Fecha"], errors="coerce")

    # Eliminar duplicados y valores nulos
    df_procesado.drop_duplicates(inplace=True)
    df_procesado.dropna(inplace=True)

    # Codificación ordinal para "Estado del Sistema"
    estado_mapping = {"Inactivo": 0, "Normal": 1, "Advertencia": 2, "Crítico": 3}
    df_procesado["Estado del Sistema Codificado"] = df_procesado["Estado del Sistema"].map(estado_mapping)

    # Codificación one-hot para "Tipo de Servidor"
    df_procesado = pd.get_dummies(df_procesado, columns=["Tipo de Servidor"], prefix="Servidor", drop_first=True)

    # Normalización de métricas continuas (según modelo)
    scaler = MinMaxScaler()
    metricas_continuas = ["Uso CPU (%)", "Temperatura (°C)", "Carga de Red (MB/s)", "Latencia Red (ms)"]

    # Aplicamos escalado diferente si es Regresión Logística
    if modelo == "Regresión Logística":
        df_procesado[metricas_continuas] = (df_procesado[metricas_continuas] - df_procesado[metricas_continuas].mean()) / df_procesado[metricas_continuas].std()
    else:
        df_procesado[metricas_continuas] = scaler.fit_transform(df_procesado[metricas_continuas])

    return df_procesado

# 📌 Estado de datos procesados
if "datos_procesados" not in st.session_state:
    st.session_state["datos_procesados"] = {}

# 📌 Función para transferir archivo desde GCP a AWS S3
def subir_a_s3(modelo):
    try:
        # Definir el nombre del archivo basado en el modelo
        archivo_salida = ARCHIVOS_PROCESADOS[modelo]
        S3_FILE_NAME = archivo_salida  # Mantiene el mismo nombre en S3
        
        # Inicializar cliente S3
        s3_client = boto3.client("s3")
        
        # Descargar el archivo desde GCP y subirlo a S3
        blob_procesado = bucket.blob(archivo_salida)
        contenido = blob_procesado.download_as_bytes()
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=S3_FILE_NAME, Body=contenido)

        st.success(f"✅ Datos de {modelo} enviados a S3: s3://{S3_BUCKET_NAME}/{S3_FILE_NAME}")
    except Exception as e:
        st.error(f"❌ Error al enviar datos a S3: {e}")

# 📌 SECCIÓN: COMPARACIÓN DE MODELOS
st.header("📊 Comparación de Modelos de Clasificación")

tab1, tab2, tab3 = st.tabs(["🌳 Árbol de Decisión", "📈 Regresión Logística", "🌲 Random Forest"])

for tab, modelo in zip([tab1, tab2, tab3], ARCHIVOS_PROCESADOS.keys()):
    with tab:
        st.subheader(f"{modelo}")

        if st.button(f"⚙️ Procesar Datos para {modelo}"):
            df_procesado = procesar_datos(df, modelo)
            st.session_state["datos_procesados"][modelo] = df_procesado
            st.success(f"✅ Datos procesados correctamente para {modelo}.")

        # 📌 Botón de exportación de datos a GCP
        if modelo in st.session_state["datos_procesados"]:
            def exportar_datos():
                try:
                    df_procesado = st.session_state["datos_procesados"][modelo]
                    archivo_salida = ARCHIVOS_PROCESADOS[modelo]
                    blob_procesado = bucket.blob(archivo_salida)
                    blob_procesado.upload_from_string(df_procesado.to_csv(index=False), content_type="text/csv")
                    st.success(f"✅ Datos procesados de {modelo} exportados a {BUCKET_NAME}/{archivo_salida}")
                except Exception as e:
                    st.error(f"❌ Error al exportar datos a GCP: {e}")

            if st.button(f"📤 Guardar Datos de {modelo} en GCP"):
                exportar_datos()

                # 📌 Botón para transferir a S3 (solo aparece si ya se exportó a GCP)
                if st.button(f"🚀 Enviar Datos de {modelo} a S3"):
                    subir_a_s3(modelo)
