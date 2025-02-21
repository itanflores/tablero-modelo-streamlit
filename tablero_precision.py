import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
from google.cloud import storage
from sklearn.preprocessing import MinMaxScaler  # ✅ Importado correctamente
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 📌 Configuración del Cliente de Google Cloud Storage (Solo se mantiene una vez)
BUCKET_NAME = "monitoreo_gcp_bucket"
ARCHIVO_DATOS = "dataset_monitoreo_servers.csv"
ARCHIVO_PROCESADO = "dataset_procesado.csv"

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

# 📌 Preprocesamiento de Datos
df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Codificación ordinal para "Estado del Sistema"
estado_mapping = {"Inactivo": 0, "Normal": 1, "Advertencia": 2, "Crítico": 3}
df["Estado del Sistema Codificado"] = df["Estado del Sistema"].map(estado_mapping)

# Codificación one-hot para "Tipo de Servidor"
df = pd.get_dummies(df, columns=["Tipo de Servidor"], prefix="Servidor", drop_first=True)

# Normalización de métricas continuas
scaler = MinMaxScaler()
metricas_continuas = ["Uso CPU (%)", "Temperatura (°C)", "Carga de Red (MB/s)", "Latencia Red (ms)"]
df[metricas_continuas] = scaler.fit_transform(df[metricas_continuas])

# 📌 División en conjunto de entrenamiento y prueba
X = df.drop(["Estado del Sistema", "Estado del Sistema Codificado", "Fecha", "Hostname"], axis=1, errors="ignore")
y = df["Estado del Sistema Codificado"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 📌 Entrenamiento del Modelo Random Forest
rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 📌 Predicción y Evaluación del Modelo
y_pred = rf_model.predict(X_test)
precision_modelo = accuracy_score(y_test, y_pred)

# 📌 Visualización en Streamlit
st.title("📊 Monitoreo de Servidores - GCP")

st.sidebar.header("Filtros")
estados_seleccionados = st.sidebar.multiselect("Selecciona Estados:", df["Estado del Sistema"].unique(), default=df["Estado del Sistema"].unique())
df_filtrado = df[df["Estado del Sistema"].isin(estados_seleccionados)]

if df_filtrado.empty:
    st.warning("⚠ No hay datos disponibles con los filtros seleccionados.")
    st.stop()

# 📌 Gráficos en Streamlit
st.subheader("📈 Evolución del Estado del Sistema")
df_grouped = df_filtrado.groupby(["Fecha", "Estado del Sistema"]).size().reset_index(name="Cantidad")
st.line_chart(df_grouped.pivot(index="Fecha", columns="Estado del Sistema", values="Cantidad").fillna(0))

st.subheader("🌡 Distribución de Temperatura por Estado")
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=df["Estado del Sistema"], y=df["Temperatura (°C)"], ax=ax)
st.pyplot(fig)

st.subheader("📊 Importancia de Variables en el Modelo")
feature_importances = pd.DataFrame({"Variable": X_train.columns, "Importancia": rf_model.feature_importances_}).sort_values(by="Importancia", ascending=False)
st.bar_chart(feature_importances.set_index("Variable"))

st.subheader("✅ Precisión del Modelo")
st.metric(label="Precisión del Modelo Random Forest", value=f"{precision_modelo:.2%}")

# 📌 SECCIÓN DE COMPARACIÓN DE MODELOS (se mantiene igual)
st.header("📊 Comparación de Modelos de Clasificación")

tab1, tab2, tab3 = st.tabs(["🌳 Árbol de Decisión", "📈 Regresión Logística", "🌲 Random Forest"])

with tab1:
    st.subheader("🌳 Árbol de Decisión")
    if st.button("Entrenar Árbol de Decisión"):
        from sklearn.tree import DecisionTreeClassifier
        tree_clf = DecisionTreeClassifier(max_depth=5)
        tree_clf.fit(X_train, y_train)
        acc_tree = accuracy_score(y_test, tree_clf.predict(X_test))
        st.metric("Precisión", f"{acc_tree:.4f}")

with tab2:
    st.subheader("📈 Regresión Logística")
    if st.button("Entrenar Regresión Logística"):
        from sklearn.linear_model import LogisticRegression
        log_clf = LogisticRegression(max_iter=50, n_jobs=-1)
        log_clf.fit(X_train, y_train)
        acc_log = accuracy_score(y_test, log_clf.predict(X_test))
        st.metric("Precisión", f"{acc_log:.4f}")

with tab3:
    st.subheader("🌲 Random Forest")
    if st.button("Entrenar Random Forest"):
        forest_clf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
        forest_clf.fit(X_train, y_train)
        acc_forest = accuracy_score(y_test, forest_clf.predict(X_test))
        st.metric("Precisión", f"{acc_forest:.4f}")

# 📌 NUEVA SECCIÓN: PROCESAMIENTO Y EXPORTACIÓN DE DATOS A GCP
if "datos_procesados" not in st.session_state:
    st.session_state["datos_procesados"] = pd.DataFrame()

if st.button("⚙️ Procesar Datos"):
    df_procesado = df.copy()
    st.session_state["datos_procesados"] = df_procesado
    st.success("✅ Datos procesados correctamente.")

if not st.session_state["datos_procesados"].empty:
    def exportar_datos():
        try:
            df_procesado = st.session_state["datos_procesados"]
            blob_procesado = bucket.blob(ARCHIVO_PROCESADO)
            blob_procesado.upload_from_string(df_procesado.to_csv(index=False), content_type="text/csv")
            st.success(f"✅ Datos procesados exportados a {BUCKET_NAME}/{ARCHIVO_PROCESADO}")
        except Exception as e:
            st.error(f"❌ Error al exportar datos a GCP: {e}")

    if st.button("📤 Guardar Datos Procesados en GCP"):
        exportar_datos()
