import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from google.cloud import storage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import io

# 📌 Configuración de la página
st.set_page_config(page_title="Tablero de Clasificación en Streamlit para la Gestión Predictiva de Infraestructura TI", page_icon="📊", layout="wide")

# 📌 Título
st.title("📊 Tablero de Clasificación en Streamlit para la Gestión Predictiva de Infraestructura TI")

# 📌 Configuración de GCP Storage
BUCKET_NAME = "monitoreo_gcp_bucket"
FILE_NAME = "dataset_monitoreo_servers.csv"

# Inicializar cliente de Google Cloud Storage
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

# 📌 Cargar Dataset desde GCP
try:
    blob = bucket.blob(FILE_NAME)
    contenido = blob.download_as_text()
    df = pd.read_csv(io.StringIO(contenido), encoding="utf-8")
    df.columns = df.columns.str.strip()
except Exception as e:
    st.error(f"❌ Error al descargar el dataset desde GCP: {e}")
    st.stop()

# 📌 Verificar columna objetivo
if "Estado del Sistema" in df.columns:
    df['Estado del Sistema Codificado'] = df['Estado del Sistema'].map({"Inactivo": 0, "Normal": 1, "Advertencia": 2, "Crítico": 3})
else:
    st.error("❌ Error: La columna 'Estado del Sistema' no se encuentra.")
    st.stop()

# 📌 Preprocesamiento
X = df.drop(["Estado del Sistema", "Estado del Sistema Codificado"], axis=1)
y = df["Estado del Sistema Codificado"]
X = pd.get_dummies(X, drop_first=True)
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 📌 Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 📌 Entrenamiento del modelo
model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 📌 **Sección 1: Evaluación General del Modelo**
st.header("📌 Evaluación General del Modelo")

col1, col2, col3 = st.columns([1.5, 2, 2])

with col1:
    st.metric("📊 Precisión del Modelo", f"{accuracy_score(y_test, y_pred):.4f}")
    st.caption("🔹 La precisión mide la proporción de predicciones correctas. Valores más altos indican mejor desempeño.")

with col2.expander("📋 Reporte de Clasificación"):
    st.text(classification_report(y_test, y_pred))

with col3:
    st.write("📊 Matriz de Confusión")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    st.pyplot(fig)

st.divider()

# 📌 **Sección 2: Importancia de Variables**
st.header("📊 Importancia de Variables en la Predicción")

df_importance = pd.DataFrame({
    "Variable": X.columns,
    "Importancia": model.feature_importances_
}).sort_values(by="Importancia", ascending=False)

st.bar_chart(df_importance.set_index("Variable"))

st.divider()

# 📌 **Sección 3: Comparación de Modelos**
st.header("📊 Comparación de Modelos de Clasificación")

# 🔹 Pestañas para organizar cada modelo
tab1, tab2, tab3 = st.tabs(["🌳 Árbol de Decisión", "📈 Regresión Logística", "🌲 Random Forest"])

# 📌 Sección Árbol de Decisión en `st.tabs()`
with tab1:
    st.subheader("🌳 Árbol de Decisión")
    if st.button("Entrenar Árbol de Decisión"):
        with st.spinner("Entrenando..."):
            from sklearn.tree import DecisionTreeClassifier
            tree_clf = DecisionTreeClassifier(max_depth=5)
            tree_clf.fit(X_train, y_train)
            st.session_state["modelo_entrenado"] = tree_clf
            st.session_state["tree_acc"] = accuracy_score(y_test, tree_clf.predict(X_test))
            st.session_state["tree_trained"] = True

    if st.session_state.get("tree_trained", False):
        st.metric("Precisión", f"{st.session_state['tree_acc']:.4f}")

# 📈 **Regresión Logística**
with tab2:
    st.subheader("📈 Regresión Logística")
    if st.button("Entrenar Regresión Logística"):
        with st.spinner("Entrenando..."):
            from sklearn.linear_model import LogisticRegression
            log_clf = LogisticRegression(max_iter=50, n_jobs=-1)
            log_clf.fit(X_train, y_train)
            acc_log = accuracy_score(y_test, log_clf.predict(X_test))
            st.metric("Precisión", f"{acc_log:.4f}")

# 🌲 **Random Forest**
with tab3:
    st.subheader("🌲 Random Forest")
    if st.button("Entrenar Random Forest"):
        with st.spinner("Entrenando..."):
            forest_clf = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                max_samples=0.5,
                n_jobs=-1,
                random_state=42
            )
            forest_clf.fit(X_train, y_train)
            acc_forest = accuracy_score(y_test, forest_clf.predict(X_test))
            st.metric("Precisión", f"{acc_forest:.4f}")

st.success("✅ Datos cargados correctamente desde GCP")
