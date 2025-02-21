import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
from google.cloud import storage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize

# 📌 Configuración del Cliente de Google Cloud Storage
BUCKET_NAME = "monitoreo_gcp_bucket"
ARCHIVO_DATOS = "dataset_monitoreo_servers.csv"

# Inicializar cliente de Google Cloud Storage
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

# 📌 Cargar Dataset desde GCP
@st.cache_data
def cargar_datos():
    try:
        blob = bucket.blob(ARCHIVO_DATOS)
        contenido = blob.download_as_text()
        df = pd.read_csv(StringIO(contenido), encoding="utf-8")
        return df
    except Exception as e:
        st.error(f"❌ Error al descargar el dataset desde GCP: {e}")
        return None

df = cargar_datos()
if df is None:
    st.stop()

df.columns = df.columns.str.strip()  # Limpiar nombres de columnas

# 📌 Verificar columna objetivo
if "Estado del Sistema" in df.columns:
    df['Estado del Sistema Codificado'] = df['Estado del Sistema'].map({"Inactivo": 0, "Normal": 1, "Advertencia": 2, "Crítico": 3})
else:
    st.error("❌ Error: La columna 'Estado del Sistema' no se encuentra.")
    st.stop()

# 📌 Preprocesamiento de Datos
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Convertir "Fecha" a datetime (si existe)
if "Fecha" in df.columns:
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")

# Normalizar y codificar variables
columnas_excluir = ["Estado del Sistema", "Estado del Sistema Codificado", "Fecha", "Hostname"]
X = df.drop(columns=columnas_excluir, errors="ignore")
X = X.select_dtypes(include=[np.number])  # Asegurar que solo haya variables numéricas

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

y = df["Estado del Sistema Codificado"]

# 📌 Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 📌 Entrenamiento del Modelo Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 📌 **Sección 1: Evaluación General del Modelo**
st.title("📊 Tablero de Clasificación en Streamlit para la Gestión Predictiva de Infraestructura TI")
st.header("📌 Evaluación General del Modelo")

col1, col2, col3 = st.columns([1.5, 2, 2])

with col1:
    st.metric("📊 Precisión del Modelo", f"{accuracy_score(y_test, y_pred):.4f}")

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

# 🔹 Inicializar `st.session_state`
if "tree_trained" not in st.session_state:
    st.session_state["tree_trained"] = False
if "tree_acc" not in st.session_state:
    st.session_state["tree_acc"] = None

# 🔹 Pestañas de modelos
tab1, tab2, tab3 = st.tabs(["🌳 Árbol de Decisión", "📈 Regresión Logística", "🌲 Random Forest"])

with tab1:
    st.subheader("🌳 Árbol de Decisión")

    if st.button("Entrenar Árbol de Decisión"):
        with st.spinner("Entrenando..."):
            from sklearn.tree import DecisionTreeClassifier
            tree_clf = DecisionTreeClassifier(max_depth=5)
            tree_clf.fit(X_train, y_train)
            st.session_state["tree_acc"] = accuracy_score(y_test, tree_clf.predict(X_test))
            st.session_state["tree_trained"] = True

    if st.session_state.get("tree_trained", False):
        st.metric("Precisión", f"{st.session_state['tree_acc']:.4f}")

st.divider()

# 📌 **Curva ROC y AUC**
st.header("📈 Curva ROC y AUC")

y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
y_pred_proba = model.predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(y_test_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fig_roc = px.line(title="Curva ROC Multiclase")
for i in range(y_test_bin.shape[1]):
    fig_roc.add_scatter(x=fpr[i], y=tpr[i], mode='lines', name=f'Clase {i} (AUC = {roc_auc[i]:.2f})')

fig_roc.add_scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Clasificador Aleatorio')

st.plotly_chart(fig_roc, use_container_width=True)

st.success("✅ Datos cargados correctamente desde GCP")
