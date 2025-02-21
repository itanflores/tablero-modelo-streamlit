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
st.set_page_config(page_title="Tablero de Clasificación en Streamlit para la Gestión Predictiva de Infraestructura TI", 
                   page_icon="📊", 
                   layout="wide")

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
columnas_excluir = ["Estado del Sistema", "Estado del Sistema Codificado", "Fecha"]
X = df.drop(columns=columnas_excluir, errors="ignore")

# 📌 Asegurar que solo quedan variables numéricas
X = X.select_dtypes(include=[np.number])

# 📌 Aplicar StandardScaler solo a variables numéricas
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 📌 Dividir datos
y = df["Estado del Sistema Codificado"]
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

# 📌 Importancia de variables
df_importance = pd.DataFrame({
    "Variable": X.columns,
    "Importancia": model.feature_importances_
}).sort_values(by="Importancia", ascending=False)

# 📌 Mostrar solo las 10 más importantes
top_n = 10
df_importance_top = df_importance.head(top_n)

# 📌 Agregar color para resaltar la variable más importante
df_importance_top["Color"] = ["red" if i == 0 else "blue" for i in range(len(df_importance_top))]

fig_imp = px.bar(df_importance_top, 
                 x="Importancia", 
                 y="Variable", 
                 orientation='h', 
                 title="📊 Importancia de Variables",
                 color="Color",  
                 color_discrete_map={"red": "red", "blue": "blue"})

# 📌 Mejorar visualización
fig_imp.update_layout(
    xaxis_tickangle=-45,   # Rotar etiquetas
    xaxis_type="log",      # Usar escala logarítmica si hay mucha diferencia
    yaxis=dict(categoryorder="total ascending")  # Ordenar de menor a mayor
)

# 📌 Mostrar gráfico en Streamlit
st.plotly_chart(fig_imp, use_container_width=True)

st.divider()

st.success("✅ Datos cargados correctamente desde GCP")
