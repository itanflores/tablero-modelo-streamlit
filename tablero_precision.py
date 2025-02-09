import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests
import io

# Configuración de la página
st.set_page_config(page_title="Tablero de Evaluación", page_icon="📊", layout="wide")

# 📌 Título
st.title("📊 Tablero de Evaluación del Modelo de Clasificación")

# Cargar Dataset
GITHUB_API_URL = "https://github.com/itanflores/tablero-modelo-streamlit/raw/main/dataset_monitoreo_servers.csv"
response = requests.get(GITHUB_API_URL, stream=True)

if response.status_code == 200:
    df = pd.read_csv(io.BytesIO(response.content), encoding="utf-8")
else:
    st.error("❌ Error al descargar el dataset.")
    st.stop()

df.columns = df.columns.str.strip()

# Verificar columna objetivo
if "Estado del Sistema" in df.columns:
    df['Estado del Sistema Codificado'] = df['Estado del Sistema'].map({"Inactivo": 0, "Normal": 1, "Advertencia": 2, "Crítico": 3})
else:
    st.error("❌ Error: La columna 'Estado del Sistema' no se encuentra.")
    st.stop()

# Preprocesamiento
X = df.drop(["Estado del Sistema", "Estado del Sistema Codificado"], axis=1)
y = df["Estado del Sistema Codificado"]
X = pd.get_dummies(X, drop_first=True)
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Entrenamiento del modelo
model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 📌 **Sección 1: Evaluación General del Modelo**
st.header("📌 Evaluación General del Modelo")

# 🔹 Reorganización con `st.columns()`
col1, col2, col3 = st.columns([1.5, 2, 2])

# 📊 **Métrica de Precisión** en `col1`
col1.metric("📊 Precisión del Modelo", f"{accuracy_score(y_test, y_pred):.4f}")

# 📋 **Reporte de Clasificación** en `col2`
with col2.expander("📋 Reporte de Clasificación"):
    st.text(classification_report(y_test, y_pred))

# 📊 **Matriz de Confusión** en `col3`
with col3:
    st.write("📊 Matriz de Confusión")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    st.pyplot(fig)

st.divider()

# 📌 **Sección 2: Importancia de Variables**
st.header("📊 Importancia de Variables en la Predicción")

# 🔹 Obtener la importancia de variables
df_importance = pd.DataFrame({
    "Variable": X.columns,
    "Importancia": model.feature_importances_
}).sort_values(by="Importancia", ascending=False)

# 📊 **Nueva distribución**
col4, col5 = st.columns([1.5, 3])

# 📋 **Tabla en un `expander`**
with col4.expander("📋 Ver Variables Importantes", expanded=True):
    st.dataframe(df_importance.head(10), height=300)

# 📊 **Gráfico de importancia en `col5`**
fig_imp = px.bar(df_importance.head(10), 
                 x="Importancia", 
                 y="Variable", 
                 orientation='h', 
                 title="📊 Importancia de Variables")

col5.plotly_chart(fig_imp, use_container_width=True)

st.divider()
