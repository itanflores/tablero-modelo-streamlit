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
from time import sleep

# 🛠️ Configuración de la página
st.set_page_config(page_title="Tablero de Evaluación", page_icon="📊", layout="wide")

# 📌 Título principal del tablero
st.title("📊 Tablero de Evaluación del Modelo de Clasificación")

# 📌 Cargar Dataset desde GitHub con Git LFS
GITHUB_API_URL = "https://github.com/itanflores/tablero-modelo-streamlit/raw/main/dataset_monitoreo_servers.csv"
response = requests.get(GITHUB_API_URL, stream=True)

if response.status_code == 200:
    df = pd.read_csv(io.BytesIO(response.content), encoding="utf-8")
else:
    st.error("❌ Error al descargar el dataset desde GitHub.")
    st.stop()

df.columns = df.columns.str.strip()

# 📌 Verificar si la columna correcta existe
if "Estado del Sistema" in df.columns:
    df['Estado del Sistema Codificado'] = df['Estado del Sistema'].map({"Inactivo": 0, "Normal": 1, "Advertencia": 2, "Crítico": 3})
else:
    st.error("❌ Error: La columna 'Estado del Sistema' no se encuentra en el dataset.")
    st.stop()

# 📌 Preprocesamiento de Datos
X = df.drop(["Estado del Sistema", "Estado del Sistema Codificado"], axis=1)
y = df["Estado del Sistema Codificado"]

# Convertir datos categóricos en variables numéricas si existen
X = pd.get_dummies(X, drop_first=True)
X = X.reindex(columns=X.columns, fill_value=0)

# Normalizar los datos para mejorar la estabilidad del modelo
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Optimización: Reducir el tamaño del dataset
X = X.astype(np.float32)
y = y.astype(np.int8)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 🔹 Sección 1: Evaluación General del Modelo
st.header("📌 Evaluación General del Modelo")

col1, col2 = st.columns([1, 2])  # 📌 Mejor distribución de métricas y gráficos

# Entrenamiento del modelo
model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 📌 Precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
col1.metric("📊 Precisión del Modelo", f"{accuracy:.4f}")

# 📌 Reporte de Clasificación en un `st.expander()`
with col1.expander("📋 Reporte de Clasificación"):
    st.text(classification_report(y_test, y_pred))

# 📌 Matriz de Confusión en un `st.expander()`
with col2.expander("📊 Matriz de Confusión"):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    st.pyplot(fig)

st.divider()

# 🔹 Sección 2: Importancia de Variables
st.header("📊 Importancia de Variables en la Predicción")

# 📌 Usando columnas para organizar la importancia de variables
col3, col4 = st.columns([1, 3])

df_importance = pd.DataFrame({"Variable": X.columns, "Importancia": model.feature_importances_}).sort_values(by="Importancia", ascending=False)

# 📌 Mostrar tabla en `col3`
col3.dataframe(df_importance.head(10), use_container_width=True)

# 📌 Mostrar gráfico en `col4`
fig_imp = px.bar(df_importance.head(10), x="Importancia", y="Variable", orientation='h', title="📊 Importancia de Variables")
col4.plotly_chart(fig_imp, use_container_width=True)

st.divider()

# 🔹 Sección 3: Comparación de Modelos en `st.tabs()`
st.header("📊 Comparación de Modelos de Clasificación")
st.markdown("**Nota:** Modelos ordenados del más rápido al más lento.")

tab1, tab2, tab3 = st.tabs(["🌳 Árbol de Decisión", "📈 Regresión Logística", "🌲 Random Forest"])

model_scores = {}

with tab1:
    if st.button("Entrenar Árbol de Decisión"):
        with st.spinner("Entrenando..."):
            from sklearn.tree import DecisionTreeClassifier
            tree_clf = DecisionTreeClassifier(max_depth=5)
            tree_clf.fit(X_train, y_train)
            acc_tree = accuracy_score(y_test, tree_clf.predict(X_test))
            st.metric("Precisión", f"{acc_tree:.4f}")

with tab2:
    if st.button("Entrenar Regresión Logística"):
        with st.spinner("Entrenando..."):
            from sklearn.linear_model import LogisticRegression
            log_clf = LogisticRegression(max_iter=200)
            log_clf.fit(X_train, y_train)
            acc_log = accuracy_score(y_test, log_clf.predict(X_test))
            st.metric("Precisión", f"{acc_log:.4f}")

with tab3:
    if st.button("Entrenar Random Forest"):
        with st.spinner("Entrenando..."):
            forest_clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
            forest_clf.fit(X_train, y_train)
            acc_forest = accuracy_score(y_test, forest_clf.predict(X_test))
            st.metric("Precisión", f"{acc_forest:.4f}")

