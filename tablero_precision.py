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
X = X.reindex(columns=X.columns, fill_value=0)  # Asegurar misma estructura en entrenamiento y prueba

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

# Usar columnas para distribuir métricas
col1, col2 = st.columns(2)

# Restaurar max_depth=None en Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
col1.metric("📊 Precisión del Modelo", f"{accuracy:.4f}")

# Matriz de Confusión
st.subheader("📊 Matriz de Confusión")
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
st.pyplot(fig)

st.divider()

# 🔹 Sección 2: Importancia de Variables
st.header("📊 Importancia de Variables en la Predicción")
df_importance = pd.DataFrame({"Variable": X.columns, "Importancia": model.feature_importances_}).sort_values(by="Importancia", ascending=False)
st.plotly_chart(px.bar(df_importance.head(10), x="Importancia", y="Variable", orientation='h', title="📊 Importancia de Variables"), use_container_width=True)

st.divider()

# 🔹 Sección 3: Comparación de Modelos en Tabla
st.header("📊 Comparación de Modelos de Clasificación")
st.markdown("**Nota:** Modelos ordenados del más rápido al más lento.")

# Checkboxes para seleccionar modelos
run_tree = st.checkbox("🌳 Árbol de Decisión (🟢 Rápido)")
run_logistic = st.checkbox("📈 Regresión Logística (🟡 Moderado)")
run_forest = st.checkbox("🌲 Random Forest (🔴 Lento)")

model_scores = {}

if run_tree:
    with st.spinner("Entrenando Árbol de Decisión..."):
        from sklearn.tree import DecisionTreeClassifier
        tree_clf = DecisionTreeClassifier(max_depth=5)
        tree_clf.fit(X_train, y_train)
        model_scores["Árbol de Decisión"] = accuracy_score(y_test, tree_clf.predict(X_test))
        sleep(0.5)

if run_logistic:
    with st.spinner("Entrenando Regresión Logística..."):
        from sklearn.linear_model import LogisticRegression
        log_clf = LogisticRegression(max_iter=200)
        log_clf.fit(X_train, y_train)
        model_scores["Regresión Logística"] = accuracy_score(y_test, log_clf.predict(X_test))
        sleep(1)

if run_forest:
    with st.spinner("Entrenando Random Forest (puede tardar más)..."):
        forest_clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
        forest_clf.fit(X_train, y_train)
        model_scores["Random Forest"] = accuracy_score(y_test, forest_clf.predict(X_test))
        sleep(2)

if model_scores:
    df_scores = pd.DataFrame.from_dict(model_scores, orient='index', columns=["Precisión Promedio"]).reset_index()
    df_scores.rename(columns={"index": "Modelo"}, inplace=True)
    st.dataframe(df_scores, use_container_width=True)
