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

col1, col2, col3 = st.columns([1.5, 2, 2])

with col1:
    st.metric("📊 Precisión del Modelo", f"{accuracy_score(y_test, y_pred):.4f}")
    st.caption("🔹 La precisión mide la proporción de predicciones correctas. Valores más altos indican mejor desempeño.")

with col2.expander("📋 Reporte de Clasificación"):
    st.text(classification_report(y_test, y_pred))
    st.caption("🔹 El reporte muestra métricas clave como precisión, recall y F1-score para evaluar el desempeño en cada categoría.")

with col3:
    st.write("📊 Matriz de Confusión")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    st.pyplot(fig)
    st.caption("🔹 La matriz de confusión muestra las predicciones correctas en la diagonal y los errores fuera de ella.")

st.divider()

# 📌 **Sección 2: Importancia de Variables**
st.header("📊 Importancia de Variables en la Predicción")

df_importance = pd.DataFrame({
    "Variable": X.columns,
    "Importancia": model.feature_importances_
}).sort_values(by="Importancia", ascending=False)

col4, col5 = st.columns([1.5, 3])

with col4.expander("📋 Ver Variables Importantes", expanded=True):
    st.dataframe(df_importance.head(10), height=300)
    st.caption("🔹 Muestra las 10 variables más importantes utilizadas por el modelo para la toma de decisiones.")

fig_imp = px.bar(df_importance.head(10), 
                 x="Importancia", 
                 y="Variable", 
                 orientation='h', 
                 title="📊 Importancia de Variables")

col5.plotly_chart(fig_imp, use_container_width=True)
st.caption("🔹 Este gráfico destaca qué variables tienen mayor peso en las predicciones del modelo.")

st.divider()

# 📌 **Sección 3: Comparación de Modelos**
st.header("📊 Comparación de Modelos de Clasificación")

# 🔹 Pestañas para organizar cada modelo
tab1, tab2, tab3 = st.tabs(["🌳 Árbol de Decisión", "📈 Regresión Logística", "🌲 Random Forest"])

# 📌 Sección Árbol de Decisión en `st.tabs()`
with tab1:
    st.subheader("🌳 Árbol de Decisión")
    st.caption("🔹 Modelo basado en reglas jerárquicas. Es fácil de interpretar pero puede sobreajustarse con demasiada profundidad.")

    if "tree_trained" not in st.session_state:
        st.session_state["tree_trained"] = False

    if st.button("Entrenar Árbol de Decisión"):
        with st.spinner("Entrenando..."):
            from sklearn.tree import DecisionTreeClassifier
            tree_clf = DecisionTreeClassifier(max_depth=5)
            tree_clf.fit(X_train, y_train)
            st.session_state["tree_acc"] = accuracy_score(y_test, tree_clf.predict(X_test))
            st.session_state["tree_cm"] = confusion_matrix(y_test, tree_clf.predict(X_test))
            st.session_state["tree_trained"] = True  

    if st.session_state["tree_trained"]:
        st.metric("Precisión", f"{st.session_state['tree_acc']:.4f}")

# 📈 **Regresión Logística**
with tab2:
    st.subheader("📈 Regresión Logística")
    st.caption("🔹 Modelo lineal utilizado para clasificaciones binarias o multiclase con buena interpretabilidad.")

    if st.button("Entrenar Regresión Logística"):
        with st.spinner("Entrenando..."):
            from sklearn.linear_model import LogisticRegression
            log_clf = LogisticRegression(max_iter=200)
            log_clf.fit(X_train, y_train)
            acc_log = accuracy_score(y_test, log_clf.predict(X_test))
            st.metric("Precisión", f"{acc_log:.4f}")

            # 📊 Mostrar matriz de confusión
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(confusion_matrix(y_test, log_clf.predict(X_test)), annot=True, fmt="d", cmap="Blues")
            st.pyplot(fig)
            st.caption("🔹 La matriz de confusión evalúa qué tan bien el modelo distingue entre clases.")

# 🌲 **Random Forest**
with tab3:
    st.subheader("🌲 Random Forest")
    st.caption("🔹 Conjunto de múltiples árboles de decisión que mejora la precisión y reduce el sobreajuste.")

    if st.button("Entrenar Random Forest"):
        with st.spinner("Entrenando..."):
            forest_clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
            forest_clf.fit(X_train, y_train)
            acc_forest = accuracy_score(y_test, forest_clf.predict(X_test))
            st.metric("Precisión", f"{acc_forest:.4f}")

            # 📊 Mostrar matriz de confusión
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(confusion_matrix(y_test, forest_clf.predict(X_test)), annot=True, fmt="d", cmap="Blues")
            st.pyplot(fig)
            st.caption("🔹 Evalúa los aciertos y errores del modelo en la clasificación.")

