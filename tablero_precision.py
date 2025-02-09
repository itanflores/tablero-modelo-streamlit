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

# 🛠️ Configuración de la página
st.set_page_config(page_title="Tablero de Precisión del Modelo", page_icon="📊", layout="wide")

# 📌 Título del tablero
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

# Entrenar modelo con hiperparámetros ajustados
model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
st.metric("Precisión del Modelo", f"{accuracy:.4f}")
st.text("Reporte de Clasificación:")
st.text(classification_report(y_test, y_pred))

# Matriz de Confusión
conf_matrix = confusion_matrix(y_test, y_pred)
st.subheader("📊 Matriz de Confusión")
fig, ax = plt.subplots(figsize=(5, 3))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Inactivo", "Normal", "Advertencia", "Crítico"], yticklabels=["Inactivo", "Normal", "Advertencia", "Crítico"])
plt.xlabel("Predicción")
plt.ylabel("Real")
st.pyplot(fig)

# 🔹 Sección 2: Importancia de Variables
df_importance = pd.DataFrame({"Variable": X.columns, "Importancia": model.feature_importances_}).sort_values(by="Importancia", ascending=False)
st.header("📊 Importancia de Variables en la Predicción")
st.plotly_chart(px.bar(df_importance.head(10), x="Importancia", y="Variable", orientation='h', title="📊 Importancia de Variables"), use_container_width=True)

# 🔹 Sección 3: Comparación de Modelos en Tabla
st.header("📊 Comparación de Modelos de Clasificación")
model_scores = {"Random Forest": accuracy}

# Evaluar otros modelos de forma opcional
if st.checkbox("Comparar con otros modelos"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    models = {"Regresión Logística": LogisticRegression(max_iter=200), "Árbol de Decisión": DecisionTreeClassifier(max_depth=5)}
    for name, clf in models.items():
        try:
            clf.fit(X_train, y_train)
            model_scores[name] = accuracy_score(y_test, clf.predict(X_test))
        except Exception as e:
            model_scores[name] = f"Error: {str(e)}"
    
    # Mostrar tabla en lugar de gráfico
    df_scores = pd.DataFrame.from_dict(model_scores, orient='index', columns=["Precisión Promedio"]).reset_index()
    df_scores.rename(columns={"index": "Modelo"}, inplace=True)
    st.dataframe(df_scores)
