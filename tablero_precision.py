import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import requests
import io

# 🛠️ Configuración de la página
st.set_page_config(page_title="Tablero de Precisión del Modelo", page_icon="📊", layout="wide")

# 📢 Título del tablero
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

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 🔹 Sección 1: Evaluación General del Modelo
st.header("📌 Evaluación General del Modelo")

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
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
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Inactivo", "Normal", "Advertencia", "Crítico"], yticklabels=["Inactivo", "Normal", "Advertencia", "Crítico"])
plt.xlabel("Predicción")
plt.ylabel("Real")
st.pyplot(fig)

# 🔹 Sección 2: Importancia de Variables
st.header("📌 Importancia de Variables en la Predicción")
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({"Variable": X.columns, "Importancia": importances}).sort_values(by="Importancia", ascending=False)
st.plotly_chart(px.bar(feature_importance_df, x="Importancia", y="Variable", orientation='h', title="📊 Importancia de Variables"), use_container_width=True)

# 🔹 Sección 3: Comparación de Modelos
st.header("📌 Comparación de Modelos de Clasificación")
model_scores = {"Random Forest": accuracy}

# Evaluar otros modelos
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
models = {"Regresión Logística": LogisticRegression(max_iter=500), "Árbol de Decisión": DecisionTreeClassifier()}
for name, clf in models.items():
    clf.fit(X_train, y_train)
    model_scores[name] = accuracy_score(y_test, clf.predict(X_test))

# Visualización de comparación
df_scores = pd.DataFrame.from_dict(model_scores, orient='index', columns=["Precisión"]).reset_index()
st.plotly_chart(px.bar(df_scores, x="Precisión", y="index", orientation='h', title="📊 Precisión de Modelos"), use_container_width=True)

# 🔹 Sección 4: Curva ROC
st.header("📌 Curvas ROC/AUC")
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2, 3])
y_score = model.predict_proba(X_test)
fpr, tpr, roc_auc = {}, {}, {}

fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
colors = ["blue", "green", "orange", "red"]
for i, color in enumerate(colors):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    ax_roc.plot(fpr[i], tpr[i], color=color, lw=2, label=f"Clase {i} (AUC = {roc_auc[i]:.2f})")
ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=2)
ax_roc.set_title("Curvas ROC por Clase")
ax_roc.set_xlabel("Tasa de Falsos Positivos")
ax_roc.set_ylabel("Tasa de Verdaderos Positivos")
ax_roc.legend(loc="lower right")
st.pyplot(fig_roc)
