# 🔹 Sección 2: Sección de Pronósticos
# [Código existente de la Sección 2]

# 🔹 Nueva Sección: Evaluación del Modelo
st.header("📈 Evaluación del Modelo")

# Calcular la Curva ROC y el AUC
y_true = df_filtrado["Estado del Sistema"].apply(lambda x: 1 if x == "Crítico" else 0)
y_pred_proba = model.predict_proba(df_filtrado[["Uso CPU (%)", "Temperatura (°C)"]])[:, 1]

fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Crear la gráfica de la Curva ROC
fig_roc = px.area(
    x=fpr,
    y=tpr,
    title=f'Curva ROC (AUC = {roc_auc:.2f})',
    labels=dict(x='Tasa de Falsos Positivos (FPR)', y='Tasa de Verdaderos Positivos (TPR)'),
    width=700, height=500
)
fig_roc.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)
fig_roc.update_yaxes(scaleanchor="x", scaleratio=1)
fig_roc.update_xaxes(constrain='domain')

# Mostrar la gráfica
st.plotly_chart(fig_roc, use_container_width=True)
st.write("""
La **Curva ROC** muestra el rendimiento de un modelo de clasificación en todos los umbrales de clasificación.
- **AUC (Área Bajo la Curva)**: Un valor cercano a 1 indica un modelo excelente, mientras que un valor cercano a 0.5 sugiere que el modelo no es mejor que una predicción aleatoria.
""")

# 🔹 Sección 3: Análisis de Outliers y Eficiencia Térmica
# [Código existente de la Sección 3]
