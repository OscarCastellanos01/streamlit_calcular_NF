import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_error

# 1) Configuración de la página
st.set_page_config(
    page_title="Predicción Examen Final",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2) Cargar modelo y datos
MODEL_PATH  = "models/modelo_notas.h5"
SCALER_PATH = "scaler/scaler_notas.pkl"
DATA_PATH   = "data/notas_estudiantes.csv"

@st.cache_data
def load_model_and_scaler(model_path, scaler_path):
    model  = tf.keras.models.load_model(model_path, compile=False)
    scaler  = joblib.load(scaler_path)
    return model, scaler

@st.cache_data
def load_dataset(path):
    df = pd.read_csv(path, sep=';', encoding='utf-8')
    df.columns = df.columns.str.strip().str.capitalize()
    df = df.rename(columns={'Actividad': 'A', 'Ef': 'EF'})
    return df

model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
df            = load_dataset(DATA_PATH)

X = df[["Parciales", "A"]]
y = df["EF"]

# 3) Métricas del modelo
preds = model.predict(scaler.transform(X))
mae   = mean_absolute_error(y, preds)

# 4) Título y resumen
st.title("📊 Predicción de Nota del Examen Final")
st.markdown(
    f"> **MAE del modelo:** {mae:.3f}  •  **Margen de Error (95%):** ±{1.96 * mae:.3f}"
)

# 5) Entradas en la barra lateral
st.sidebar.header("🔧 Parámetros de Entrada")
parciales = st.sidebar.number_input(
    "Puntaje en Parciales", min_value=0, max_value=30, value=26
)
a_val     = st.sidebar.number_input(
    "Puntaje en A", min_value=0, max_value=30, value=24
)
btn       = st.sidebar.button("▶️ Predecir EF")

# 6) Mostrar predicción cuando el usuario haga clic
if btn:
    X_new = scaler.transform([[parciales, a_val]])
    ef_pred = float(model.predict(X_new)[0][0])
    lower   = ef_pred - 1.96 * mae
    upper   = ef_pred + 1.96 * mae

    # Usamos dos columnas para alinear las métricas
    col1, col2 = st.columns(2)
    col1.metric(
        label="Predicción EF", 
        value=f"{ef_pred:.1f}"
    )
    col2.metric(
        label="Intervalo 95%", 
        value=f"[{lower:.1f}, {upper:.1f}]"
    )

# 7) Mostrar tabla de datos originales
with st.expander("📋 Ver datos de entrenamiento"):
    st.dataframe(df, use_container_width=True)
