import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
import pandas as pd

# -----------------------
# Carga de recursos
# -----------------------
MODEL_PATH  = 'modelo_notas.h5'
SCALER_PATH = 'scaler_notas.pkl'
DATA_PATH   = 'data/notas_estudiantes.csv'

model  = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Carga datos para calcular MAE, RMSE y margen de error
df = pd.read_csv(DATA_PATH, sep=';')
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])
X = df[['Parciales', 'A']].values
y = df['EF'].values
y_pred_full = model.predict(scaler.transform(X)).flatten()
mae = mean_absolute_error(y, y_pred_full)
abs_errors = np.abs(y - y_pred_full)
se_mae = np.std(abs_errors, ddof=1) / np.sqrt(len(abs_errors))
moe95 = 1.96 * se_mae

# -----------------------
# CSS embebido
# -----------------------
st.markdown(
    """
    <style>
    .stApp { background: #F0F2F6; }
    .input-container { background: white; border-radius: 8px; padding: 20px;
                       box-shadow: 0px 4px 8px rgba(0,0,0,0.1); }
    .result-container { background: white; border-radius: 8px; padding: 20px;
                        box-shadow: 0px 4px 8px rgba(0,0,0,0.1); }
    .stButton>button { background-color: #0072C3; color: white; font-weight: bold;
                       border-radius: 6px; padding: 10px 24px; }
    h1, h2, .stMetricValue { color: #333; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Título
# -----------------------
st.markdown("<h1 style='text-align:center;'>📊 Predicción de Nota del Examen Final</h1>", unsafe_allow_html=True)
st.markdown(f"**MAE del modelo:** {mae:.3f}  •  **Margen de Error (95%):** ±{moe95:.3f}", unsafe_allow_html=True)

# -----------------------
# Layout en dos columnas
# -----------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.subheader("Entrada de Datos")
    with st.form(key="input_form"):
        parciales = st.number_input("Parciales", min_value=0, max_value=30, value=25,
                                    help="Suma de parciales del estudiante")
        A = st.number_input("A", min_value=0, max_value=30, value=25,
                            help="Calificación de la actividad A")
        submitted = st.form_submit_button("Predecir EF")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    st.subheader("Resultado")
    if submitted:
        entrada = np.array([[parciales, A]])
        entrada_scaled = scaler.transform(entrada)
        pred = model.predict(entrada_scaled)[0][0]
        st.metric(label="📈 Predicción EF", value=f"{pred:.1f}")
        lim_inf = pred - moe95
        lim_sup = pred + moe95
        st.write(f"Intervalo aproximado (95%): [{lim_inf:.1f}, {lim_sup:.1f}]")
    else:
        st.info("Complete los campos y haga click en 'Predecir EF' para ver los resultados.")
    st.markdown('</div>', unsafe_allow_html=True)
