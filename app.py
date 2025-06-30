import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import itertools
import warnings
import pmdarima as pm
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error as mape


warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")

st.set_page_config(
    page_title="Hidrondina - Integrador", page_icon="⚡"
    )

st.title("Pantalla de Predicción")

st.sidebar.image("https://www.fonafe.gob.pe/pw_content/empresas/38/Img/Hidrandina.png")
st.sidebar.header("Hidrandina")
st.sidebar.write("Pronostico de Consumo Electrico en base a series de tiempo")
st.sidebar.selectbox("Departamento", ["Ancash"], index=0)

# Inicializa la variable de estado
st.markdown("---")
st.subheader("Modelos de Predicción:")
if 'modelo' not in st.session_state:
    st.session_state.modelo = "ARIMA"

# Crea los botones
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("ARIMA"):
        st.session_state.modelo = "ARIMA"
with col2:
    if st.button("SARIMA"):
        st.session_state.modelo = "SARIMA"
with col3:
    if st.button("Prophet"):
        st.session_state.modelo = "Prophet"
st.markdown("---")

@st.cache_data
def cargar_datos():
    df = pd.read_csv("data_consumo_unificado.csv")
    df = df[df['DEPARTAMENTO'] == 'Ancash']
    df['PERIODO'] = df['PERIODO'].astype(str).str[:6]
    df['PERIODO'] = pd.to_datetime(df['PERIODO'], format='%Y%m')
    provincias = sorted(df['PROVINCIA'].unique())
    return df, provincias

df_raw, provincias = cargar_datos()

# Selección de provincia
provincia_sel = st.sidebar.selectbox("Provincia", provincias)

# Filtrar y agrupar
df_filtrado = df_raw[df_raw['PROVINCIA'] == provincia_sel]
df_grouped = df_filtrado.groupby('PERIODO').agg({
    'CONSUMO': 'mean',
    'IMPORTE': 'mean'
}).rename(columns={'CONSUMO': 'CONSUMO_PROMEDIO', 'IMPORTE': 'IMPORTE_PROMEDIO'})

serie = df_grouped['CONSUMO_PROMEDIO'].copy()
serie.index = pd.to_datetime(serie.index)
serie = serie.asfreq('MS')

st.subheader(f"Pronóstico de Consumo - Provincia: {provincia_sel}")

# Modelo SARIMA
if st.session_state.modelo  == "SARIMA":
    with st.spinner("Entrenando modelo SARIMA automáticamente..."):
        modelo_auto = pm.auto_arima(
            serie,
            seasonal=True,
            m=12,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore"
        )
        forecast, confint = modelo_auto.predict(n_periods=3, return_conf_int=True)
        fechas_futuras = pd.date_range(serie.index[-1] + pd.offsets.MonthBegin(1), periods=3, freq='MS')
        pred_series = pd.Series(forecast, index=fechas_futuras)

        # Métrica
        y_true = serie[-3:]
        y_pred = modelo_auto.predict_in_sample(start=len(serie) - 3)
        mape_val = mape(y_true, y_pred)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(serie, label='Histórico')
        ax.plot(pred_series, label='Pronóstico (3 meses)', color='orange')
        ax.fill_between(fechas_futuras, confint[:, 0], confint[:, 1], color='orange', alpha=0.3)
        ax.set_title("Consumo Promedio Mensual + Pronóstico (SARIMA)")
        ax.set_ylabel("Consumo Promedio (kWh)")
        ax.legend()
        st.pyplot(fig)

        st.metric("MAPE (SARIMA)", f"{mape_val*100:.2f}%")

# Modelo Prophet
elif st.session_state.modelo  == "Prophet":
    with st.spinner("Entrenando modelo Prophet..."):
        df_prophet = serie.reset_index()
        df_prophet.columns = ['ds', 'y']
        model = Prophet()
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=3, freq='MS')
        forecast = model.predict(future)

        # Métrica
        y_true = df_prophet['y'][-3:]
        y_pred = model.predict(df_prophet.tail(3))['yhat']
        mape_val = mape(y_true, y_pred)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_prophet['ds'], df_prophet['y'], label='Histórico')
        ax.plot(forecast['ds'], forecast['yhat'], label='Pronóstico (3 meses)', color='green')
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='green', alpha=0.3)
        ax.set_title("Consumo Promedio Mensual + Pronóstico (Prophet)")
        ax.set_ylabel("Consumo Promedio (kWh)")
        ax.legend()
        st.pyplot(fig)

        st.metric("MAPE (Prophet)", f"{mape_val*100:.2f}%")

# Mostrar tabla final
st.subheader("Tabla de Datos Promedios por Mes")
st.dataframe(df_grouped.reset_index(), use_container_width=True)

