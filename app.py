import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Configuración general
warnings.filterwarnings("ignore")
st.set_page_config(layout="centered", page_title="Hidrandina - Integrador", page_icon="⚡")
st.title("Pantalla de Predicción")

# Sidebar
st.sidebar.image("https://www.fonafe.gob.pe/pw_content/empresas/38/Img/Hidrandina.png")
st.sidebar.header("Hidrandina")
st.sidebar.write("Pronóstico de Consumo Eléctrico en base a series de tiempo")
st.sidebar.selectbox("Departamento", ["Ancash"], index=0)

# Inicializa variable de modelo
st.subheader("Modelos de Predicción:")
if 'modelo' not in st.session_state:
    st.session_state.modelo = "ARIMA"

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
   df = pd.read_csv("data_consumo_ancash.csv")
   df['PERIODO'] = pd.to_datetime(df['PERIODO'].astype(str).str[:6], format='%Y%m')
   provincias = sorted(df['PROVINCIA'].unique())
   return df, provincias


# Funciones auxiliares
def calcular_metricas(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return mse, mae, rmse, mape

def mostrar_metricas(mse, mae, rmse, mape):
    st.write("\U0001F4CA Métricas de evaluación:")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("MSE", round(mse, 2))
    with col2:
        st.metric("MAE", round(mae, 2))
    with col3:
        st.metric("RMSE", round(rmse, 2))
    with col4:
        st.metric("MAPE (%)", f"{round(mape, 2)}%")

def interpretar_metricas():
    with st.expander("\U0001F9E0 ¿Cómo interpretar estas métricas?"):
        st.markdown("""
        - **\U0001F4D8 MSE**: Penaliza errores grandes al elevarlos al cuadrado. Útil cuando necesitas castigar errores fuertes.
        - **\U0001F4CF MAE**: Error promedio absoluto, fácil de interpretar porque está en las mismas unidades que los datos.
        - **\U0001F4CA RMSE**: Similar al MAE pero con mayor sensibilidad a errores grandes.
        - **\U0001F4CC MAPE**: Error porcentual promedio, útil para interpretar la precisión en términos relativos.
        """)

def escalar_miles(df, columnas):
    for col in columnas:
        df[col] = (df[col] / 1000).round(1)
    return df

# Carga de datos
df_raw, provincias = cargar_datos()
provincias_con_todas = ["Todas"] + provincias
provincia_sel = st.sidebar.selectbox("Provincia", provincias_con_todas)

# Filtrado y agregación
if provincia_sel == "Todas":
    df_filtrado = df_raw.copy()
else:
    df_filtrado = df_raw[df_raw['PROVINCIA'] == provincia_sel]

if df_filtrado.empty:
    st.warning("No hay datos disponibles para la provincia seleccionada.")
    st.stop()

df_grouped = df_filtrado.groupby('PERIODO').agg({'CONSUMO': 'sum', 'IMPORTE': 'sum'})
df_grouped.rename(columns={'CONSUMO': 'CONSUMO TOTAL', 'IMPORTE': 'IMPORTE TOTAL'}, inplace=True)
df_grouped = escalar_miles(df_grouped, ['CONSUMO TOTAL', 'IMPORTE TOTAL'])

serie = df_grouped['CONSUMO TOTAL'].copy()
serie.index = pd.to_datetime(serie.index)
serie = serie.asfreq('MS')

st.subheader(f"Pronóstico de Consumo - Provincia: {provincia_sel}")
st.subheader("Datos Totales por Mes")
st.caption("Valores expresados en miles. Ejemplo: 12.5 equivale a 12,500 kWh.")
st.dataframe(df_grouped.reset_index(), use_container_width=True)

# Modelos
if st.session_state.modelo == "ARIMA":
    model = ARIMA(serie, order=(1, 1, 1)).fit()
    pred = model.forecast(steps=3)
    fechas_futuras = pd.date_range(serie.index[-1] + pd.offsets.MonthBegin(1), periods=3, freq='MS')
    pred_series = pd.Series(pred, index=fechas_futuras)
    y_true = serie[-3:]
    y_pred = pred
    mse, mae, rmse, mape = calcular_metricas(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(serie, label='Histórico')
    ax.plot(pred_series, label='Pronóstico (3 meses)', color='blue')
    ax.set_title("Consumo Total Mensual + Pronóstico (ARIMA)")
    ax.set_ylabel("Consumo Total (mil kWh)")
    ax.legend()
    st.pyplot(fig)
    mostrar_metricas(mse, mae, rmse, mape)
    interpretar_metricas()

elif st.session_state.modelo == "SARIMA":
    with st.spinner("Entrenando modelo SARIMA..."):
        modelo_auto = pm.auto_arima(serie, start_p=0, start_q=0, max_p=3, max_q=3, d=None,
                                    seasonal=True, m=12, D=1, trace=True, error_action='ignore',
                                    suppress_warnings=True, stepwise=True, seasonal_test='ch')
        forecast, confint = modelo_auto.predict(n_periods=3, return_conf_int=True)
        fechas_futuras = pd.date_range(serie.index[-1] + pd.offsets.MonthBegin(1), periods=3, freq='MS')
        pred_series = pd.Series(forecast, index=fechas_futuras)
        y_true = serie[-3:]
        y_pred = modelo_auto.predict_in_sample()[-3:]
        mse, mae, rmse, mape = calcular_metricas(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(serie, label='Histórico')
        ax.plot(pred_series, label='Pronóstico (3 meses)', color='orange')
        ax.fill_between(fechas_futuras, confint[:, 0], confint[:, 1], color='orange', alpha=0.3)
        ax.set_title("Consumo Total Mensual + Pronóstico (SARIMA)")
        ax.set_ylabel("Consumo Total (mil kWh)")
        ax.legend()
        st.pyplot(fig)
        mostrar_metricas(mse, mae, rmse, mape)
        interpretar_metricas()

elif st.session_state.modelo == "Prophet":
    with st.spinner("Entrenando modelo Prophet..."):
        df_prophet = serie.reset_index()
        df_prophet.columns = ['ds', 'y']
        model = Prophet()
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=3, freq='MS')
        forecast = model.predict(future)

        y_true = df_prophet['y'][-3:]
        y_pred = model.predict(df_prophet.tail(3))['yhat']
        mse, mae, rmse, mape = calcular_metricas(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_prophet['ds'], df_prophet['y'], label='Histórico')
        ax.plot(forecast['ds'], forecast['yhat'], label='Pronóstico (3 meses)', color='green')
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='green', alpha=0.3)
        ax.set_title("Consumo Total Mensual + Pronóstico (Prophet)")
        ax.set_ylabel("Consumo Total (mil kWh)")
        ax.legend()
        st.pyplot(fig)
        mostrar_metricas(mse, mae, rmse, mape)
        interpretar_metricas()
