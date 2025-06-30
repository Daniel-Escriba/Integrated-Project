import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from prophet import Prophet
from sklearn.metrics import  mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


warnings.filterwarnings("ignore")
st.set_page_config(layout="centered", page_title="Hidrondina - Integrador", page_icon="⚡",
    initial_sidebar_state="expanded"
    )


st.title("Pantalla de Predicción")

st.sidebar.image("https://www.fonafe.gob.pe/pw_content/empresas/38/Img/Hidrandina.png")
st.sidebar.header("Hidrandina")
st.sidebar.write("Pronostico de Consumo Electrico en base a series de tiempo")
st.sidebar.selectbox("Departamento", ["Ancash"], index=0)

# Inicializa la variable de estado
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
    df = pd.read_csv("data_consumo_ancash.csv")
    df = df[df['DEPARTAMENTO'] == 'Ancash']
    df['PERIODO'] = df['PERIODO'].astype(str).str[:6]
    df['PERIODO'] = pd.to_datetime(df['PERIODO'], format='%Y%m')
    provincias = sorted(df['PROVINCIA'].unique())
    return df, provincias

df_raw, provincias = cargar_datos()

provincias_con_todas = ["Todas"] + provincias
provincia_sel = st.sidebar.selectbox("Provincia", provincias_con_todas)

# Filtrar y agrupar
if provincia_sel == "Todas":
    df_filtrado = df_raw.copy()
else:
    df_filtrado = df_raw[df_raw['PROVINCIA'] == provincia_sel]
df_grouped = df_filtrado.groupby('PERIODO').agg({
    'CONSUMO': 'mean',
    'IMPORTE': 'mean'
}).rename(columns={'CONSUMO': 'CONSUMO_PROMEDIO', 'IMPORTE': 'IMPORTE_PROMEDIO'})

serie = df_grouped['CONSUMO_PROMEDIO'].copy()
serie.index = pd.to_datetime(serie.index)
serie = serie.asfreq('MS')

st.subheader(f"Pronóstico de Consumo - Provincia: {provincia_sel}")

# Mostrar tabla final
st.subheader("Datos Promedios por Mes")
st.dataframe(df_grouped.reset_index(), use_container_width=True)

# Modelo ARIMA
if st.session_state.modelo == "ARIMA":
    model = ARIMA(serie, order=(1, 1, 1))
    model_fit = model.fit()

    # Predecir 3 meses futuros
    pred = model_fit.forecast(steps=3)
    fechas_futuras = pd.date_range(serie.index[-1] + pd.offsets.MonthBegin(1), periods=3, freq='MS')
    pred_series = pd.Series(pred, index=fechas_futuras)


    # Métrica (usar últimos 3 reales vs últimos 3 predichos in-sample)
    y_true = serie[-3:]
    y_pred = pred

    # Cálculo de métricas
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # en %

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(serie, label='Histórico')
    ax.plot(pred_series, label='Pronóstico (3 meses)', color='blue')
    ax.set_title("Consumo Promedio Mensual + Pronóstico (ARIMA)")
    ax.set_ylabel("Consumo Promedio (kWh)")
    ax.legend()
    st.pyplot(fig)

    # Mostrar en Streamlit
    st.write("📊 Métricas de evaluación:")
    st.metric("MSE", round(mse, 2))
    st.metric("MAE", round(mae, 2))
    st.metric("RMSE", round(rmse, 2))
    st.metric("MAPE (%)", round(mape, 2))

    # Muestra la interpretación de las métricas
    st.write("🧠 Interpretación de las métricas:")
    st.info("""
    - **MSE (Error Cuadrático Medio)**:  
    Es el promedio de los errores al cuadrado entre lo que el modelo predice y lo que realmente ocurrió.  
    Como los errores están elevados al cuadrado, **los errores grandes pesan más**.  
    🔍 *Útil cuando quieres penalizar errores grandes.*

    - **MAE (Error Absoluto Medio)**:  
    Es el promedio de los errores absolutos (la diferencia sin importar el signo).  
    Es más fácil de entender porque está en las **mismas unidades que los datos reales**.  
    📏 *Por ejemplo, si el consumo promedio es en kWh, el MAE te dice cuánto se equivoca el modelo, en promedio, en kWh.*

    - **RMSE (Raíz del Error Cuadrático Medio)**:  
    Es como el MSE, pero se le saca la raíz cuadrada para que esté en las mismas unidades que los datos.  
    Es sensible a errores grandes, pero más fácil de interpretar que el MSE.  
    📊 *Un valor más bajo indica mejores predicciones.*

    - **MAPE (Error Porcentual Absoluto Medio)**:  
    Muestra en promedio **cuánto se equivoca el modelo en porcentaje respecto al valor real**.  
    Por ejemplo, un MAPE de 8% significa que el modelo, en promedio, **falla un 8% respecto a los valores reales**.  
    📌 *Muy útil para comparar el desempeño del modelo, sin importar las unidades de medida.*
    """)




# Modelo SARIMA
elif st.session_state.modelo  == "SARIMA":
    with st.spinner("Entrenando modelo SARIMA automáticamente..."):
        modelo_auto = pm.auto_arima(
                        serie,             # tu serie de tiempo (por ejemplo, un pd.Series)
                        start_p=0,
                        start_q=0,
                        max_p=3,
                        max_q=3,
                        d=None,            # deja que lo calcule automáticamente
                        seasonal=True,
                        m=12,              # estacionalidad de 12 si es mensual
                        D=1,               # forzamos 1 diferencia estacional
                        trace=True,
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True,
                        seasonal_test='ch'  # alternativa al default ('ocsb'), menos sensible
                    )
        # Forecast 3 meses futuros
        forecast, confint = modelo_auto.predict(n_periods=3, return_conf_int=True)
        fechas_futuras = pd.date_range(serie.index[-1] + pd.offsets.MonthBegin(1), periods=3, freq='MS')
        pred_series = pd.Series(forecast, index=fechas_futuras)

        # Métrica (usar últimos 3 reales vs últimos 3 predichos in-sample)
        y_true = serie[-3:]
        y_pred = modelo_auto.predict_in_sample()[-3:]

        # Cálculo de métricas
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # en %

# Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(serie, label='Histórico')
        ax.plot(pred_series, label='Pronóstico (3 meses)', color='orange')
        ax.fill_between(fechas_futuras, confint[:, 0], confint[:, 1], color='orange', alpha=0.3)
        ax.set_title("Consumo Promedio Mensual + Pronóstico (SARIMA)")
        ax.set_ylabel("Consumo Promedio (kWh)")
        ax.legend()
        st.pyplot(fig)

        # Mostrar en Streamlit
        st.write("📊 Métricas de evaluación:")
        st.metric("MSE", round(mse, 2))
        st.metric("MAE", round(mae, 2))
        st.metric("RMSE", round(rmse, 2))
        st.metric("MAPE (%)", round(mape, 2))


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
