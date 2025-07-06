import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
import os
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Configuración general
warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="Hidrandina - Integrador", page_icon="⚡", initial_sidebar_state="expanded")

st.title("Pantalla de Predicción")

# Sidebar
st.sidebar.image("https://www.fonafe.gob.pe/pw_content/empresas/38/Img/Hidrandina.png")
st.sidebar.header("Hidrandina")
st.sidebar.write("Pronóstico de Consumo Eléctrico en base a series de tiempo")
st.sidebar.selectbox("Departamento", ["Ancash"], index=0)

# Función para cargar provincias disponibles
def obtener_provincias():
    archivos = os.listdir("Data")
    provincias = [archivo.replace(".csv", "") for archivo in archivos if archivo.endswith(".csv")]
    provincias.sort()
    return ["Total"] + provincias

# Función para cargar datos
def cargar_datos(nombre):
    df = pd.read_csv(f"Data/{nombre}.csv")
    df['PERIODO'] = pd.to_datetime(df['PERIODO'].astype(str), format='%Y%m')
    return df

# Métricas
def calcular_metricas(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return mse, mae, rmse, mape

def mostrar_metricas(mse, mae, rmse, mape):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("MAPE (%)", f"{round(mape, 2)}%")
    with col2:
        st.metric("MAE", round(mae, 2))
    with col3:
        st.metric("RMSE", round(rmse, 2))
    with col4:
        st.metric("MSE", round(mse, 2))

def interpretar_metricas():
    with st.expander("\U0001F9E0 ¿Cómo interpretar estas métricas?"):
        st.markdown("""
        - **\U0001F4D8 MAPE**: Error porcentual promedio, útil para interpretar la precisión en términos relativos. 
        Por ejemplo, un MAPE de 8% significa que el modelo, en promedio, **falla un 8% respecto a los valores reales**.
        
        - **\U0001F4CF MAE**: Error promedio absoluto, fácil de interpretar porque está en las mismas unidades que los datos.
        *Por ejemplo, si el consumo promedio es en kWh, el MAE te dice cuánto se equivoca el modelo, en promedio, en kWh.*
        
        - **\U0001F4CA RMSE**: Similar al MAE pero con mayor sensibilidad a errores grandes.
        *Un valor más bajo indica mejores predicciones.*
        
        - **\U0001F4CC MSE**: Penaliza errores grandes al elevarlos al cuadrado. Útil cuando necesitas castigar errores fuertes.
        *Útil cuando quieres penalizar errores grandes.*
        """)

def mostrar_comparacion(fechas_futuras, pred_arima, sarima_forecast, prophet_forecast):
    df_resultado = pd.DataFrame({
        "Periodo": fechas_futuras.strftime('%Y-%m'),
        "ARIMA": pred_arima,
        "SARIMA": sarima_forecast,
        "Prophet": prophet_forecast[prophet_forecast['ds'].isin(fechas_futuras)]['yhat'].values
    })
    st.caption("Comparación de predicciones para los próximos 3 meses")
    st.dataframe(df_resultado.set_index("Periodo"), use_container_width=True)


# Interfaz
provincias = obtener_provincias()
provincia_sel = st.sidebar.selectbox("Provincia", provincias, index=provincias.index("Total"))

df = cargar_datos(provincia_sel)
df.rename(columns={"CONSUMO_PROMEDIO": "CONSUMO TOTAL", "IMPORTE_PROMEDIO": "IMPORTE TOTAL"}, inplace=True)
df['CONSUMO TOTAL'] = (df['CONSUMO TOTAL']).round(1)
df['IMPORTE TOTAL'] = (df['IMPORTE TOTAL']).round(1)

serie = df.set_index("PERIODO")["CONSUMO TOTAL"].asfreq("MS")

st.subheader(f"Pronóstico de Consumo - Provincia: {provincia_sel}")
st.caption("Valores expresados en miles. Ejemplo: 12.5 equivale a 12,500 kWh.")

with st.expander("\U0001F4C5 Resumen de Datos"):
    df_origen = df.copy()
    df_origen["PERIODO"] = df_origen["PERIODO"].dt.strftime('%Y-%m')
    st.dataframe(df_origen, use_container_width=True)

# === CALCULAMOS TODOS LOS MODELOS ANTES DE MOSTRAR COMPARACIÓN ===

# Fechas futuras
fechas_futuras = pd.date_range(serie.index[-1] + pd.offsets.MonthBegin(1), periods=3, freq='MS')

# ARIMA
modelo_arima = ARIMA(serie, order=(1, 1, 1)).fit()
pred_arima = modelo_arima.forecast(steps=3)

# SARIMA
modelo_sarima = pm.auto_arima(serie, start_p=0, start_q=0, max_p=3, max_q=3, d=None,
                               seasonal=True, m=12, D=1, trace=False,
                               error_action='ignore', suppress_warnings=True,
                               stepwise=True, seasonal_test='ch')
sarima_forecast, sarima_confint = modelo_sarima.predict(n_periods=3, return_conf_int=True)

# Prophet
df_prophet = serie.reset_index()
df_prophet.columns = ['ds', 'y']
modelo_prophet = Prophet()
modelo_prophet.fit(df_prophet)
future = modelo_prophet.make_future_dataframe(periods=3, freq='MS')
prophet_forecast = modelo_prophet.predict(future)

# === PESTAÑAS (puedes envolverlas en columnas si quieres centrarlas visualmente) ===
tabs = st.tabs(["ARIMA", "SARIMA", "Prophet"])

# ARIMA
with tabs[0]:
    y_true = serie[-3:]
    y_pred = pred_arima
    mse, mae, rmse, mape = calcular_metricas(y_true, y_pred)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=serie.index, y=serie, name='Histórico'))
    fig.add_trace(go.Scatter(x=fechas_futuras, y=pred_arima, name='Pronóstico (ARIMA)', line=dict(color='blue')))
    fig.update_layout(title='Pronóstico con ARIMA', xaxis_title='Fecha', yaxis_title='Consumo Total (mil kWh)')
    st.plotly_chart(fig)

    mostrar_comparacion(fechas_futuras, pred_arima, sarima_forecast, prophet_forecast)

    mostrar_metricas(mse, mae, rmse, mape)
    interpretar_metricas()

# SARIMA
with tabs[1]:
    y_true = serie[-3:]
    y_pred = modelo_sarima.predict_in_sample()[-3:]
    mse, mae, rmse, mape = calcular_metricas(y_true, y_pred)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=serie.index, y=serie, name='Histórico'))
    fig.add_trace(go.Scatter(x=fechas_futuras, y=sarima_forecast, name='Pronóstico (SARIMA)', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=fechas_futuras, y=sarima_confint[:, 0], showlegend=False, line=dict(color='orange', dash='dot')))
    fig.add_trace(go.Scatter(x=fechas_futuras, y=sarima_confint[:, 1], showlegend=False, fill='tonexty', line=dict(color='orange', dash='dot')))
    fig.update_layout(title='Pronóstico con SARIMA', xaxis_title='Fecha', yaxis_title='Consumo Total (mil kWh)')
    st.plotly_chart(fig)

    mostrar_comparacion(fechas_futuras, pred_arima, sarima_forecast, prophet_forecast)
    
    mostrar_metricas(mse, mae, rmse, mape)
    interpretar_metricas()

# Prophet
with tabs[2]:
    y_true = df_prophet['y'][-3:]
    y_pred = modelo_prophet.predict(df_prophet.tail(3))['yhat']
    mse, mae, rmse, mape = calcular_metricas(y_true, y_pred)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], name='Histórico'))
    fig.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat'], name='Pronóstico (Prophet)', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat_lower'], showlegend=False, line=dict(color='green', dash='dot')))
    fig.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat_upper'], showlegend=False, fill='tonexty', line=dict(color='green', dash='dot')))
    fig.update_layout(title='Pronóstico con Prophet', xaxis_title='Fecha', yaxis_title='Consumo Total (mil kWh)')
    st.plotly_chart(fig)

    mostrar_comparacion(fechas_futuras, pred_arima, sarima_forecast, prophet_forecast)

    mostrar_metricas(mse, mae, rmse, mape)
    interpretar_metricas()