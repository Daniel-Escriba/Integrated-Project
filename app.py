# Requiere: pip install streamlit pandas numpy plotly statsmodels pmdarima prophet

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
import os
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from prophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# === Config ===
warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="Hidrandina - Integrador", page_icon="⚡", initial_sidebar_state="expanded")

st.title("Pantalla de Predicción")

# Sidebar
st.sidebar.image("Logo.png", use_column_width=True)
st.sidebar.header("Hidrandina")
st.sidebar.write("Pronóstico de Consumo Eléctrico en base a series de tiempo")
st.sidebar.selectbox("Departamento", ["Ancash"], index=0)

# === Funciones ===
def obtener_provincias():
    archivos = os.listdir("Data")
    provincias = [archivo.replace(".csv", "") for archivo in archivos if archivo.endswith(".csv")]
    provincias = list(set(provincias))  # Elimina duplicados
    provincias = [p for p in provincias if p.lower() != "total"]  # Excluye 'Total.csv' del menú
    provincias.sort()
    provincias = ["Todas"] + provincias  # Muestra 'Todas' al usuario
    return provincias

def cargar_datos(nombre):
    if nombre == "Todas":
        nombre = "Total"  # Usamos el CSV 'Total.csv' cuando el usuario elige "Todas"
    df = pd.read_csv(f"Data/{nombre}.csv")
    df['PERIODO'] = pd.to_datetime(df['PERIODO'].astype(str), format='%Y%m')
    return df

def calcular_metricas(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return mse, mae, rmse, mape

def mostrar_metricas(mse, mae, rmse, mape):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAPE (%)", f"{round(mape, 2)}%")
    col2.metric("MAE", round(mae, 2))
    col3.metric("RMSE", round(rmse, 2))
    col4.metric("MSE", round(mse, 2))

def feedback_automatico(mape, mae, rmse, mse):
    comentarios = []
    if mape < 5:
        comentarios.append("✅ Alta precisión (<5%). Puedes tomar decisiones con confianza.")
    elif mape < 10:
        comentarios.append("🟡 Precisión aceptable (<10%). Apto para decisiones generales.")
    elif mape < 20:
        comentarios.append("⚠️ Precisión moderada. Usa con precaución o refina el modelo.")
    else:
        comentarios.append("❌ Precisión baja (>20%). No apto para decisiones críticas.")

    comentarios.append(f"📏 MAE: `{mae:.2f}` mil kWh (error promedio absoluto)")
    comentarios.append(f"📉 RMSE: `{rmse:.2f}` mil kWh (impacto de errores grandes)")
    comentarios.append(f"📐 MSE: `{mse:.2f}` mil² kWh² (error cuadrático medio)")

    if mape < 10 and mae < 2:
        recomendaciones = "🟢 Recomendado para planificación mensual y alertas operativas."
    elif mape < 20:
        recomendaciones = "🟡 Usa con margen de seguridad."
    else:
        recomendaciones = "🔴 No recomendado para decisiones sin validación."

    return "\n\n".join(comentarios + ["", recomendaciones])

def detectar_outliers(serie, threshold=2.5):
    descomposicion = seasonal_decompose(serie, model='additive')
    resid = descomposicion.resid
    std = resid.std()
    outliers = abs(resid) > threshold * std
    return outliers

def aplicar_suavizado_outliers(serie):
    outliers = detectar_outliers(serie)
    serie_suavizada = serie.copy()
    serie_suavizada[outliers] = np.nan
    return serie_suavizada.interpolate()

def aplicar_holt_winters(serie):
    modelo = ExponentialSmoothing(serie, trend='add', seasonal='add', seasonal_periods=12)
    ajuste = modelo.fit()
    return ajuste.fittedvalues

def mostrar_comparacion(fechas_futuras, pred_arima, sarima_forecast, prophet_forecast):
    df_resultado = pd.DataFrame({
        "Periodo": fechas_futuras.strftime('%Y-%m'),
        "ARIMA": pred_arima,
        "SARIMA": sarima_forecast,
        "Prophet": prophet_forecast[prophet_forecast['ds'].isin(fechas_futuras)]['yhat'].values
    })
    st.caption("Comparación de predicciones para los próximos 3 meses")
    st.dataframe(df_resultado.set_index("Periodo"), use_container_width=True)

# === Interfaz ===
provincias = obtener_provincias()
provincia_sel = st.sidebar.selectbox("Provincia", provincias, index=0)
df = cargar_datos(provincia_sel)
df.rename(columns={"CONSUMO_PROMEDIO": "CONSUMO TOTAL", "IMPORTE_PROMEDIO": "IMPORTE TOTAL"}, inplace=True)
df['CONSUMO TOTAL'] = df['CONSUMO TOTAL'].round(1)
df['IMPORTE TOTAL'] = df['IMPORTE TOTAL'].round(1)

# Serie original
serie_original = df.set_index("PERIODO")["CONSUMO TOTAL"].asfreq("MS")
outlier_info = detectar_outliers(serie_original)

with st.expander("⚙️ Opciones de Preprocesamiento de Serie de Tiempo"):

    # Serie final a usar
    serie = serie_original.copy()

    aplicar_outliers = st.checkbox("Eliminar outliers mediante interpolación", value=True)

    if aplicar_outliers:
        serie = aplicar_suavizado_outliers(serie_original)
        st.success("✅ Outliers eliminados mediante interpolación.")
    
    else:
        serie = serie.copy()

    st.caption("🔍 Se aplicó interpolación para eliminar outliers detectados.")
    serie_suavizada = aplicar_holt_winters(serie_original)
    st.caption("📉 Se aplicó suavizado.")

# Comparación visual
    st.caption("🔍 Comparación: Original vs Suavizada")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=serie_original.index, y=serie_original, name="Original"))
    fig.add_trace(go.Scatter(x=serie_suavizada.index, y=serie_suavizada, name="Suavizada", line=dict(color="green")))
    fig.update_layout(title="Serie de Consumo (Original vs. Suavizada)", xaxis_title="Fecha", yaxis_title="Consumo Total (mil kWh)")
    st.plotly_chart(fig)


# === Alerta automática por diferencias significativas
diferencias = np.abs(serie_original - serie_suavizada)
umbral = 4 * diferencias.std()

anomalos = diferencias > umbral
n_anomalias = anomalos.sum()

if n_anomalias > 0:
    st.warning(f"⚠️ Se detectaron {n_anomalias} posibles anomalías entre la serie original y la suavizada.")

    # Mostrar tabla con detalles
    df_anomalias = pd.DataFrame({
        "Fecha": serie_original.index[anomalos],
        "Valor Original": serie_original[anomalos],
        "Suavizado": serie_suavizada[anomalos],
        "Diferencia Absoluta": diferencias[anomalos]
    }).reset_index(drop=True)
    df_anomalias["Fecha"] = df_anomalias["Fecha"].dt.strftime("%Y-%m")
    
    st.dataframe(df_anomalias, use_container_width=True)
else:
    st.info("✅ No se detectaron anomalías importantes al comparar la serie original con la suavizada.")

# Resumen de datos
st.subheader(f"Pronóstico de Consumo - Provincia: {provincia_sel}")
st.caption("Valores expresados en miles. Ejemplo: 12.5 equivale a 12,500 kWh.")
with st.expander("📅 Resumen de Datos"):
    df_vista = df.copy()
    df_vista["PERIODO"] = df_vista["PERIODO"].dt.strftime('%Y-%m')
    st.dataframe(df_vista, use_container_width=True)

# === Modelado ===
fechas_futuras = pd.date_range(serie.index[-1] + pd.offsets.MonthBegin(1), periods=3, freq='MS')

modelo_arima_auto = pm.auto_arima(
    serie,
    start_p=0, start_q=0,
    max_p=3, max_q=3,
    d=None, seasonal=False,
    trace=False, error_action='ignore',
    suppress_warnings=True, stepwise=True
)

pred_arima = modelo_arima_auto.predict(n_periods=3)

modelo_sarima = pm.auto_arima(
    serie,
    start_p=0, start_q=0,
    max_p=3, max_q=3,
    d=None, seasonal=True,
    m=12, D=1, 
    trace=False, suppress_warnings=True
)
sarima_forecast, sarima_confint = modelo_sarima.predict(n_periods=3, return_conf_int=True)

df_prophet = pd.DataFrame({
    "ds": serie.index,
    "y": serie.values
})
modelo_prophet = Prophet()
modelo_prophet.fit(df_prophet)
future = modelo_prophet.make_future_dataframe(periods=3, freq='MS')
prophet_forecast = modelo_prophet.predict(future)

# === Pestañas ===
tabs = st.tabs(["ARIMA", "SARIMA", "Prophet"])

# ARIMA
with tabs[0]:
    # Usando auto_arima con pmdarima
    forecast, conf_int = modelo_arima_auto.predict(n_periods=3, return_conf_int=True)


    y_true = serie[-3:]
    y_pred = pred_arima
    mse, mae, rmse, mape = calcular_metricas(y_true, y_pred)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=serie.index, y=serie, 
    name='Histórico'))

    # Pronóstico ARIMA
    fig.add_trace(go.Scatter(x=fechas_futuras, y=pred_arima, name='Pronóstico (ARIMA)', line=dict(color='blue')))

    # Banda de confianza
    fig.add_trace(go.Scatter(x=fechas_futuras, y=conf_int[:, 0], showlegend=False, line=dict(color='blue', dash='dot')))

    fig.add_trace(go.Scatter(x=fechas_futuras, y=conf_int[:, 1], showlegend=False, fill='tonexty', line=dict(color='blue', dash='dot')))


    fig.update_layout(
        title='Pronóstico con ARIMA',
        xaxis_title='Fecha',
        yaxis_title='Consumo Total (mil kWh)',
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig)

    mostrar_comparacion(fechas_futuras, pred_arima, sarima_forecast, prophet_forecast)
    mostrar_metricas(mse, mae, rmse, mape)
    st.info(feedback_automatico(mape, mae, rmse, mse))

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
    fig.update_layout(title='Pronóstico con SARIMA', xaxis_title='Fecha', yaxis_title='Consumo Total (mil kWh)',legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
    st.plotly_chart(fig)
    mostrar_comparacion(fechas_futuras, pred_arima, sarima_forecast, prophet_forecast)
    mostrar_metricas(mse, mae, rmse, mape)
    st.info(feedback_automatico(mape, mae, rmse, mse))

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
    fig.update_layout(title='Pronóstico con Prophet', xaxis_title='Fecha', yaxis_title='Consumo Total (mil kWh)',legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
    st.plotly_chart(fig)
    mostrar_comparacion(fechas_futuras, pred_arima, sarima_forecast, prophet_forecast)
    mostrar_metricas(mse, mae, rmse, mape)
    st.info(feedback_automatico(mape, mae, rmse, mse))
