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
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# === Config ===
warnings.filterwarnings("ignore")
st.set_page_config(
    layout="wide",
    page_title="Hidrandina - Integrador",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# === Estilos CSS ===
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: #1f77b4; }
    div[data-testid="stExpander"] summary { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Pantalla de Predicción")

# === Sidebar ===
st.sidebar.image("logo.png", use_container_width=True)  # ✅ FIX: use_column_width → use_container_width
st.sidebar.header("Hidrandina")
st.sidebar.write("Pronóstico de Consumo Eléctrico en base a series de tiempo")
st.sidebar.selectbox("Departamento", ["Ancash"], index=0)

# === Funciones de datos ===
def obtener_provincias():
    archivos = os.listdir("Data")
    provincias = [f.replace(".csv", "") for f in archivos if f.endswith(".csv")]
    provincias = list(set(provincias))
    provincias = [p for p in provincias if p.lower() != "total"]
    provincias.sort()
    return ["Todas"] + provincias

@st.cache_resource(show_spinner="Cargando datos...")
def cargar_datos(nombre: str) -> pd.DataFrame:
    if nombre == "Todas":
        nombre = "Total"
    df = pd.read_csv(f"Data/{nombre}.csv")
    df["PERIODO"] = pd.to_datetime(df["PERIODO"].astype(str), format="%Y%m")
    return df

# === Métricas ===
def calcular_metricas(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return mse, mae, rmse, mape

def mostrar_metricas(mse, mae, rmse, mape):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAPE (%)", f"{mape:.2f}%")
    col2.metric("MAE",      f"{mae:.2f}")
    col3.metric("RMSE",     f"{rmse:.2f}")
    col4.metric("MSE",      f"{mse:.2f}")

def feedback_automatico(mape, mae, rmse, mse):
    if mape < 5:
        nivel = "✅ Alta precisión (<5%). Puedes tomar decisiones con confianza."
    elif mape < 10:
        nivel = "🟡 Precisión aceptable (<10%). Apto para decisiones generales."
    elif mape < 20:
        nivel = "⚠️ Precisión moderada. Usa con precaución o refina el modelo."
    else:
        nivel = "❌ Precisión baja (>20%). No apto para decisiones críticas."

    if mape < 10 and mae < 2:
        recomendacion = "🟢 Recomendado para planificación mensual y alertas operativas."
    elif mape < 20:
        recomendacion = "🟡 Usa con margen de seguridad."
    else:
        recomendacion = "🔴 No recomendado para decisiones sin validación adicional."

    return (
        f"{nivel}\n\n"
        f"📏 MAE: `{mae:.2f}` mil kWh — error promedio absoluto\n\n"
        f"📉 RMSE: `{rmse:.2f}` mil kWh — impacto de errores grandes\n\n"
        f"📐 MSE: `{mse:.2f}` mil² kWh² — error cuadrático medio\n\n"
        f"{recomendacion}"
    )

# === Preprocesamiento ===
def detectar_outliers(serie, threshold=2.5):
    descomposicion = seasonal_decompose(serie, model="additive")
    resid = descomposicion.resid
    return abs(resid) > threshold * resid.std()

def aplicar_suavizado_outliers(serie):
    outliers = detectar_outliers(serie)
    serie_suavizada = serie.copy()
    serie_suavizada[outliers] = np.nan
    return serie_suavizada.interpolate()

def aplicar_holt_winters(serie, seasonal_periods=12):
    modelo = ExponentialSmoothing(serie, trend="add", seasonal="add", seasonal_periods=seasonal_periods)
    return modelo.fit().fittedvalues

# === Modelado (cacheado) ===
@st.cache_resource(show_spinner="Entrenando ARIMA...")
def entrenar_arima(serie_values, serie_index):
    serie = pd.Series(serie_values, index=serie_index)
    modelo = pm.auto_arima(
        serie,
        start_p=0, start_q=0, max_p=3, max_q=3,
        d=None, seasonal=False,
        trace=False, error_action="ignore",
        suppress_warnings=True, stepwise=True
    )
    forecast, conf_int = modelo.predict(n_periods=3, return_conf_int=True)
    fitted = modelo.predict_in_sample()
    return forecast, conf_int, fitted

@st.cache_resource(show_spinner="Entrenando SARIMA...")
def entrenar_sarima(serie_values, serie_index):
    serie = pd.Series(serie_values, index=serie_index)
    modelo = pm.auto_arima(
        serie,
        start_p=0, start_q=0, max_p=3, max_q=3,
        d=None, seasonal=True, m=12, D=1,
        trace=False, suppress_warnings=True, stepwise=True
    )
    forecast, conf_int = modelo.predict(n_periods=3, return_conf_int=True)
    fitted = modelo.predict_in_sample()
    return forecast, conf_int, fitted

@st.cache_resource(show_spinner="Entrenando Prophet...")
def entrenar_prophet(serie_values, serie_index):
    df_prophet = pd.DataFrame({"ds": serie_index, "y": serie_values})
    modelo = Prophet()
    modelo.fit(df_prophet)
    future = modelo.make_future_dataframe(periods=3, freq="MS")
    forecast = modelo.predict(future)
    return forecast

# === Tabla comparativa ===
def mostrar_comparacion(fechas_futuras, pred_arima, pred_sarima, prophet_forecast):
    prophet_vals = prophet_forecast[prophet_forecast["ds"].isin(fechas_futuras)]["yhat"].values
    df_res = pd.DataFrame({
        "Periodo": fechas_futuras.strftime("%Y-%m"),
        "ARIMA":   np.round(pred_arima, 2),
        "SARIMA":  np.round(pred_sarima, 2),
        "Prophet": np.round(prophet_vals, 2),
    })
    st.caption("📊 Comparación de predicciones para los próximos 3 meses")
    st.dataframe(df_res.set_index("Periodo"), use_container_width=True)

# ============================================================
# === INTERFAZ PRINCIPAL ===
# ============================================================
provincias    = obtener_provincias()
provincia_sel = st.sidebar.selectbox("Provincia", provincias, index=0)

df = cargar_datos(provincia_sel)
df.rename(columns={
    "CONSUMO_PROMEDIO": "CONSUMO TOTAL",
    "IMPORTE_PROMEDIO": "IMPORTE TOTAL"
}, inplace=True)
df["CONSUMO TOTAL"] = df["CONSUMO TOTAL"].round(1)
df["IMPORTE TOTAL"] = df["IMPORTE TOTAL"].round(1)

serie_original = df.set_index("PERIODO")["CONSUMO TOTAL"].asfreq("MS")

# === Preprocesamiento ===
with st.expander("⚙️ Opciones de Preprocesamiento de Serie de Tiempo", expanded=False):
    aplicar_outliers = st.checkbox("Eliminar outliers mediante interpolación", value=True)

    if aplicar_outliers:
        serie = aplicar_suavizado_outliers(serie_original)
        st.success("✅ Outliers eliminados mediante interpolación.")
    else:
        serie = serie_original.copy()

    serie_suavizada = aplicar_holt_winters(serie_original)

    st.caption("🔍 Comparación: Original vs Suavizada")
    fig_pre = go.Figure()
    fig_pre.add_trace(go.Scatter(x=serie_original.index, y=serie_original, name="Original", line=dict(color="#636EFA")))
    fig_pre.add_trace(go.Scatter(x=serie_suavizada.index, y=serie_suavizada, name="Suavizada", line=dict(color="#00CC96")))
    fig_pre.update_layout(
        title="Serie de Consumo (Original vs. Suavizada)",
        xaxis_title="Fecha", yaxis_title="Consumo Total (mil kWh)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        margin=dict(t=40, b=20)
    )
    st.plotly_chart(fig_pre, use_container_width=True)

# === Anomalías ===
diferencias = np.abs(serie_original - serie_suavizada)
umbral      = 4 * diferencias.std()
anomalos    = diferencias > umbral

if anomalos.sum() > 0:
    st.warning(f"⚠️ Se detectaron {anomalos.sum()} posibles anomalías entre la serie original y la suavizada.")
    df_anomalias = pd.DataFrame({
        "Fecha":               serie_original.index[anomalos].strftime("%Y-%m"),
        "Valor Original":      serie_original[anomalos].round(2).values,
        "Suavizado":           serie_suavizada[anomalos].round(2).values,
        "Diferencia Absoluta": diferencias[anomalos].round(2).values,
    })
    st.dataframe(df_anomalias, use_container_width=True)
else:
    st.info("✅ No se detectaron anomalías importantes al comparar la serie original con la suavizada.")

# === Resumen ===
st.subheader(f"📍 Pronóstico de Consumo — Provincia: {provincia_sel}")
st.caption("Valores expresados en miles de kWh. Ejemplo: 12.5 → 12,500 kWh.")

with st.expander("📅 Resumen de Datos Históricos"):
    df_vista = df.copy()
    df_vista["PERIODO"] = df_vista["PERIODO"].dt.strftime("%Y-%m")
    st.dataframe(df_vista, use_container_width=True)

# === Entrenamiento ===
fechas_futuras = pd.date_range(
    serie.index[-1] + pd.offsets.MonthBegin(1), periods=3, freq="MS"
)

with st.spinner("Entrenando modelos... ⏳"):
    pred_arima,  conf_arima,  fitted_arima  = entrenar_arima(serie.values, serie.index)
    pred_sarima, conf_sarima, fitted_sarima = entrenar_sarima(serie.values, serie.index)
    prophet_forecast                        = entrenar_prophet(serie.values, serie.index)

# === Pestañas ===
tabs = st.tabs(["📈 ARIMA", "📊 SARIMA", "🔮 Prophet"])

# ── ARIMA ──
with tabs[0]:
    mse, mae, rmse, mape = calcular_metricas(serie[-3:], pred_arima)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=serie.index, y=serie, name="Histórico", line=dict(color="#636EFA")))
    fig.add_trace(go.Scatter(x=fechas_futuras, y=pred_arima, name="Pronóstico (ARIMA)", line=dict(color="#1f77b4", width=2.5)))
    fig.add_trace(go.Scatter(x=fechas_futuras, y=conf_arima[:, 0], showlegend=False, line=dict(color="#1f77b4", dash="dot", width=1)))
    fig.add_trace(go.Scatter(x=fechas_futuras, y=conf_arima[:, 1], showlegend=False, fill="tonexty",
                             fillcolor="rgba(31,119,180,0.15)", line=dict(color="#1f77b4", dash="dot", width=1)))
    fig.update_layout(title="Pronóstico con ARIMA", xaxis_title="Fecha", yaxis_title="Consumo Total (mil kWh)",
                      legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
    st.plotly_chart(fig, use_container_width=True)

    mostrar_comparacion(fechas_futuras, pred_arima, pred_sarima, prophet_forecast)
    mostrar_metricas(mse, mae, rmse, mape)
    st.info(feedback_automatico(mape, mae, rmse, mse))

# ── SARIMA ──
with tabs[1]:
    mse, mae, rmse, mape = calcular_metricas(serie[-3:], fitted_sarima[-3:])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=serie.index, y=serie, name="Histórico", line=dict(color="#636EFA")))
    fig.add_trace(go.Scatter(x=fechas_futuras, y=pred_sarima, name="Pronóstico (SARIMA)", line=dict(color="#FF7F0E", width=2.5)))
    fig.add_trace(go.Scatter(x=fechas_futuras, y=conf_sarima[:, 0], showlegend=False, line=dict(color="#FF7F0E", dash="dot", width=1)))
    fig.add_trace(go.Scatter(x=fechas_futuras, y=conf_sarima[:, 1], showlegend=False, fill="tonexty",
                             fillcolor="rgba(255,127,14,0.15)", line=dict(color="#FF7F0E", dash="dot", width=1)))
    fig.update_layout(title="Pronóstico con SARIMA", xaxis_title="Fecha", yaxis_title="Consumo Total (mil kWh)",
                      legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
    st.plotly_chart(fig, use_container_width=True)

    mostrar_comparacion(fechas_futuras, pred_arima, pred_sarima, prophet_forecast)
    mostrar_metricas(mse, mae, rmse, mape)
    st.info(feedback_automatico(mape, mae, rmse, mse))

# ── Prophet ──
with tabs[2]:
    df_hist = pd.DataFrame({"ds": serie.index, "y": serie.values})
    y_true  = df_hist["y"].tail(3).values
    y_pred  = prophet_forecast[prophet_forecast["ds"].isin(df_hist["ds"].tail(3))]["yhat"].values
    mse, mae, rmse, mape = calcular_metricas(y_true, y_pred)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_hist["ds"], y=df_hist["y"], name="Histórico", line=dict(color="#636EFA")))
    fig.add_trace(go.Scatter(x=prophet_forecast["ds"], y=prophet_forecast["yhat"],
                             name="Pronóstico (Prophet)", line=dict(color="#2CA02C", width=2.5)))
    fig.add_trace(go.Scatter(x=prophet_forecast["ds"], y=prophet_forecast["yhat_lower"],
                             showlegend=False, line=dict(color="#2CA02C", dash="dot", width=1)))
    fig.add_trace(go.Scatter(x=prophet_forecast["ds"], y=prophet_forecast["yhat_upper"],
                             showlegend=False, fill="tonexty",
                             fillcolor="rgba(44,160,44,0.15)", line=dict(color="#2CA02C", dash="dot", width=1)))
    fig.update_layout(title="Pronóstico con Prophet", xaxis_title="Fecha", yaxis_title="Consumo Total (mil kWh)",
                      legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
    st.plotly_chart(fig, use_container_width=True)

    mostrar_comparacion(fechas_futuras, pred_arima, pred_sarima, prophet_forecast)
    mostrar_metricas(mse, mae, rmse, mape)
    st.info(feedback_automatico(mape, mae, rmse, mse))
