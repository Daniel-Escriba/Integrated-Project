import streamlit as st

# Test básico
st.title("🚀 Test de Conexión")
st.write("Si ves esto, la app está funcionando")

# Test de pandas y CSV
try:
    import pandas as pd
    st.success("✅ Pandas importado correctamente")
    
    # Intentar cargar el CSV
    df = pd.read_csv('data_consumo_ancash.csv')
    st.success("✅ CSV cargado correctamente")
    st.write(f"Datos: {df.shape[0]} filas, {df.shape[1]} columnas")
    st.dataframe(df.head())
    
except FileNotFoundError:
    st.error("❌ No se encontró el archivo data_consumo_ancash.csv")
except Exception as e:
    st.error(f"❌ Error: {str(e)}")

# Test de plotly (si está en requirements)
try:
    import plotly.express as px
    st.success("✅ Plotly importado correctamente")
    
    # Solo si el CSV se cargó
    if 'df' in locals():
        # Gráfico simple (ajusta las columnas según tu CSV)
        if len(df.columns) >= 2:
            fig = px.bar(df.head(10), x=df.columns[0], y=df.columns[1])
            st.plotly_chart(fig)
        
except ImportError:
    st.warning("⚠️ Plotly no disponible")
except Exception as e:
    st.error(f"❌ Error con Plotly: {str(e)}")