import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="AutoValue IA", page_icon="🚗", layout="centered")

# --- 2. CARGA DE ACTIVOS ---
@st.cache_resource
def load_assets():
    # Rutas relativas al archivo app.py
    model = joblib.load('modelo_autos.pkl')
    df = pd.read_csv('car_pricing.csv')
    return model, df

model, df = load_assets()

# --- 3. UI Y LOGO ---
st.image("autovalue.jpg", width=250)
st.title("Simulador de Precios AutoValue")
st.markdown("Herramienta de tasación inteligente para el equipo de ventas.")

# --- 4. FILTRADO JERÁRQUICO EN CASCADA ---
# Manufacturer
manuf = st.selectbox("Manufacturer:", sorted(df['Manufacturer'].unique()))

# Model (filtrado)
df_manuf = df[df['Manufacturer'] == manuf]
modelo = st.selectbox("Model:", sorted(df_manuf['Model'].unique()))

# Category (filtrado)
df_model = df_manuf[df_manuf['Model'] == modelo]
cat = st.selectbox("Category:", sorted(df_model['Category'].unique()))

# Fuel type (filtrado)
df_cat = df_model[df_model['Category'] == cat]
fuel = st.selectbox("Fuel type:", sorted(df_cat['Fuel type'].unique()))

# Gear box type (filtrado)
df_fuel = df_cat[df_cat['Fuel type'] == fuel]
gear = st.selectbox("Gear box type:", sorted(df_fuel['Gear box type'].unique()))

# --- 5. LÓGICA DE PREDICCIÓN Y COMPARACIÓN ---
if st.button("💰 Calcular y Comparar"):
    resultado = df_fuel[df_fuel['Gear box type'] == gear]
    
    if not resultado.empty:
        auto_df = resultado.iloc[0:1].copy()
        
        # --- CORRECCIÓN CRÍTICA: Ajuste de columnas para el modelo ---
        # Si el modelo espera columnas que no están en el CSV (ej. Antiguedad_Anios),
        # las creamos con 0 para evitar el KeyError.
        for col in model.feature_names_in_:
            if col not in auto_df.columns:
                auto_df[col] = 0
        
        # Filtramos solo las columnas que el modelo conoce
        datos_input = auto_df[model.feature_names_in_]
        
        # Inferencia
        prediccion = model.predict(datos_input)[0]
        precio_real = auto_df.iloc[0]['Price']
        
        # UI Resultados
        st.success(f"✅ Auto encontrado: {auto_df.iloc[0]['Manufacturer']} {auto_df.iloc[0]['Model']}")
        
        col1, col2 = st.columns(2)
        col1.metric("🚀 Predicción IA", f"${prediccion:,.2f}")
        col2.metric("💵 Precio Real", f"${precio_real:,.2f}")
        
        # Cálculo de diferencia
        diferencia = abs(prediccion - precio_real)
        st.metric("📊 Diferencia absoluta", f"${diferencia:,.2f}")
        
    else:
        st.warning("No se encontraron coincidencias exactas en el historial.")

# --- 6. FOOTER ---
st.markdown("---")
st.caption("AutoValue IA | Desarrollado para optimizar el ciclo de ventas. Precisión objetivo: MAPE < 12%.")
