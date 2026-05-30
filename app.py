import streamlit as st
import pandas as pd
import joblib

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="AutoValue IA", page_icon="🚗", layout="centered")

# --- 2. CARGA DE MODELO Y DATOS ---
@st.cache_resource
def load_assets():
    # Asegúrate de que estos archivos estén en la raíz de tu repositorio
    model = joblib.load('modelo_autos.pkl')
    df = pd.read_csv('car_pricing.csv')
    return model, df

model, df = load_assets()

# --- 3. UI Y LOGO ---
st.image("autovalue.jpg", width=250)
st.title("Simulador de Precios AutoValue")
st.markdown("Herramienta de tasación inteligente para el equipo de ventas.")

# --- 4. FILTRADO JERÁRQUICO (Lógica en Cascada) ---
# Manufacturer
manuf = st.selectbox("Manufacturer:", sorted(df['Manufacturer'].unique()))

# Model (filtrado por Manufacturer)
df_manuf = df[df['Manufacturer'] == manuf]
modelo = st.selectbox("Model:", sorted(df_manuf['Model'].unique()))

# Category (filtrado por Model)
df_model = df_manuf[df_manuf['Model'] == modelo]
cat = st.selectbox("Category:", sorted(df_model['Category'].unique()))

# Fuel type (filtrado por Category)
df_cat = df_model[df_model['Category'] == cat]
fuel = st.selectbox("Fuel type:", sorted(df_cat['Fuel type'].unique()))

# Gear box type (filtrado por Fuel type)
df_fuel = df_cat[df_cat['Fuel type'] == fuel]
gear = st.selectbox("Gear box type:", sorted(df_fuel['Gear box type'].unique()))

# --- 5. LÓGICA DE PREDICCIÓN Y COMPARACIÓN ---
if st.button("💰 Calcular y Comparar"):
    # Obtener el subconjunto final
    resultado = df_fuel[df_fuel['Gear box type'] == gear]
    
    if not resultado.empty:
        auto = resultado.iloc[0]
        
        # Preparar datos: Filtramos solo las columnas que el modelo conoce
        # Esto evita el error de "feature names mismatch"
        datos_input = auto.to_frame().T[model.feature_names_in_]
        
        # Inferencia
        prediccion = model.predict(datos_input)[0]
        precio_real = auto['Price']
        
        # UI Resultados
        st.success(f"✅ Auto encontrado: {auto['Manufacturer']} {auto['Model']}")
        
        col1, col2 = st.columns(2)
        col1.metric("🚀 Predicción IA", f"${prediccion:,.2f}")
        col2.metric("💵 Precio Real", f"${precio_real:,.2f}")
        
        # Cálculo de métrica de negocio (diferencia)
        st.metric("📊 Diferencia absoluta", f"${abs(prediccion - precio_real):,.2f}")
        
    else:
        st.warning("No se encontraron coincidencias exactas en el historial.")

# --- 6. FOOTER PARA STAKEHOLDERS ---
st.markdown("---")
st.caption("AutoValue IA | Desarrollado para optimizar el ciclo de ventas. Precisión objetivo: MAPE < 12%.")
