import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="AutoValue IA", page_icon="🚗")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_assets():
    model = joblib.load(os.path.join(BASE_DIR, 'modelo_autos.pkl'))
    # Dataset con texto para los selectores
    df_text = pd.read_csv(os.path.join(BASE_DIR, 'car_pricing.csv'))
    # Dataset codificado para el modelo (usamos la primera columna como índice)
    df_encoded = pd.read_csv(os.path.join(BASE_DIR, 'car_pricing_encoded.csv'), index_col=0)
    return model, df_text, df_encoded

model, df, df_encoded = load_assets()

# --- 2. UI Y FILTROS EN CASCADA ---
st.image("autovalue.jpg", width=200)
st.title("Simulador AutoValue")
st.markdown("Herramienta de tasación inteligente con consistencia de datos en la nube.")

# Filtrado jerárquico real
manuf = st.selectbox("Manufacturer:", sorted(df['Manufacturer'].unique()))
df_f = df[df['Manufacturer'] == manuf]

modelo = st.selectbox("Model:", sorted(df_f['Model'].unique()))
df_f = df_f[df_f['Model'] == modelo]

cat = st.selectbox("Category:", sorted(df_f['Category'].unique()))
df_f = df_f[df_f['Category'] == cat]

fuel = st.selectbox("Fuel type:", sorted(df_f['Fuel type'].unique()))
df_f = df_f[df_f['Fuel type'] == fuel]

gear = st.selectbox("Gear box type:", sorted(df_f['Gear box type'].unique()))

# --- 3. PREDICCIÓN CON CONSISTENCIA DE ÍNDICES ---
if st.button("💰 Calcular Precio"):
    resultado = df_f[df_f['Gear box type'] == gear]
    
    if not resultado.empty:
        # 1. Identificamos el índice original del auto encontrado
        idx_original = resultado.index[0]
        
        # 2. Extraemos la fila codificada exacta desde df_encoded usando ese índice
        fila_encoded = df_encoded.loc[[idx_original]]
        
        # 3. Nos aseguramos de pasarle al modelo solo las columnas con las que fue entrenado
        datos_input = fila_encoded[model.feature_names_in_]
        
        # 4. Inferencia con datos reales transformados
        prediccion = model.predict(datos_input)[0]
        precio_real = resultado.iloc[0]['Price']
        
        # Resultados UI
        st.success(f"✅ Auto encontrado en el histórico (Índice: {idx_original})")
        
        col1, col2 = st.columns(2)
        col1.metric("🚀 Predicción IA", f"${prediccion:,.2f}")
        col2.metric("💵 Precio Real", f"${precio_real:,.2f}")
        
        st.metric("📊 Desviación Absoluta", f"${abs(prediccion - precio_real):,.2f}")
        
    else:
        st.warning("No se encontraron coincidencia exactas en el historial.")

# --- 4. PREDICCIÓN ---
if st.button("💰 Calcular Precio"):
    resultado = df_f[df_f['Gear box type'] == gear]
    
    if not resultado.empty:
        auto_seleccionado = resultado.iloc[0:1].copy()
        
        # Usamos la función robusta que evita el array de 0 samples
        input_final = construir_fila_prediccion(auto_seleccionado, model.feature_names_in_)
        
        # Inferencia
        prediccion = model.predict(input_final)[0]
        precio_real = auto_seleccionado['Price'].iloc[0]
        
        # Resultados UI
        st.success(f"✅ Predicción de la IA: ${prediccion:,.2f}")
        st.metric("Precio Real en Histórico", f"${precio_real:,.2f}")
        st.metric("Desviación Absoluta", f"${abs(prediccion - precio_real):,.2f}")
        
    else:
        st.warning("No se encontraron coincidencias exactas en el historial.")
