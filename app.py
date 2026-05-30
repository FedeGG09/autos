import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="AutoValue IA", page_icon="🚗")

# Rutas robustas para evitar el FileNotFoundError
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'modelo_autos.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'car_pricing.csv')

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    return model, df

model, df = load_assets()

# --- 2. FUNCIÓN ROBUSTA DE TRANSFORMACIÓN ---
def construir_fila_prediccion(auto_row, expected_features):
    """
    Construye un DataFrame de exactamente 1 fila (shape 1, N) 
    asegurando que solo contenga datos numéricos y coincida 
    con las columnas que el modelo exige.
    """
    # Si viene como DataFrame de 1 fila, lo pasamos a Series
    if isinstance(auto_row, pd.DataFrame):
        auto_row = auto_row.iloc[0]
        
    datos_seguros = {}
    
    for col in expected_features:
        if col in auto_row.index:
            valor = auto_row[col]
            # Si el valor es texto (string), ponemos 0 como fallback de seguridad
            if isinstance(valor, str):
                datos_seguros[col] = [0]
            else:
                # Si es numérico, lo usamos (o 0 si viene nulo/NaN)
                datos_seguros[col] = [0 if pd.isna(valor) else valor]
        else:
            # Si la columna no existe en el dataset, le ponemos 0
            datos_seguros[col] = [0]
            
    # Al pasar listas [valor], Pandas asegura que se cree exactamente 1 fila
    return pd.DataFrame(datos_seguros)

# --- 3. UI Y FILTROS EN CASCADA ---
st.image("autovalue.jpg", width=200)
st.title("Simulador AutoValue")

manuf = st.selectbox("Manufacturer:", sorted(df['Manufacturer'].unique()))
df_f = df[df['Manufacturer'] == manuf]

modelo = st.selectbox("Model:", sorted(df_f['Model'].unique()))
df_f = df_f[df_f['Model'] == modelo]

cat = st.selectbox("Category:", sorted(df_f['Category'].unique()))
df_f = df_f[df_f['Category'] == cat]

fuel = st.selectbox("Fuel type:", sorted(df_f['Fuel type'].unique()))
df_f = df_f[df_f['Fuel type'] == fuel]

gear = st.selectbox("Gear box type:", sorted(df_f['Gear box type'].unique()))

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
