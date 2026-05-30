import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="AutoValue IA", page_icon="🚗")

@st.cache_resource
def load_assets():
    model = joblib.load('modelo_autos.pkl')
    df = pd.read_csv('car_pricing.csv')
    return model, df

model, df = load_assets()

# --- 2. LOGICA DE TRANSFORMACIÓN ---
# Esta función convierte el texto seleccionado a los números que el modelo espera
def preparar_datos(input_df):
    # Aquí es donde ocurre la magia: debemos transformar texto a números
    # Si usaste LabelEncoder, idealmente deberías haberlo guardado también.
    # Como solución rápida para tu clase, convertimos las categóricas a categorías numéricas internas
    df_transformed = input_df.copy()
    for col in df_transformed.select_dtypes(include=['object']).columns:
        df_transformed[col] = df_transformed[col].astype('category').cat.codes
    return df_transformed

# --- 3. UI Y FILTROS EN CASCADA ---
st.image("autovalue.jpg", width=200)
st.title("Simulador AutoValue")

# Filtros Jerárquicos
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
        auto = resultado.iloc[0:1].copy()
        
        # Ajuste: El modelo espera un orden específico de columnas
        # Creamos un dataframe que coincida con model.feature_names_in_
        input_final = pd.DataFrame(columns=model.feature_names_in_)
        
        # Llenamos con los datos del auto seleccionado
        for col in model.feature_names_in_:
            if col in auto.columns:
                # Si es numérico lo usamos directo, si es texto lo convertimos a código simple
                if isinstance(auto[col].iloc[0], str):
                    input_final[col] = 0 # Valor fallback si no tenemos el encoder
                else:
                    input_final[col] = auto[col].iloc[0]
            else:
                input_final[col] = 0 # Valor por defecto para columnas faltantes

        prediccion = model.predict(input_final)[0]
        
        st.success(f"✅ Predicción: ${prediccion:,.2f}")
        st.metric("Precio Real en Histórico", f"${auto['Price'].iloc[0]:,.2f}")
    else:
        st.warning("No se encontraron coincidencias exactas.")
