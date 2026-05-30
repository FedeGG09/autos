import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="AutoValue IA", page_icon="🚗", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 2. CARGA DE ACTIVOS CON VERIFICACIÓN ROBUSTA ---
@st.cache_resource
def load_assets():
    model_path = os.path.join(BASE_DIR, 'modelo_autos.pkl')
    text_path = os.path.join(BASE_DIR, 'car_pricing.csv')
    encoded_path = os.path.join(BASE_DIR, 'car_pricing_encoded.csv')
    
    # Validaciones de seguridad para mostrar mensajes limpios en la interfaz si falta algo
    if not os.path.exists(model_path):
        st.error(f"❌ No se encontró el archivo del modelo: `{model_path}`. Asegúrate de que subió correctamente mediante Git LFS.")
        st.stop()
    if not os.path.exists(text_path):
        st.error(f"❌ Falta el archivo `{text_path}` en tu repositorio de GitHub. Súbelo para activar los selectores de texto.")
        st.stop()
    if not os.path.exists(encoded_path):
        st.error(f"❌ Falta el archivo `{encoded_path}` en tu repositorio. Súbelo desde tu Colab para corregir la desviación.")
        st.stop()
        
    model = joblib.load(model_path)
    df_text = pd.read_csv(text_path)
    df_encoded = pd.read_csv(encoded_path, index_col=0)
    return model, df_text, df_encoded

# Ejecuta la carga segura
model, df, df_encoded = load_assets()

# --- 3. INTERFAZ DE USUARIO ---
st.image("autovalue.jpg", width=200)
st.title("Simulador AutoValue")
st.markdown("Herramienta de tasación inteligente con consistencia matemática total (Mismo resultado que en Colab).")

# --- 4. FILTRADO JERÁRQUICO EN CASCADA ---
manuf = st.selectbox("Manufacturer:", sorted(df['Manufacturer'].unique()))
df_f = df[df['Manufacturer'] == manuf]

modelo = st.selectbox("Model:", sorted(df_f['Model'].unique()))
df_f = df_f[df_f['Model'] == modelo]

cat = st.selectbox("Category:", sorted(df_f['Category'].unique()))
df_f = df_f[df_f['Category'] == cat]

fuel = st.selectbox("Fuel type:", sorted(df_f['Fuel type'].unique()))
df_f = df_f[df_f['Fuel type'] == fuel]

gear = st.selectbox("Gear box type:", sorted(df_f['Gear box type'].unique()))

# --- 5. PROCESAMIENTO Y PREDICCIÓN DE ALTA PRECISIÓN ---
if st.button("💰 Calcular Precio"):
    resultado = df_f[df_f['Gear box type'] == gear]
    
    if not resultado.empty:
        # 1. Obtenemos el índice exacto de la fila que el usuario seleccionó
        idx_original = resultado.index[0]
        
        # 2. Buscamos esa misma fila en el dataframe codificado numéricamente
        fila_encoded = df_encoded.loc[[idx_original]]
        
        # 3. Filtramos ordenando las columnas exactamente como el modelo las espera
        datos_input = fila_encoded[model.feature_names_in_]
        
        # 4. Inferencia pura con los mismos números del entrenamiento
        prediccion = model.predict(datos_input)[0]
        precio_real = resultado.iloc[0]['Price']
        
        # Renderizado de métricas en la UI
        st.success(f"✅ Auto localizado con éxito (Historial ID: {idx_original})")
        
        col1, col2 = st.columns(2)
        col1.metric("🚀 Predicción IA", f"${prediccion:,.2f}")
        col2.metric("💵 Precio Real", f"${precio_real:,.2f}")
        
        # Al usar los mismos datos indexados, la desviación volverá a los valores óptimos de tu Colab
        st.metric("📊 Desviación Absoluta", f"${abs(prediccion - precio_real):,.2f}")
        
    else:
        st.warning("No se encontraron coincidencias exactas para esta combinación en el historial.")

# --- 6. FOOTER ---
st.markdown("---")
st.caption("AutoValue IA | Modo de Sincronización por Índices Activo.")
