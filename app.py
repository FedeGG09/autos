import streamlit as st
import pandas as pd
import joblib

# 1. Configuración de la App
st.set_page_config(page_title="AutoValue IA", page_icon="🚗")

# 2. Carga de recursos
@st.cache_resource
def load_assets():
    model = joblib.load('modelo_autos.pkl')
    df = pd.read_csv('car_pricing.csv')
    # Nota: Asegúrate de tener df_encoded cargado si el modelo lo requiere
    return model, df

model, df = load_assets()

# 3. Logo y Título
st.image("autovalue.jpg", width=200)
st.title("Simulador de Precios AutoValue")

# 4. Selectores (Lógica de filtrado)
manuf = st.selectbox("Manufacturer:", sorted(df['Manufacturer'].unique()))
model_options = sorted(df[df['Manufacturer'] == manuf]['Model'].unique())
modelo = st.selectbox("Model:", model_options)
cat = st.selectbox("Category:", sorted(df['Category'].unique()))
fuel = st.selectbox("Fuel type:", sorted(df['Fuel type'].unique()))
gear = st.selectbox("Gear box type:", sorted(df['Gear box type'].unique()))

# 5. Función de Comparación y Predicción
if st.button("💰 Calcular y Comparar"):
    # Filtro
    resultado = df[(df['Manufacturer'] == manuf) & 
                   (df['Model'] == modelo) & 
                   (df['Category'] == cat) & 
                   (df['Fuel type'] == fuel) & 
                   (df['Gear box type'] == gear)]
    
    if not resultado.empty:
        # Extraer el auto del dataset
        auto = resultado.iloc[0]
        
        # Preparar para el modelo (usando las features que espera)
        # Importante: aquí pasamos el auto a DataFrame para que el modelo lo entienda
        datos_input = auto.to_frame().T[model.feature_names_in_]
        
        prediccion = model.predict(datos_input)[0]
        precio_real = auto['Price']
        
        # Visualización de resultados
        st.success(f"✅ Auto encontrado: {auto['Manufacturer']} {auto['Model']}")
        
        col1, col2 = st.columns(2)
        col1.metric("🚀 Predicción IA", f"${prediccion:,.2f}")
        col2.metric("💵 Precio Real", f"${precio_real:,.2f}")
        
        st.metric("📊 Diferencia absoluta", f"${abs(prediccion - precio_real):,.2f}")
        
    else:
        st.warning("No se encontraron coincidencias exactas en el historial.")

st.markdown("---")
st.write("Herramienta de validación para Equipo de Ventas y Marketing.")