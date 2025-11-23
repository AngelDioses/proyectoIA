import streamlit as st
import numpy as np
import torch
import cv2
import pandas as pd
import os
import json
import requests
import base64

# Nota: Ya no se importa UNetMultiTask ni se carga el modelo localmente
# Usamos el URL base para la conexión
API_URL = "http://127.0.0.1:8000/predict_mri"
IMG_SIZE = 256 # Necesario para el resize de la imagen de visualización

# --- Función de Cliente API ---
def get_prediction_from_api(uploaded_file):
    if uploaded_file is None:
        return None, None, None
        
    # Crear un buffer de Bytes para enviar el archivo
    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    
    try:
        # Enviar la imagen a la API
        response = requests.post(API_URL, files=files)
        response.raise_for_status() 
        
        data = response.json()
        
        # Decodificar la máscara de Base64 a una imagen de NumPy (cv2)
        mask_base64 = data.get("mask_png_base64")
        mask_bytes = base64.b64decode(mask_base64)
        mask_array = np.frombuffer(mask_bytes, np.uint8)
        mask_decoded = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)

        return mask_decoded, data.get("confidence", 0), data.get("prediction", "Error")
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error de conexión con el Agente API (FastAPI). ¿Está Uvicorn corriendo en http://127.0.0.1:8000?")
        return None, None, None

# --- Función para obtener métricas del archivo local (sin cambios) ---
METRICS_FILE = 'real_metrics_pytorch.json'

def load_local_metrics():
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f:
            return json.load(f)
    return None

# --- Interfaz de Streamlit ---
st.set_page_config(page_title="Agente DL para Tumores MRI", layout="wide")

st.title("🧠 Agente de IA: Detección y Segmentación de Tumores MRI")
st.markdown("El Agente (FastAPI) ejecuta la inferencia en la **RTX 3050** y Streamlit actúa como la Interfaz de Usuario.")
st.write("---")

## 1. Reporte de Métricas (REALES)
st.header("📊 Reporte de Métricas de Prueba (Resultados Reales)")
real_metrics = load_local_metrics()

if real_metrics:
    # ... (Lógica de visualización de métricas usando real_metrics) ...
    # Usando valores de ejemplo para mantener la brevedad:
    df_metrics = pd.DataFrame({'Métrica': ['Accuracy', 'F1-Score', 'Mean IoU'], 
                                'Valor': [real_metrics['Accuracy'], real_metrics['F1-Score'], real_metrics['Mean IoU']]}).set_index('Métrica')
    st.dataframe(df_metrics)
    st.info(f"IoU Final: {real_metrics['Mean IoU']}")
else:
    st.warning("⚠️ No se encontró el reporte de métricas. Ejecute `python train_pytorch.py` primero.")

st.write("---")

## 2. Predicción en Tiempo Real
st.header("🚀 Predicción en Tiempo Real via Agente API")

uploaded_file = st.file_uploader("Sube una imagen MRI (tif/png) para analizar", type=["tif", "png"])

if uploaded_file is not None:
    # 1. Preprocesamiento local de la imagen original para visualización
    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) # Cargamos a color para la superposición
    original_img = cv2.resize(original_img, (IMG_SIZE, IMG_SIZE))

    with st.spinner('Enviando imagen al Agente API para inferencia en GPU...'):
        mask_pred, prob_tumor, clase_detectada = get_prediction_from_api(uploaded_file)
    
    if mask_pred is not None:
        # --- Lógica para mostrar la Confianza Consistente ---
        prob_tumor_raw = prob_tumor # La confianza es el valor entre 0 y 1
        
        if clase_detectada == "Tumor Detectado":
            confianza_display = prob_tumor_raw * 100
        else:
            confianza_display = (1 - prob_tumor_raw) * 100
        
        st.subheader("Resultados del Análisis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(original_img, caption="Imagen de Entrada", use_container_width=True) 
            st.metric(label="Clasificación Final", value=clase_detectada, delta=f"Confianza: {confianza_display:.2f}%")
            
        with col2:
            # Creación de la máscara roja (BGR)
            color_mask = np.zeros_like(original_img, dtype=np.uint8)
            color_mask[mask_pred > 0] = [255, 0, 0] # Rojo en BGR/RGB invertido
            
            combined_img = cv2.addWeighted(original_img, 0.7, color_mask, 0.3, 0)
            
            st.image(combined_img, caption="Máscara de Segmentación Superpuesta (Roja)", use_container_width=True)
            
        st.markdown("### Interpretación del Agente")
        if clase_detectada == "Tumor Detectado":
            st.error(f"El Agente predice una **Anomalía** con alta convicción. La zona roja (Segmentación) es la prueba.")
        else:
            st.success(f"El Agente no detectó ninguna anomalía (Confianza de Ausencia: {confianza_display:.2f}%).")