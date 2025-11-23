from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel # Para el esquema de respuesta
import torch
import numpy as np
import cv2
import base64
from io import BytesIO
import os

# Importar la arquitectura del modelo
from pytorch_model import UNetMultiTask, IMG_SIZE, DEVICE 

# --- Configuración ---
app = FastAPI(title="Agente de Predicción de Tumores MRI")
MODEL_PATH = 'unet_multi_task_pytorch.pth'
model = None 

# Función para codificar la máscara (NumPy array) a una cadena Base64
def encode_mask_to_base64(mask_array):
    # Usamos cv2.imencode para guardar el array como PNG en memoria
    _, buffer = cv2.imencode('.png', mask_array)
    return base64.b64encode(buffer).decode('utf-8')

@app.on_event("startup")
def load_model_and_device():
    """Carga el modelo en la GPU al iniciar el servidor (RTX 3050)."""
    global model
    print(f"Iniciando Agente de IA en {DEVICE}...")
    model = UNetMultiTask()
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Archivo de modelo '{MODEL_PATH}' no encontrado. Ejecute train_pytorch.py primero.")
        # Se lanza una excepción para evitar que el servidor funcione sin el modelo
        raise RuntimeError("Modelo no encontrado.")
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("✅ Agente API listo. Modelo cargado en la GPU.")

@app.post("/predict_mri")
async def predict_mri(file: UploadFile = File(...)):
    """
    ENDPOINT PRINCIPAL DEL AGENTE: Recibe la imagen, ejecuta la inferencia
    acelerada por GPU y devuelve la predicción.
    """
    
    # 1. Lectura y Preprocesamiento
    content = await file.read()
    file_bytes = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen.")

    original_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Preparar tensor para PyTorch: (1, 1, H, W)
    img_tensor = torch.from_numpy(original_img).float().unsqueeze(0).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(DEVICE)
    
    # 2. Inferencia en la GPU
    with torch.no_grad():
        seg_pred, clf_logits = model(img_tensor)
        
    # 3. Post-procesamiento y Decisión
    prob_tumor = torch.sigmoid(clf_logits).item() 
    mask_array = (seg_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    
    clase_detectada = "Tumor Detectado" if prob_tumor > 0.5 else "Sin Anomalía"
    
    # 4. Preparar Respuesta JSON
    return {
        "prediction": clase_detectada,
        "confidence": round(prob_tumor, 4),
        "mask_png_base64": encode_mask_to_base64(mask_array)
    }

# Ruta para la documentación (http://127.0.0.1:8000/docs)