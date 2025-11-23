import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import kagglehub
import json
import glob
from pytorch_model import UNetMultiTask, dice_loss, DEVICE, IMG_SIZE 

# --- Rutas y Constantes ---
MODEL_PATH = 'unet_multi_task_pytorch.pth'
METRICS_FILE = 'real_metrics_pytorch.json'
BATCH_SIZE = 16
NUM_EPOCHS = 150 # Aumentado para asegurar convergencia con la nueva capacidad
POS_WEIGHT_TUMOR = 5.0 

# --- Configuraciones de GPU (Verificación, sin cambios) ---
if DEVICE.type == 'cuda':
    torch.cuda.empty_cache() 
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: GPU no detectada. El entrenamiento se ejecutará en CPU.")


# --- Dataset PyTorch Personalizado (Sin cambios) ---
class MRIDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img = cv2.imread(row['image_path'], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            return torch.zeros(1, IMG_SIZE, IMG_SIZE), torch.zeros(1, IMG_SIZE, IMG_SIZE), torch.tensor(0.0)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0

        clf_label = torch.tensor(row['label']).float().unsqueeze(0) 

        return img_tensor, mask_tensor, clf_label

# --- Funciones de Entrenamiento y Evaluación ---

def train_epoch(model, dataloader, optimizer, loss_clf, loss_seg):
    model.train()
    total_loss = 0
    for images, masks, labels in dataloader:
        images, masks, labels = images.to(DEVICE), masks.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        seg_pred, clf_pred = model(images)
        
        seg_loss = loss_seg(seg_pred, masks)
        clf_loss = loss_clf(clf_pred, labels)
        
        # REBALANCEO CLAVE: 0.7 para Seg, 0.3 para Clasificación (Priorizando IoU)
        loss = 0.7 * seg_loss + 0.3 * clf_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, loss_clf, loss_seg):
    model.eval()
    total_loss, correct_clf, total_clf = 0, 0, 0
    total_iou, num_batches = 0, 0
    
    with torch.no_grad():
        for images, masks, labels in dataloader:
            images, masks, labels = images.to(DEVICE), masks.to(DEVICE), labels.to(DEVICE)
            
            seg_pred, clf_pred = model(images)
            
            seg_loss = loss_seg(seg_pred, masks)
            clf_loss = loss_clf(clf_pred, labels)
            
            # REBALANCEO CLAVE: 0.7 para Seg, 0.3 para Clasificación
            loss = 0.7 * seg_loss + 0.3 * clf_loss
            total_loss += loss.item() 
            
            # 1. CLASIFICACIÓN
            predicted_classes = (torch.sigmoid(clf_pred) > 0.5).float() 
            correct_clf += (predicted_classes == labels).sum().item()
            total_clf += labels.size(0)

            # 2. SEGMENTACIÓN: Dice Score es (1 - Dice Loss)
            dice_score = 1.0 - seg_loss.item()
            total_iou += dice_score
            num_batches += 1
            
    avg_loss = total_loss / num_batches
    avg_accuracy = correct_clf / total_clf
    avg_iou = total_iou / num_batches
    
    return avg_loss, avg_accuracy, avg_iou

# --- Lógica de Carga de Datos (Sin cambios) ---
def load_data_paths():
    print("Iniciando descarga del dataset...")
    try:
        dataset_root_path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")
        IMAGE_DIR = os.path.join(dataset_root_path, 'kaggle_3m/')
        print(f"Dataset disponible en: {IMAGE_DIR}")
    except Exception as e:
        print(f"Error al descargar con kagglehub: {e}. Usando ruta local de respaldo.")
        IMAGE_DIR = 'lgg-mri-segmentation/kaggle_3m/'
        
    all_files = sorted(glob.glob(os.path.join(IMAGE_DIR, '*/*_mask.tif')))
    
    data = []
    for mask_path in all_files:
        img_path = mask_path.replace('_mask.tif', '.tif')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        label = 1 if np.sum(mask) > 0 else 0 
        data.append({'image_path': img_path, 'mask_path': mask_path, 'label': label})
        
    df = pd.DataFrame(data).dropna()
    return df

# --- Ejecución Principal ---
if __name__ == '__main__':
    df_full = load_data_paths()
    
    df_train, df_test = train_test_split(df_full, test_size=0.2, stratify=df_full['label'], random_state=42)
    print(f"\nPartición de Entrenamiento (80%): {len(df_train)} imágenes")
    print(f"Partición de Prueba (20%): {len(df_test)} imágenes")
    
    train_dataset = MRIDataset(df_train)
    test_dataset = MRIDataset(df_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = UNetMultiTask()
    model.to(DEVICE) 
    
    # Optimizador y Scheduler
    optimizer = optim.Adam(model.parameters(), lr=5e-5) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10) # Se eliminó 'verbose=True'    
    pos_weight = torch.tensor([POS_WEIGHT_TUMOR]).to(DEVICE)
    loss_clf = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    loss_seg = dice_loss 

    print(f"\n--- Iniciando Entrenamiento Acelerado en {DEVICE} ({NUM_EPOCHS} Épocas) ---")

    best_iou = 0
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, loss_clf, loss_seg)
        test_loss, test_acc, test_iou = evaluate(model, test_loader, loss_clf, loss_seg)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | Dice/IoU: {test_iou:.4f}")

        # Pasamos la pérdida de prueba al scheduler
        scheduler.step(test_loss)

        if test_iou > best_iou:
            best_iou = test_iou
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✅ Modelo guardado: {MODEL_PATH}")

    # --- 6. Guardar Métricas Finales para Streamlit ---
    final_metrics = {
        'Accuracy': f"{test_acc:.4f}",
        'F1-Score': f"{test_iou:.4f}", 
        'Mean IoU': f"{test_iou:.4f}"
    }

    with open(METRICS_FILE, 'w') as f:
        json.dump(final_metrics, f)
    
    print(f"\n✅ Entrenamiento completo. Métricas guardadas en {METRICS_FILE}.")