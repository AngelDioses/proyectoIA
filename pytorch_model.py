import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Configuraciones Globales ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256

# --- Métricas y Pérdida ---
def dice_loss(pred, target, smooth=1e-6):
    """Calcula la Dice Loss (1 - Dice Score) para la segmentación."""
    pred = pred.view(-1)
    target = target.view(-1).float()
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1.0 - dice

# --- Bloque de la Arquitectura U-Net ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetMultiTask(nn.Module):
    def __init__(self, in_channels=1, n_classes_seg=1):
        super().__init__()
        
        # CODIFICADOR (Arquitectura Escalada: 32, 64, 128)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down1 = DoubleConv(in_channels, 32)  
        self.down2 = DoubleConv(32, 64)           
        self.bridge = DoubleConv(64, 128)         
        
        # CABEZA DE CLASIFICACIÓN (Salida en Logits)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.clf_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1), 
            # Sigmoid eliminada para usar BCEWithLogitsLoss
        )
        
        # DECODIFICADOR
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) 
        self.up2 = DoubleConv(64 + 64, 64) 
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2) 
        self.up1 = DoubleConv(32 + 32, 32)
        
        # CABEZA DE SEGMENTACIÓN
        self.seg_head = nn.Conv2d(32, n_classes_seg, kernel_size=1)
        
    def forward(self, x):
        # Codificador
        x1 = self.down1(x); x2 = self.maxpool(x1)     
        x3 = self.down2(x2); x4 = self.maxpool(x3)     
        x_bridge = self.bridge(x4) 
        
        # 1. SALIDA DE CLASIFICACIÓN (Logits)
        x_clf = self.avgpool(x_bridge) 
        x_clf = x_clf.view(x_clf.size(0), -1) 
        clf_output = self.clf_head(x_clf)
        
        # Decodificador
        x = self.upconv2(x_bridge); x = torch.cat([x, x3], dim=1); x = self.up2(x)
        x = self.upconv1(x); x = torch.cat([x, x1], dim=1); x = self.up1(x)
        
        # 2. SALIDA DE SEGMENTACIÓN (Probabilidad)
        seg_output = torch.sigmoid(self.seg_head(x))
        
        return seg_output, clf_output