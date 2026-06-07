import warnings
warnings.filterwarnings("ignore")
import os
import pathlib
import torch

from src.core.model import get_model
from src.data.dataset import get_dataloaders

# =============================
# HIPERPARÁMETROS ✅
# =============================
EPOCHS = 4  # para pruebas rápidas, aumentar para mejor rendimiento
LEARNING_RATE = 0.0005

# =============================
# DATOS
# =============================
train_loader, test_loader = get_dataloaders()

# =============================
# DISPOSITIVO
# =============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Usando dispositivo: {device}")

# =============================
# MODELO
# =============================
model = get_model().to(device)

# =============================
# ENTRENAMIENTO
# =============================
criterion = torch.nn.CrossEntropyLoss()

# ✅ FINE-TUNING (entrena TODAS las capas)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ✅ SCHEDULER (reduce learning rate progresivamente)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,   # cada 3 epochs
    gamma=0.5      # baja a la mitad
)

# =============================
# LOOP DE ENTRENAMIENTO
# =============================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

# =============================
# GUARDAR MODELO ✅
# =============================
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
models_dir = BASE_DIR / "models"
models_dir.mkdir(parents=True, exist_ok=True)

model_path = models_dir / "resnet18.pth"

torch.save(model.state_dict(), model_path)
print(f"✅ Modelo guardado en: {model_path}")

# =============================
# ✅ QUANTIZATION (OPTIMIZACIÓN)
# =============================
print("⚙️ Aplicando quantization...")

model.cpu()  # necesario para quantization

quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

quantized_path = models_dir / "resnet18_quantized.pth"

torch.save(quantized_model.state_dict(), quantized_path)

print(f"✅ Modelo cuantizado guardado en: {quantized_path}")
