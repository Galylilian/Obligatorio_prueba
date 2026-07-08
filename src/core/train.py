import warnings
warnings.filterwarnings("ignore")
import pathlib
import torch

from src.core.model import get_model
from src.data.dataset import get_dataloaders
from src.utils.logger import get_logger

logger = get_logger("train")

# =============================
# HIPERPARÁMETROS
# =============================
EPOCHS = 4          # suficiente para fine-tuning con dataset chico
LEARNING_RATE = 0.0005

# =============================
# DISPOSITIVO
# =============================
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Usando dispositivo: {device}")

# =============================
# DATOS
# =============================
train_loader, valid_loader, test_loader = get_dataloaders()

# =============================
# MODELO
# =============================
# pretrained=True → carga pesos de ImageNet para fine-tuning real
# Esto es clave cuando el dataset es chico (mejor punto de partida)
model = get_model(pretrained=True).to(device)

logger.info("Modelo ResNet18 cargado con pesos ImageNet (fine-tuning)")

# =============================
# ENTRENAMIENTO
# =============================

# Class weights inversamente proporcionales a la frecuencia de cada clase.
# Fórmula: n_samples / (n_classes * count_per_class)
# Esto hace que el loss de la clase minoritaria pese más durante el backprop.
_targets = torch.tensor(train_loader.dataset.targets)
_counts = torch.bincount(_targets)
_class_weights = (_targets.size(0) / (len(_counts) * _counts.float())).to(device)

classes = train_loader.dataset.classes
for cls, w in zip(classes, _class_weights):
    logger.info(f"Class weight '{cls}': {w:.4f}")

criterion = torch.nn.CrossEntropyLoss(weight=_class_weights)

# Adam con todas las capas — fine-tuning completo
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Scheduler: reduce el learning rate cada 3 epochs a la mitad
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.5
)

# =============================
# PATHS
# =============================
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
models_dir = BASE_DIR / "models"
models_dir.mkdir(parents=True, exist_ok=True)

model_path = models_dir / "resnet18.pth"

# =============================
# LOOP DE ENTRENAMIENTO
# =============================
best_val_accuracy = 0.0

for epoch in range(EPOCHS):

    # ── TRAIN ──────────────────────────────
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

    # ── VALIDACIÓN ─────────────────────────
    # Evaluamos en valid_loader al final de cada epoch
    # para saber si el modelo está mejorando y guardar el mejor.
    # test_loader se reserva para una única evaluación final
    # y no participa en la selección del modelo (evita leakage).
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_accuracy = correct / total if total > 0 else 0.0

    logger.info(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {total_loss:.4f} | "
        f"Val Accuracy: {val_accuracy*100:.2f}%"
    )

    # ── GUARDAR MEJOR MODELO ───────────────
    # Solo guardamos el modelo si mejoró respecto al epoch anterior
    # Así evitamos guardar un modelo que empeoró al final
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), model_path)
        logger.info(f"✅ Mejor modelo guardado (accuracy: {val_accuracy*100:.2f}%)")

logger.info(f"Entrenamiento finalizado. Mejor accuracy (valid): {best_val_accuracy*100:.2f}%")
logger.info(f"Modelo guardado en: {model_path}")

# =============================
# EVALUACIÓN FINAL (TEST)
# =============================
# Única vez que se usa test_loader: no participó en la selección del
# modelo durante el entrenamiento, por lo que da una estimación no sesgada.
best_model = get_model(pretrained=False)
best_model.load_state_dict(torch.load(model_path, map_location="cpu"))
best_model.cpu()
best_model.eval()

test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = best_model(images)
        preds = outputs.argmax(dim=1)
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

test_accuracy = test_correct / test_total if test_total > 0 else 0.0
logger.info(f"Test Accuracy (evaluación final): {test_accuracy*100:.2f}%")

# =============================
# QUANTIZATION (OPTIMIZACIÓN)
# =============================
# Se aplica sobre el mejor modelo guardado para no perder el best checkpoint
logger.info("Aplicando quantization al mejor modelo...")

quantized_model = torch.quantization.quantize_dynamic(
    best_model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

quantized_path = models_dir / "resnet18_quantized.pth"
torch.save(quantized_model.state_dict(), quantized_path)

logger.info(f"✅ Modelo cuantizado guardado en: {quantized_path}")