import torch

from src.core.model import get_model
from src.data.dataset import get_dataloaders
from src.utils.metrics import compute_metrics
from src.settings.config import MODEL_PATH, DEVICE

# =============================
# DATA
# =============================
_, test_loader = get_dataloaders()

# =============================
# MODELO
# =============================
model = get_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ✅ IMPORTANTE (esto faltaba)
y_true = []
y_pred = []

# =============================
# EVALUACIÓN
# =============================
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()

        y_pred.extend(preds)
        y_true.extend(labels.numpy())

# =============================
# MÉTRICAS
# =============================
metrics = compute_metrics(y_true, y_pred)

print("\n📊 RESULTADOS:")
for k, v in metrics.items():
    print(f"{k}: {v}")
