# Modelo y entrenamiento

[[← Inicio|Home]] · [[Pipeline offline]]

---

## Modelo

- **Arquitectura:** ResNet18 preentrenada en ImageNet
- **Salida:** 2 clases (`fall`, `not_fall`)
- **Artefactos:**
  - `models/resnet18_best.pth` — pesos
  - `models/label_encoder.pkl` — mapeo índice → etiqueta

---

## Entrenamiento en 2 fases

| Fase | Qué entrena | Épocas | Learning rate |
|------|-------------|--------|---------------|
| 1 — Head | Solo capa `fc` | 3 | 0.001 |
| 2 — Fine-tune | `layer4` + `fc` | hasta 12 | 0.0001 |

Early stopping: 4 épocas sin mejora en `val_acc`.

Código: `src/core/train.py` · Config: `src/settings/config.py`

---

## Augmentación (train)

- `RandomResizedCrop(224)`
- Flip horizontal, rotación ±10°
- `ColorJitter` (brillo/contraste)

Inferencia usa resize 224×224 + normalización ImageNet (sin augmentación).

---

## Balance de clases

- `WeightedRandomSampler` en train
- Pesos en `CrossEntropyLoss` vía `get_binary_class_weights()`

---

## Métricas de referencia

| Split | Accuracy aprox. |
|-------|-------------------|
| Validación | ~92% |
| Test | ~87% |

> Las métricas exactas dependen del hardware y semilla. Ejecutar `evaluate` tras entrenar.

---

## Parámetros configurables (`.env`)

```
NUM_EPOCHS_HEAD=3
NUM_EPOCHS_FINETUNE=12
BATCH_SIZE=32
LEARNING_RATE=0.001
FINETUNE_LEARNING_RATE=0.0001
EARLY_STOPPING_PATIENCE=4
```

---

## Limitaciones conocidas

- Imágenes **fuera del dominio** (fotos de stock, fondos distintos) pueden dar falsos positivos/negativos.
- Grad-CAM es aproximado; no siempre marca solo al cuerpo.
- El modelo aprendió de datasets Roboflow indoor; no generaliza a cualquier foto de internet.

Para demo confiable: usar imágenes de `data/fused/test/`.
