# Modelo y entrenamiento

[[← Inicio|Home]] · [[Pipeline offline]]

Acá explicamos *qué* aprende el modelo y *cómo* lo entrenamos, sin entrar en cada línea de código.

---

## El modelo en criollo

Usamos **ResNet18**: una red que ya "sabe ver" (entrenada en ImageNet) y la adaptamos a nuestro problema de 2 clases:

- `fall` — caída
- `not_fall` — todo lo demás

Al terminar tenés dos archivos importantes:
- `models/resnet18_best.pth` — el cerebro
- `models/label_encoder.pkl` — traduce números a etiquetas

---

## Entrenamiento en 2 fases (por qué)

No entrenamos toda la red de golpe. Hacemos esto:

| Fase | Qué pasa | Analogía |
|------|----------|----------|
| **1** | Solo la última capa aprende (3 épocas) | "Decime fall o not_fall con lo que ya sabés ver" |
| **2** | Afinamos `layer4` + capa final (hasta 12 épocas) | "Ahora sí, especializate en caídas" |

Si en validación no mejora 4 épocas seguidas, paramos solos (early stopping).

---

## Datos de entrenamiento

- Imágenes a **224×224**
- En train: rotamos, volteamos, cambiamos brillo (augmentación)
- En valid/test: solo resize, sin trucos

Para que no gane siempre la clase mayoritaria usamos **balanceo** en el DataLoader y pesos en la loss.

---

## ¿Qué resultados esperar?

Después del fine-tuning, referencia orientativa:

| Conjunto | Accuracy |
|----------|----------|
| Validación | ~92% |
| Test | ~87% |

Corré `evaluate` en tu máquina para ver tus números exactos.

---

## Ajustar parámetros

En `.env` podés tocar:

```
NUM_EPOCHS_HEAD=3
NUM_EPOCHS_FINETUNE=12
BATCH_SIZE=32
LEARNING_RATE=0.001
FINETUNE_LEARNING_RATE=0.0001
```

---

## Seamos honestos: limitaciones

1. **Fotos random de Google** pueden fallar — el modelo aprendió de cámaras indoor de Roboflow, no de fotos de stock con fondo gris.
2. **Grad-CAM** es una pista, no la verdad — a veces marca el sillón en vez de la persona.
3. **Confianza baja** (< 70%) = el modelo duda; no confíes ciegamente.

Para la demo del obligatorio, usá imágenes de `data/fused/test/` — ahí funciona mucho mejor.
