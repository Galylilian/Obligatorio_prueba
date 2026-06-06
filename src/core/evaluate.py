import torch

from src.core.model import get_model
from src.preprocessing.dataset import get_dataloaders
from src.preprocessing.eda_stats import count_processed_classes
from src.settings.config import DEVICE, MODEL_PATH
from src.utils.metrics import compute_metrics


def load_model():
    model = get_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def evaluate(model, loader):
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    return compute_metrics(y_true, y_pred)


def _split_size(split: str) -> int:
    counts = count_processed_classes()
    return sum(counts.get(split, {}).values())


def main():
    _, valid_loader, test_loader, class_names = get_dataloaders()
    model = load_model()

    valid_n = _split_size("valid")
    test_n = _split_size("test")
    primary_split = "valid" if test_n < valid_n else "test"

    print(f"Clases: {class_names}")
    print(f"Tamaños: valid={valid_n}, test={test_n} → evaluación principal en '{primary_split}'")

    print("\n=== Validación ===")
    valid_metrics = evaluate(model, valid_loader)
    print(valid_metrics["report"])
    print(valid_metrics)

    print("\n=== Test ===")
    test_metrics = evaluate(model, test_loader)
    print(test_metrics["report"])
    print(test_metrics)

    if primary_split == "valid":
        print("\nNota: test es pequeño; usar métricas de valid como referencia principal.")


if __name__ == "__main__":
    main()
