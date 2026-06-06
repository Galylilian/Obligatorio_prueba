import os

import torch

from src.core.model import (
    freeze_backbone,
    get_model,
    trainable_parameter_groups,
    unfreeze_from_layer4,
)
from src.preprocessing.dataset import get_dataloaders
from src.preprocessing.eda_stats import get_binary_class_weights
from src.settings.config import settings
from src.utils.label_encoder import save_label_encoder


def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / max(len(loader), 1), correct / max(total, 1)


def _run_phase(
    *,
    model,
    train_loader,
    valid_loader,
    criterion,
    device,
    optimizer,
    num_epochs: int,
    phase_name: str,
    model_file: str,
    encoder_file: str,
    class_names: list[str],
    best_val_acc: float,
    patience: int,
) -> tuple[float, int]:
    epochs_without_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss, val_acc = evaluate_epoch(model, valid_loader, criterion, device)
        print(
            f"[{phase_name}] Epoch {epoch + 1}/{num_epochs} | "
            f"train_loss={train_loss / len(train_loader):.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improve = 0
            torch.save(model.state_dict(), model_file)
            save_label_encoder(class_names, encoder_file)
            print(f"  -> Mejor modelo guardado (val_acc={val_acc:.4f})")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"  -> Early stopping ({patience} epocas sin mejora)")
                break

    return best_val_acc, epochs_without_improve


def main():
    train_loader, valid_loader, _, class_names = get_dataloaders()
    device = settings.device
    print(f"Dispositivo: {device}")
    print(f"Clases: {class_names}")

    model = get_model().to(device)
    freeze_backbone(model)
    class_weights = torch.tensor(get_binary_class_weights(), dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    models_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
    os.makedirs(models_dir, exist_ok=True)
    model_file = os.path.join(models_dir, "resnet18_best.pth")
    encoder_file = os.path.join(models_dir, "label_encoder.pkl")

    best_val_acc = 0.0
    patience = settings.early_stopping_patience

    print("\n=== Fase 1: entrenar solo capa final (fc) ===")
    head_optimizer = torch.optim.Adam(model.fc.parameters(), lr=settings.learning_rate)
    best_val_acc, _ = _run_phase(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        device=device,
        optimizer=head_optimizer,
        num_epochs=settings.num_epochs_head,
        phase_name="head",
        model_file=model_file,
        encoder_file=encoder_file,
        class_names=class_names,
        best_val_acc=best_val_acc,
        patience=patience,
    )

    print("\n=== Fase 2: fine-tuning layer4 + fc ===")
    unfreeze_from_layer4(model)
    finetune_params = trainable_parameter_groups(model)
    print(f"Parametros entrenables: {sum(p.numel() for p in finetune_params):,}")
    finetune_optimizer = torch.optim.Adam(finetune_params, lr=settings.finetune_learning_rate)
    best_val_acc, _ = _run_phase(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        device=device,
        optimizer=finetune_optimizer,
        num_epochs=settings.num_epochs_finetune,
        phase_name="finetune",
        model_file=model_file,
        encoder_file=encoder_file,
        class_names=class_names,
        best_val_acc=best_val_acc,
        patience=patience,
    )

    print(f"\nMejor val_acc: {best_val_acc:.4f}")
    print(f"Modelo guardado en: {model_file}")
    print(f"Label encoder guardado en: {encoder_file}")


if __name__ == "__main__":
    main()
