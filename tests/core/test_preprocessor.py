from PIL import Image
from src.core.preprocessing.transforms import get_train_transforms, get_test_transforms
import torch


def test_train_transforms_output_shape():
    """Las transforms de train deben devolver un tensor de 3x224x224."""
    transform = get_train_transforms()
    img = Image.new("RGB", (300, 400))
    tensor = transform(img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)


def test_test_transforms_output_shape():
    """Las transforms de test deben devolver un tensor de 3x224x224."""
    transform = get_test_transforms()
    img = Image.new("RGB", (640, 480))
    tensor = transform(img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)


def test_transforms_are_different():
    """Train y test transforms deben ser distintas (data augmentation)."""
    train = get_train_transforms()
    test = get_test_transforms()

    # el número de transforms debe diferir (train tiene augmentation)
    assert len(train.transforms) != len(test.transforms)