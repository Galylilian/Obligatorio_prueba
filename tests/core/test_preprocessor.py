from PIL import Image

from src.preprocessing.transforms import get_eval_transforms, get_train_transforms


def test_train_transforms_output_shape():
    transform = get_train_transforms()
    img = Image.new("RGB", (480, 640))
    tensor = transform(img)
    assert tensor.shape == (3, 224, 224)


def test_eval_transforms_output_shape():
    transform = get_eval_transforms()
    img = Image.new("RGB", (480, 640))
    tensor = transform(img)
    assert tensor.shape == (3, 224, 224)
