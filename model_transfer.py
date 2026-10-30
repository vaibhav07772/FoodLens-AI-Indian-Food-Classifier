"""
model_transfer.py
------------------
Reproduces the architecture of the trained model in
model/food_classifier.h5: a MobileNetV2 backbone (pretrained on
ImageNet, frozen) with a small classification head on top. This is a
stronger alternative to the from-scratch CNN in model.py — it's what
the synopsis's Literature Review points to when it mentions MobileNet
as a lightweight, mobile-friendly deep learning backbone.

Use this if you want to retrain or fine-tune on a different/updated
food dataset while keeping the same architecture as the shipped model.
"""

from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0

IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3

# "mobilenetv2" = fast, lighter, good baseline
# "efficientnetb0" = usually 3-5% higher top-1 accuracy than MobileNetV2
#                 at a similar parameter budget — the better choice
#                 when "best accuracy" matters more than raw speed
BACKBONES = {
    "mobilenetv2": MobileNetV2,
    "efficientnetb0": EfficientNetB0,
}


def build_model(num_classes: int, backbone: str = "efficientnetb0", fine_tune_base: bool = False) -> models.Model:
    """Builds a transfer-learning food classifier.

    Args:
        num_classes: number of food categories (25 for the bundled
            Indian Food dataset — see dataset/README.md).
        backbone: "efficientnetb0" (recommended, higher accuracy) or
            "mobilenetv2" (faster, smaller, matches the originally
            shipped model).
        fine_tune_base: if True, unfreezes the backbone for a second,
            low-learning-rate fine-tuning pass after the head has
            already been trained. This second phase is usually where
            most of the accuracy gain over a frozen-backbone model
            comes from.
    """
    base_cls = BACKBONES[backbone]
    base = base_cls(
        input_shape=(*IMAGE_SIZE, IMAGE_CHANNELS),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = fine_tune_base

    inputs = layers.Input(shape=(*IMAGE_SIZE, IMAGE_CHANNELS))
    x = base(inputs, training=fine_tune_base)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name=f"food_{backbone}")

    learning_rate = 1e-5 if fine_tune_base else 1e-3
    model.compile(
        optimizer=optimizers.Adam(learning_rate),
        # label_smoothing helps generalization / reduces overconfidence
        # on visually-similar classes (e.g. poori vs chappati)
        loss=losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    return model


def get_preprocessing_function(backbone: str = "efficientnetb0"):
    """Returns the preprocessing function matching the chosen backbone
    (pass as `preprocessing_function=` to ImageDataGenerator, and use
    the same one at inference time in app.py)."""
    if backbone == "efficientnetb0":
        from tensorflow.keras.applications.efficientnet import preprocess_input
    else:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    return preprocess_input


if __name__ == "__main__":
    demo_model = build_model(num_classes=25, backbone="efficientnetb0")
    demo_model.summary()
