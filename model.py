"""
model.py
--------
Defines the Convolutional Neural Network (CNN) architecture used by the
AI-Based Food Classification system, exactly following the pipeline
described in the project synopsis:

    Input (224x224x3)
      -> Convolution + ReLU  (feature extraction)
      -> Max Pooling         (dimensionality reduction)
      -> Convolution + ReLU
      -> Max Pooling
      -> Convolution + ReLU
      -> Max Pooling
      -> Flatten
      -> Fully Connected (Dense)
      -> Dropout              (regularization / generalization)
      -> Softmax               (class probabilities)
"""

from tensorflow.keras import layers, models

IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3


def build_model(num_classes: int) -> models.Sequential:
    """Builds and compiles the CNN described in the synopsis.

    Args:
        num_classes: number of food categories the model should predict
            (e.g. Pizza, Burger, Dosa, Idli, Biryani, Pasta, Sandwich,
            Ice Cream, Samosa, ...).

    Returns:
        A compiled tf.keras.Model ready for training.
    """
    model = models.Sequential(name="food_classification_cnn")

    # --- Block 1 ---
    model.add(layers.Input(shape=(*IMAGE_SIZE, IMAGE_CHANNELS)))
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # --- Block 2 ---
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # --- Block 3 ---
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # --- Block 4 ---
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # --- Classification head ---
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    # Quick sanity check: prints the architecture summary.
    demo_model = build_model(num_classes=10)
    demo_model.summary()
