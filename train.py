"""
train.py
--------
Trains the Indian food classification model (or any ImageFolder-style
dataset) using a 2-phase transfer-learning strategy, which is what
actually pushes accuracy up compared to a single frozen-backbone
training run:

    Phase 1 (head warm-up):  backbone frozen, only the new
        classification head trains. Fast, stabilizes the head so
        Phase 2 doesn't destroy the pretrained weights with large
        gradients.

    Phase 2 (fine-tuning):   backbone unfrozen, whole network trains
        at a much lower learning rate. This is where most of the
        accuracy gain over Phase 1 alone comes from.

Expected dataset layout (ImageFolder style), matching the ~70/15/15
train/val/test split described in dataset/README.md:

    dataset/
        train/
            biryani/  img1.jpg  img2.jpg  ...
            samosa/   ...
        val/
            biryani/ ...
            samosa/  ...
        test/
            biryani/ ...
            samosa/  ...

Usage:
    python train.py --dataset dataset --backbone efficientnetb0 \
        --head-epochs 8 --finetune-epochs 20 --batch-size 32
"""

import argparse
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from model_transfer import build_model, get_preprocessing_function, IMAGE_SIZE


def get_generators(dataset_dir: str, batch_size: int, backbone: str):
    """Image Pre-processing + Data Augmentation modules: resize to
    224x224, augment the training set (rotation/flip/zoom/brightness/
    shift), and normalize both train and val with the preprocessing
    function that matches the chosen backbone."""

    preprocess = get_preprocessing_function(backbone)

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        brightness_range=(0.8, 1.2),
        horizontal_flip=True,
        fill_mode="nearest",
    )
    eval_datagen = ImageDataGenerator(preprocessing_function=preprocess)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(dataset_dir, "train"),
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
    )
    val_gen = eval_datagen.flow_from_directory(
        os.path.join(dataset_dir, "val"),
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return train_gen, val_gen


def get_test_generator(dataset_dir: str, batch_size: int, backbone: str):
    preprocess = get_preprocessing_function(backbone)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess)
    return test_datagen.flow_from_directory(
        os.path.join(dataset_dir, "test"),
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Train the food classification model (2-phase transfer learning).")
    parser.add_argument("--dataset", default="dataset", help="Path to dataset root folder")
    parser.add_argument("--backbone", default="efficientnetb0", choices=["efficientnetb0", "mobilenetv2"],
                         help="efficientnetb0 = best accuracy (default), mobilenetv2 = faster/lighter")
    parser.add_argument("--head-epochs", type=int, default=8, help="Phase 1: frozen-backbone epochs")
    parser.add_argument("--finetune-epochs", type=int, default=20, help="Phase 2: fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="model/food_classifier.h5", help="Where to save the trained model")
    args = parser.parse_args()

    train_gen, val_gen = get_generators(args.dataset, args.batch_size, args.backbone)
    class_indices = train_gen.class_indices  # {"pizza": 0, "samosa": 1, ...}
    idx_to_class = {v: k for k, v in class_indices.items()}
    num_classes = len(class_indices)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # One label per line, in index order — regenerated fresh from the
    # dataset's actual folder names every run, so it can never drift
    # out of sync with the model's output indices.
    with open("model/labels.txt", "w") as f:
        for i in range(num_classes):
            f.write(idx_to_class[i] + "\n")
    print(f"Found {num_classes} classes. Labels saved to model/labels.txt")

    # ---------- Phase 1: train the head with a frozen backbone ----------
    print("\n=== Phase 1: training classification head (backbone frozen) ===")
    model = build_model(num_classes=num_classes, backbone=args.backbone, fine_tune_base=False)
    model.summary()

    phase1_callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ModelCheckpoint(args.output, monitor="val_accuracy", save_best_only=True),
    ]
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.head_epochs,
        callbacks=phase1_callbacks,
    )

    # ---------- Phase 2: unfreeze and fine-tune the whole network ----------
    print("\n=== Phase 2: fine-tuning full network (low learning rate) ===")
    fine_tune_model = build_model(num_classes=num_classes, backbone=args.backbone, fine_tune_base=True)
    fine_tune_model.set_weights(model.get_weights())  # carry Phase 1 weights over

    phase2_callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ModelCheckpoint(args.output, monitor="val_accuracy", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ]
    fine_tune_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.finetune_epochs,
        callbacks=phase2_callbacks,
    )

    fine_tune_model.save(args.output)
    print(f"\nTraining complete. Best model saved to {args.output}")

    # ---------- Final check on the held-out test set ----------
    test_dir = os.path.join(args.dataset, "test")
    if os.path.isdir(test_dir):
        print("\n=== Evaluating on test set ===")
        test_gen = get_test_generator(args.dataset, args.batch_size, args.backbone)
        loss, acc = fine_tune_model.evaluate(test_gen)
        print(f"Test accuracy: {acc * 100:.2f}%  |  Test loss: {loss:.4f}")


if __name__ == "__main__":
    main()
