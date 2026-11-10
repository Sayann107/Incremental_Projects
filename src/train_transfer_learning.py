from pathlib import Path
import csv
import tensorflow as tf

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    Input,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    GlobalAveragePooling2D,
    Dropout,
    Dense
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input
)
from tensorflow.keras.callbacks import EarlyStopping


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

RANDOM_STATE = 42

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 160

BATCH_SIZE = 32
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 5

DATA_DIRECTORY = Path(
    "data/fashion_products"
)

MODEL_DIRECTORY = Path("models")

MODEL_PATH = (
    MODEL_DIRECTORY /
    "fashion_transfer_model.keras"
)

RESULTS_PATH = (
    MODEL_DIRECTORY /
    "day7_transfer_results.csv"
)

HISTORY_PATH = (
    MODEL_DIRECTORY /
    "day7_transfer_history.csv"
)

CLASS_NAMES_PATH = (
    MODEL_DIRECTORY /
    "day7_class_names.txt"
)

MODEL_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

tf.keras.utils.set_random_seed(
    RANDOM_STATE
)


# --------------------------------------------------
# 2. Check dataset
# --------------------------------------------------

if not DATA_DIRECTORY.exists():

    raise FileNotFoundError(
        "Dataset folder not found:\n"
        "data/fashion_products\n\n"
        "Download and arrange product images "
        "inside category subfolders first."
    )


category_folders = [
    folder
    for folder in DATA_DIRECTORY.iterdir()
    if folder.is_dir()
]

if len(category_folders) < 2:

    raise ValueError(
        "At least two category folders are required.\n"
        "Example:\n"
        "data/fashion_products/Apparel/\n"
        "data/fashion_products/Footwear/"
    )


# --------------------------------------------------
# 3. Load training dataset
# --------------------------------------------------

train_dataset = (
    tf.keras.utils.image_dataset_from_directory(
        DATA_DIRECTORY,
        validation_split=0.20,
        subset="training",
        seed=RANDOM_STATE,
        image_size=(
            IMAGE_HEIGHT,
            IMAGE_WIDTH
        ),
        batch_size=BATCH_SIZE,
        label_mode="int"
    )
)


# --------------------------------------------------
# 4. Load validation dataset
# --------------------------------------------------

validation_dataset = (
    tf.keras.utils.image_dataset_from_directory(
        DATA_DIRECTORY,
        validation_split=0.20,
        subset="validation",
        seed=RANDOM_STATE,
        image_size=(
            IMAGE_HEIGHT,
            IMAGE_WIDTH
        ),
        batch_size=BATCH_SIZE,
        label_mode="int"
    )
)


# --------------------------------------------------
# 5. Read class names
# --------------------------------------------------

class_names = train_dataset.class_names

number_of_classes = len(class_names)

print("\nClass names:")
print(class_names)

print(
    "\nNumber of classes:",
    number_of_classes
)


# --------------------------------------------------
# 6. Save class names
# --------------------------------------------------

with open(
    CLASS_NAMES_PATH,
    "w",
    encoding="utf-8"
) as file:

    for class_name in class_names:

        file.write(
            class_name + "\n"
        )


# --------------------------------------------------
# 7. Improve dataset performance
# --------------------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = (
    validation_dataset.prefetch(
        buffer_size=AUTOTUNE
    )
)


# --------------------------------------------------
# 8. Data augmentation
# --------------------------------------------------

data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.10),
    RandomZoom(0.10)
], name="data_augmentation")


# --------------------------------------------------
# 9. Load pretrained MobileNetV2
# --------------------------------------------------

base_model = MobileNetV2(
    input_shape=(
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        3
    ),
    include_top=False,
    weights="imagenet"
)


# --------------------------------------------------
# 10. Freeze pretrained layers
# --------------------------------------------------

base_model.trainable = False

print(
    "\nBase model frozen:",
    not base_model.trainable
)


# --------------------------------------------------
# 11. Build transfer-learning model
# --------------------------------------------------

inputs = Input(
    shape=(
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        3
    )
)

x = data_augmentation(inputs)

x = preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = GlobalAveragePooling2D()(x)

x = Dropout(0.30)(x)

outputs = Dense(
    number_of_classes,
    activation="softmax"
)(x)

model = Model(
    inputs,
    outputs,
    name="retailmind_transfer_classifier"
)


# --------------------------------------------------
# 12. Compile frozen model
# --------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nTransfer-learning architecture:")

model.summary()


# --------------------------------------------------
# 13. Early stopping
# --------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)


# --------------------------------------------------
# 14. Train new classification head
# --------------------------------------------------

print("\nTraining classification head...")

history_initial = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=INITIAL_EPOCHS,
    callbacks=[early_stopping],
    verbose=1
)


# --------------------------------------------------
# 15. Fine-tune upper layers
# --------------------------------------------------

base_model.trainable = True

fine_tune_from = len(
    base_model.layers
) - 20

for layer in base_model.layers[
    :fine_tune_from
]:

    layer.trainable = False


# --------------------------------------------------
# 16. Recompile with low learning rate
# --------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# --------------------------------------------------
# 17. Fine-tune model
# --------------------------------------------------

print("\nFine-tuning top layers...")

history_fine = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[early_stopping],
    verbose=1
)


# --------------------------------------------------
# 18. Evaluate final model
# --------------------------------------------------

validation_loss, validation_accuracy = (
    model.evaluate(
        validation_dataset,
        verbose=0
    )
)

print("\n" + "=" * 60)
print("DAY 7 TRANSFER-LEARNING RESULTS")
print("=" * 60)

print(
    f"Validation loss    : "
    f"{validation_loss:.4f}"
)

print(
    f"Validation accuracy: "
    f"{validation_accuracy:.4f}"
)


# --------------------------------------------------
# 19. Save final model
# --------------------------------------------------

model.save(
    MODEL_PATH
)


# --------------------------------------------------
# 20. Combine training histories
# --------------------------------------------------

initial_epochs_completed = len(
    history_initial.history["loss"]
)

fine_tune_epochs_completed = len(
    history_fine.history["loss"]
)

combined_history = []

for epoch in range(
    initial_epochs_completed
):

    combined_history.append({
        "stage": "frozen_base",
        "epoch": epoch + 1,
        "loss": (
            history_initial
            .history["loss"][epoch]
        ),
        "accuracy": (
            history_initial
            .history["accuracy"][epoch]
        ),
        "val_loss": (
            history_initial
            .history["val_loss"][epoch]
        ),
        "val_accuracy": (
            history_initial
            .history["val_accuracy"][epoch]
        )
    })


for epoch in range(
    fine_tune_epochs_completed
):

    combined_history.append({
        "stage": "fine_tuning",
        "epoch": epoch + 1,
        "loss": (
            history_fine
            .history["loss"][epoch]
        ),
        "accuracy": (
            history_fine
            .history["accuracy"][epoch]
        ),
        "val_loss": (
            history_fine
            .history["val_loss"][epoch]
        ),
        "val_accuracy": (
            history_fine
            .history["val_accuracy"][epoch]
        )
    })


# --------------------------------------------------
# 21. Save training history
# --------------------------------------------------

with open(
    HISTORY_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    field_names = [
        "stage",
        "epoch",
        "loss",
        "accuracy",
        "val_loss",
        "val_accuracy"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=field_names
    )

    writer.writeheader()

    writer.writerows(
        combined_history
    )


# --------------------------------------------------
# 22. Save result summary
# --------------------------------------------------

with open(
    RESULTS_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Model",
        "Pretrained Weights",
        "Image Size",
        "Classes",
        "Validation Loss",
        "Validation Accuracy"
    ])

    writer.writerow([
        "MobileNetV2 Transfer Learning",
        "ImageNet",
        f"{IMAGE_HEIGHT}x{IMAGE_WIDTH}",
        number_of_classes,
        round(validation_loss, 4),
        round(validation_accuracy, 4)
    ])


print("\nGenerated files:")

print("Model       :", MODEL_PATH)
print("Class names :", CLASS_NAMES_PATH)
print("Results     :", RESULTS_PATH)
print("History     :", HISTORY_PATH)