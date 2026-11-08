from pathlib import Path
import csv
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping


# --------------------------------------------------
# 1. Settings and paths
# --------------------------------------------------

RANDOM_STATE = 42
BATCH_SIZE = 32
MAX_EPOCHS = 30

MODEL_DIRECTORY = Path("models")
MODEL_PATH = MODEL_DIRECTORY / "fashion_mnist_cnn.keras"

RESULTS_PATH = (
    MODEL_DIRECTORY /
    "day6_fashion_mnist_results.csv"
)

MODEL_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

tf.keras.utils.set_random_seed(
    RANDOM_STATE
)


# --------------------------------------------------
# 2. Fashion-MNIST class names
# --------------------------------------------------

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


# --------------------------------------------------
# 3. Load Fashion-MNIST
# --------------------------------------------------

(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

print("=" * 60)
print("DAY 6: FASHION-MNIST CNN")
print("=" * 60)

print("Training shape:", X_train.shape)
print("Testing shape :", X_test.shape)


# --------------------------------------------------
# 4. Normalise pixel values
# --------------------------------------------------

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0


# --------------------------------------------------
# 5. Add channel dimension
# --------------------------------------------------

X_train = X_train[..., tf.newaxis]
X_test = X_test[..., tf.newaxis]

print("CNN training shape:", X_train.shape)
print("CNN testing shape :", X_test.shape)


# --------------------------------------------------
# 6. Build CNN
# --------------------------------------------------

model = Sequential([
    Input(shape=(28, 28, 1)),

    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),

    MaxPooling2D(
        pool_size=(2, 2)
    ),

    Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),

    MaxPooling2D(
        pool_size=(2, 2)
    ),

    Dropout(0.25),

    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.40),

    Dense(
        10,
        activation="softmax"
    )
])


# --------------------------------------------------
# 7. Compile CNN
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN architecture:")
model.summary()


# --------------------------------------------------
# 8. Early stopping
# --------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)


# --------------------------------------------------
# 9. Train CNN
# --------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1
)


# --------------------------------------------------
# 10. Evaluate on test data
# --------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n" + "=" * 60)
print("FASHION-MNIST TEST RESULTS")
print("=" * 60)

print(f"Test loss    : {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")


# --------------------------------------------------
# 11. Predict classes
# --------------------------------------------------

probabilities = model.predict(
    X_test,
    verbose=0
)

predictions = tf.argmax(
    probabilities,
    axis=1
)


# --------------------------------------------------
# 12. Confusion matrix
# --------------------------------------------------

confusion_matrix = tf.math.confusion_matrix(
    y_test,
    predictions,
    num_classes=len(CLASS_NAMES)
).numpy()

print("\nConfusion matrix:")
print(confusion_matrix)


# --------------------------------------------------
# 13. Save model
# --------------------------------------------------

model.save(MODEL_PATH)


# --------------------------------------------------
# 14. Save test results
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
        "Dataset",
        "Test Loss",
        "Test Accuracy",
        "Epochs Completed"
    ])

    writer.writerow([
        "Fashion-MNIST CNN",
        "Fashion-MNIST",
        round(test_loss, 4),
        round(test_accuracy, 4),
        len(history.history["loss"])
    ])


# --------------------------------------------------
# 15. Save confusion matrix
# --------------------------------------------------

confusion_path = (
    MODEL_DIRECTORY /
    "day6_fashion_mnist_confusion_matrix.csv"
)

with open(
    confusion_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow(
        ["Actual / Predicted"] +
        CLASS_NAMES
    )

    for index, row in enumerate(
        confusion_matrix
    ):
        writer.writerow(
            [CLASS_NAMES[index]] +
            row.tolist()
        )


# --------------------------------------------------
# 16. Save training history
# --------------------------------------------------

history_path = (
    MODEL_DIRECTORY /
    "day6_fashion_mnist_history.csv"
)

history_keys = list(
    history.history.keys()
)

with open(
    history_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "epoch",
        *history_keys
    ])

    number_of_epochs = len(
        history.history["loss"]
    )

    for epoch in range(
        number_of_epochs
    ):
        writer.writerow([
            epoch + 1,
            *[
                history.history[key][epoch]
                for key in history_keys
            ]
        ])


print("\nSaved files:")

print("Model            :", MODEL_PATH)
print("Results          :", RESULTS_PATH)
print("Confusion matrix :", confusion_path)
print("Training history :", history_path)
