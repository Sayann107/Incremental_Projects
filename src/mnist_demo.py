import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense


# --------------------------------------------------
# 1. Load MNIST dataset
# --------------------------------------------------

(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

print("Training shape:", X_train.shape)
print("Testing shape :", X_test.shape)


# --------------------------------------------------
# 2. Normalise pixels
# --------------------------------------------------

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0


# --------------------------------------------------
# 3. Create neural network
# --------------------------------------------------

model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])


# --------------------------------------------------
# 4. Compile model
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# --------------------------------------------------
# 5. Train model
# --------------------------------------------------

model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)


# --------------------------------------------------
# 6. Evaluate model
# --------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nMNIST RESULTS")
print("Test loss    :", round(test_loss, 4))
print("Test accuracy:", round(test_accuracy, 4))