from pathlib import Path
import sys
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 160

MODEL_PATH = Path(
    "models/fashion_transfer_model.keras"
)

CLASS_NAMES_PATH = Path(
    "models/day7_class_names.txt"
)


# --------------------------------------------------
# 2. Check generated files
# --------------------------------------------------

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        "Transfer-learning model not found.\n"
        "Run src/train_transfer_learning.py first."
    )


if not CLASS_NAMES_PATH.exists():

    raise FileNotFoundError(
        "Class names file not found.\n"
        "Run src/train_transfer_learning.py first."
    )


# --------------------------------------------------
# 3. Load model
# --------------------------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH
)


# --------------------------------------------------
# 4. Load class names
# --------------------------------------------------

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:

    class_names = [
        line.strip()
        for line in file
        if line.strip()
    ]


# --------------------------------------------------
# 5. Prediction function
# --------------------------------------------------

def predict_product(image_path):

    image_path = Path(image_path)

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = tf.keras.utils.load_img(
        image_path,
        target_size=(
            IMAGE_HEIGHT,
            IMAGE_WIDTH
        )
    )

    image_array = (
        tf.keras.utils.img_to_array(
            image
        )
    )

    image_array = tf.expand_dims(
        image_array,
        axis=0
    )

    image_array = preprocess_input(
        image_array
    )

    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(
        tf.argmax(probabilities)
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    return {
        "predicted_category": predicted_class,
        "confidence": round(
            confidence,
            4
        )
    }


# --------------------------------------------------
# 6. Command-line use
# --------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage:\n"
            "python src/predict_transfer_learning.py "
            "\"path/to/product.jpg\""
        )

        raise SystemExit(1)

    result = predict_product(
        sys.argv[1]
    )

    print("\nTransfer-learning prediction:")

    print(result)