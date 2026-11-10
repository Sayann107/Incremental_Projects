from pathlib import Path
import csv
import sys

from ultralytics import YOLO


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

MODEL_NAME = "yolo11n.pt"

OUTPUT_DIRECTORY = Path(
    "outputs/day7_detection"
)

RESULTS_PATH = Path(
    "models/day7_detection_results.csv"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# 2. Choose source
# --------------------------------------------------

if len(sys.argv) >= 2:

    source = sys.argv[1]

else:

    source = (
        "https://ultralytics.com/"
        "images/bus.jpg"
    )

    print(
        "No local image was provided."
    )

    print(
        "Using the Ultralytics sample image."
    )


# --------------------------------------------------
# 3. Load pretrained detector
# --------------------------------------------------

model = YOLO(
    MODEL_NAME
)


# --------------------------------------------------
# 4. Run object detection
# --------------------------------------------------

results = model.predict(
    source=source,
    conf=0.25,
    save=True,
    project=str(
        OUTPUT_DIRECTORY.parent
    ),
    name=OUTPUT_DIRECTORY.name,
    exist_ok=True,
    verbose=True
)


# --------------------------------------------------
# 5. Extract detection details
# --------------------------------------------------

detection_rows = []

for result in results:

    image_name = Path(
        str(result.path)
    ).name

    if result.boxes is None:
        continue

    for box in result.boxes:

        class_id = int(
            box.cls[0].item()
        )

        confidence = float(
            box.conf[0].item()
        )

        coordinates = (
            box.xyxy[0]
            .cpu()
            .tolist()
        )

        x1, y1, x2, y2 = coordinates

        class_name = result.names[
            class_id
        ]

        detection_rows.append({
            "image": image_name,
            "class_id": class_id,
            "detected_object": class_name,
            "confidence": round(
                confidence,
                4
            ),
            "x1": round(x1, 2),
            "y1": round(y1, 2),
            "x2": round(x2, 2),
            "y2": round(y2, 2)
        })


# --------------------------------------------------
# 6. Save detection CSV
# --------------------------------------------------

field_names = [
    "image",
    "class_id",
    "detected_object",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2"
]

with open(
    RESULTS_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=field_names
    )

    writer.writeheader()

    writer.writerows(
        detection_rows
    )


# --------------------------------------------------
# 7. Display results
# --------------------------------------------------

print("\n" + "=" * 60)

print("DAY 7 OBJECT DETECTION RESULTS")

print("=" * 60)

if detection_rows:

    for detection in detection_rows:

        print(
            detection["detected_object"],
            "| Confidence:",
            detection["confidence"],
            "| Box:",
            (
                detection["x1"],
                detection["y1"],
                detection["x2"],
                detection["y2"]
            )
        )

else:

    print(
        "No objects were detected "
        "above the confidence threshold."
    )


print("\nAnnotated image directory:")

print(OUTPUT_DIRECTORY)

print("\nDetection result CSV:")

print(RESULTS_PATH)