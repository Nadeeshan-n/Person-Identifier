import os
import pickle
import face_recognition
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

MODEL_FILE = "models/knn_model.pkl"
TEST_DIR = "dataset/test"

# Temporary threshold.
# We will tune this after seeing the complete distance distribution.
THRESHOLD = 0.55


with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)


y_true = []
y_pred = []


for folder_name in os.listdir(TEST_DIR):

    folder_path = os.path.join(TEST_DIR, folder_name)

    if not os.path.isdir(folder_path):
        continue

    # person3(kasun) is our unknown test person
    if folder_name == "person3(kasun)":
        actual_label = "Unknown"
    else:
        actual_label = folder_name


    for filename in os.listdir(folder_path):

        image_path = os.path.join(folder_path, filename)

        try:

            image = face_recognition.load_image_file(image_path)

            locations = face_recognition.face_locations(image)

            if len(locations) != 1:

                print(
                    f"Skipping {image_path}: "
                    f"expected 1 face, found {len(locations)}"
                )

                continue


            encoding = face_recognition.face_encodings(
                image,
                locations
            )[0]


            # Get nearest known person
            predicted_person = model.predict([encoding])[0]


            # Get distance to nearest training sample
            distance = model.kneighbors(
                [encoding],
                n_neighbors=1,
                return_distance=True
            )[0][0][0]


            # Unknown-person decision
            if distance >= THRESHOLD:
                predicted_person = "Unknown"


            y_true.append(actual_label)
            y_pred.append(predicted_person)


            print(
                f"{filename:25} "
                f"Actual: {actual_label:20} "
                f"Predicted: {predicted_person:20} "
                f"Distance: {distance:.4f}"
            )


        except Exception as e:

            print(
                f"Error processing {image_path}: {e}"
            )


if not y_true:

    print("No valid test images found.")
    raise SystemExit


accuracy = accuracy_score(y_true, y_pred)


print("\n" + "=" * 70)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("=" * 70)


labels = sorted(set(y_true) | set(y_pred))


print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0
    )
)


print("\nConfusion Matrix:")

matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)

print("Labels:")
print(labels)

print()
print(matrix)