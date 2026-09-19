import os
import sys
import pickle
import face_recognition

MODEL_FILE = "models/knn_model.pkl"
THRESHOLD = 0.55


# --------------------------------------------------
# Get image path from command line
# --------------------------------------------------

if len(sys.argv) < 2:
    print("Usage:")
    print('python recognize.py "path/to/image.jpg"')
    raise SystemExit


image_path = sys.argv[1]


if not os.path.isfile(image_path):
    print(f"Image not found: {image_path}")
    raise SystemExit


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)


# --------------------------------------------------
# Load image
# --------------------------------------------------

image = face_recognition.load_image_file(image_path)


# --------------------------------------------------
# Detect faces
# --------------------------------------------------

face_locations = face_recognition.face_locations(image)

if len(face_locations) == 0:
    print("No face detected.")
    raise SystemExit


face_encodings = face_recognition.face_encodings(
    image,
    face_locations
)


# --------------------------------------------------
# Recognize each face
# --------------------------------------------------

for i, encoding in enumerate(face_encodings):

    prediction = model.predict([encoding])[0]

    distance = model.kneighbors(
        [encoding],
        n_neighbors=1,
        return_distance=True
    )[0][0][0]

    if distance <= THRESHOLD:
        result = prediction
    else:
        result = "Unknown"

    print(f"\nFace {i + 1}")
    print(f"Nearest person: {prediction}")
    print(f"Distance: {distance:.4f}")
    print(f"Final result: {result}")