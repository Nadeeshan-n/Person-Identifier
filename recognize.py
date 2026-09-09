import pickle
import face_recognition

MODEL_FILE = "models/knn_model.pkl"

# Start with this as an experiment.
# We will tune it later using your own test data.
THRESHOLD = 0.45

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

image = face_recognition.load_image_file("test_images/test.jpeg")

locations = face_recognition.face_locations(image)
encodings = face_recognition.face_encodings(image, locations)

if not encodings:
    print("No face detected.")
    raise SystemExit

for i, encoding in enumerate(encodings):

    prediction = model.predict([encoding])[0]

    distances = model.kneighbors(
        [encoding],
        n_neighbors=1,
        return_distance=True
    )[0]

    distance = distances[0][0]

    print(f"\nFace {i + 1}")
    print(f"Distance: {distance:.4f}")

    if distance < THRESHOLD:
        print(f"Recognized: {prediction}")
    else:
        print("Unknown person")