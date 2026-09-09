import pickle
import face_recognition

MODEL_FILE = "models/knn_model.pkl"

# Lower distance = more similar
# Start with this value, then tune it using your test results.
THRESHOLD = 0.50

with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)

image_path = "test_images/test.jpg"

image = face_recognition.load_image_file(image_path)

face_locations = face_recognition.face_locations(image)

if len(face_locations) == 0:
    print("No face detected.")
    raise SystemExit

face_encodings = face_recognition.face_encodings(
    image,
    face_locations
)

for i, encoding in enumerate(face_encodings):

    prediction = model.predict([encoding])[0]

    distances = model.kneighbors(
        [encoding],
        n_neighbors=1,
        return_distance=True
    )[0]

    distance = distances[0][0]

    if distance <= THRESHOLD:
        result = prediction
    else:
        result = "Unknown"

    print(f"Face {i + 1}")
    print(f"Nearest person: {prediction}")
    print(f"Distance: {distance:.4f}")
    print(f"Final result: {result}")