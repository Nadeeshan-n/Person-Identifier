import pickle
import face_recognition

MODEL_FILE = "models/knn_model.pkl"

with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)

image_path = "test_images/test.jpg"

image = face_recognition.load_image_file(image_path)

face_locations = face_recognition.face_locations(image)
face_encodings = face_recognition.face_encodings(
    image,
    face_locations
)

if not face_encodings:
    print("No face detected.")
    raise SystemExit

for index, encoding in enumerate(face_encodings):

    prediction = model.predict([encoding])[0]

    distances = model.kneighbors(
        [encoding],
        n_neighbors=1,
        return_distance=True
    )[0]

    distance = distances[0][0]

    print(f"Face {index + 1}")
    print(f"Predicted person: {prediction}")
    print(f"Distance: {distance:.4f}")