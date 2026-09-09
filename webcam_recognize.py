import cv2
import pickle
import face_recognition

MODEL_FILE = "models/knn_model.pkl"

# Start with the threshold we observed from your test data.
THRESHOLD = 0.55


# Load trained KNN model
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)


# Open laptop webcam
video = cv2.VideoCapture(0)

if not video.isOpened():
    raise RuntimeError("Could not open webcam.")


print("Webcam started.")
print("Press Q to quit.")


while True:

    ret, frame = video.read()

    if not ret:
        print("Could not read frame.")
        break


    # OpenCV uses BGR.
    # face_recognition expects RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)


    # Create embeddings
    face_encodings = face_recognition.face_encodings(
        rgb_frame,
        face_locations
    )


    for (top, right, bottom, left), encoding in zip(
        face_locations,
        face_encodings
    ):

        # Find closest known person
        predicted_person = model.predict([encoding])[0]


        # Find distance to closest training sample
        distance = model.kneighbors(
            [encoding],
            n_neighbors=1,
            return_distance=True
        )[0][0][0]


        # Unknown decision
        if distance >= THRESHOLD:
            name = "Unknown"
        else:
            name = predicted_person


        # Draw bounding box
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )


        # Display name
        cv2.putText(
            frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        # Display distance
        cv2.putText(
            frame,
            f"Distance: {distance:.3f}",
            (left, bottom + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


    # Show webcam
    cv2.imshow("Person Identification", frame)


    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


video.release()
cv2.destroyAllWindows()