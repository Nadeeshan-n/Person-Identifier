import cv2
import pickle
import face_recognition

MODEL_FILE = "models/knn_model.pkl"

THRESHOLD = 0.55

# Performance
PROCESS_EVERY = 5
SCALE = 0.25

# Load model
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

# Open webcam
video = cv2.VideoCapture(0)

video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not video.isOpened():
    raise RuntimeError("Could not open webcam")

print("Part 2 - Real-Time Multi-Person Recognition")
print("Press Q to quit")

frame_count = 0
last_results = []


while True:

    ret, frame = video.read()

    if not ret:
        print("Could not read frame.")
        break

    frame_count += 1

    # Run recognition every few frames
    if frame_count % PROCESS_EVERY == 0:

        # Resize for faster processing
        small_frame = cv2.resize(
            frame,
            (0, 0),
            fx=SCALE,
            fy=SCALE
        )

        # BGR -> RGB
        rgb_small = cv2.cvtColor(
            small_frame,
            cv2.COLOR_BGR2RGB
        )

        # Detect faces
        face_locations = face_recognition.face_locations(
            rgb_small,
            model="hog"
        )

        # Create embeddings
        face_encodings = face_recognition.face_encodings(
            rgb_small,
            face_locations,
            num_jitters=0,
            model="small"
        )

        results = []

        for location, encoding in zip(
            face_locations,
            face_encodings
        ):

            # Predict closest known person
            prediction = model.predict(
                [encoding]
            )[0]

            # Calculate distance
            distance = model.kneighbors(
                [encoding],
                n_neighbors=1,
                return_distance=True
            )[0][0][0]

            # Unknown detection
            if distance >= THRESHOLD:
                name = "Unknown"
            else:
                name = prediction

            # Original small-frame coordinates
            top, right, bottom, left = location

            # Scale back to 640x480
            top = int(top / SCALE)
            right = int(right / SCALE)
            bottom = int(bottom / SCALE)
            left = int(left / SCALE)

            results.append({
                "box": (left, top, right, bottom),
                "name": name,
                "distance": distance
            })

        last_results = results

    # Draw cached recognition results
    for result in last_results:

        left, top, right, bottom = result["box"]
        name = result["name"]
        distance = result["distance"]

        # Face box
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )

        # Name
        cv2.putText(
            frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Distance
        cv2.putText(
            frame,
            f"Distance: {distance:.3f}",
            (left, bottom + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    # Number of detected people
    cv2.putText(
        frame,
        f"People: {len(last_results)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Show
    cv2.imshow(
        "Part 2 - Person Recognition",
        frame
    )

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


video.release()
cv2.destroyAllWindows()