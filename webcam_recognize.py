import cv2
import pickle
import face_recognition

MODEL_FILE = "models/knn_model.pkl"

THRESHOLD = 0.55

# Performance settings
PROCESS_EVERY = 3
SCALE = 0.50


# -----------------------------
# Load trained KNN model
# -----------------------------

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully.")


# -----------------------------
# Start webcam
# -----------------------------

video = cv2.VideoCapture(0)

video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not video.isOpened():
    raise RuntimeError("Could not open webcam")


print("Real-time recognition started.")
print("Press Q to quit.")


frame_count = 0

last_results = []


# -----------------------------
# Main loop
# -----------------------------

while True:

    ret, frame = video.read()

    if not ret:
        print("Could not read frame.")
        break

    frame_count += 1


    # -------------------------
    # Process every few frames
    # -------------------------

    if frame_count % PROCESS_EVERY == 0:

        # Resize
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


        # ---------------------
        # Detect faces
        # ---------------------

        face_locations = face_recognition.face_locations(
            rgb_small,
            model="hog"
        )


        # ---------------------
        # Generate embeddings
        # ---------------------

        face_encodings = face_recognition.face_encodings(
            rgb_small,
            face_locations,
            num_jitters=0,
            model="small"
        )


        results = []


        # ---------------------
        # Recognize each face
        # ---------------------

        for location, encoding in zip(
            face_locations,
            face_encodings
        ):

            prediction = model.predict(
                [encoding]
            )[0]


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


            # Original coordinates
            top, right, bottom, left = location


            # Convert back to full frame
            top = int(top / SCALE)
            right = int(right / SCALE)
            bottom = int(bottom / SCALE)
            left = int(left / SCALE)


            results.append({
                "box": (
                    left,
                    top,
                    right,
                    bottom
                ),
                "name": name,
                "distance": distance
            })


        # Save latest results
        last_results = results


    # -----------------------------
    # Draw results
    # -----------------------------

    for result in last_results:

        left, top, right, bottom = result["box"]

        name = result["name"]

        distance = result["distance"]


        # Face rectangle
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


    # Number of faces
    cv2.putText(
        frame,
        f"Faces: {len(last_results)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # Display
    cv2.imshow(
        "Person Identification - Part 2",
        frame
    )


    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# Cleanup
# -----------------------------

video.release()
cv2.destroyAllWindows()