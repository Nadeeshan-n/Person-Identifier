import cv2
import pickle
import face_recognition
import time


MODEL_FILE = "models/knn_model.pkl"

THRESHOLD = 0.55

# Performance settings
PROCESS_EVERY = 20
SCALE = 0.20


# Load KNN model
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)


# Camera
video = cv2.VideoCapture(0)

video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


if not video.isOpened():
    raise RuntimeError("Could not open webcam")


print("Optimized webcam started")
print("Press Q to quit")


frame_count = 0
last_results = []

# Cache last recognized person
last_name = "Unknown"
last_distance = 0.0


while True:

    ret, frame = video.read()

    if not ret:
        break


    frame_count += 1


    # Run AI only every few frames
    if frame_count % PROCESS_EVERY == 0:


        # Resize for faster detection
        small_frame = cv2.resize(
            frame,
            (0, 0),
            fx=SCALE,
            fy=SCALE
        )


        rgb_small = cv2.cvtColor(
            small_frame,
            cv2.COLOR_BGR2RGB
        )


        # Fast face detection
        face_locations = face_recognition.face_locations(
            rgb_small,
            model="hog"
        )


        start = time.time()


        face_encodings = face_recognition.face_encodings(
            rgb_small,
            face_locations,
            num_jitters=0,
            model="small"
        )


        print("Encoding time:", time.time() - start)


        results = []


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


            if distance >= THRESHOLD:
                name = "Unknown"
            else:
                name = prediction

            # Save latest recognition result
            last_name = name
            last_distance = distance



            top, right, bottom, left = location


            # Convert coordinates back
            top = int(top / SCALE)
            right = int(right / SCALE)
            bottom = int(bottom / SCALE)
            left = int(left / SCALE)


            results.append(
                (
                    left,
                    top,
                    right,
                    bottom,
                    name,
                    distance
                )
            )


        last_results = results



    # Draw previous AI results
    for (
        left,
        top,
        right,
        bottom,
        name,
        distance
    ) in last_results:


        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0,255,0),
            2
        )


        cv2.putText(
            frame,
            name,
            (left, top-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )


        cv2.putText(
            frame,
            f"Distance: {distance:.3f}",
            (left, bottom+25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )


    cv2.imshow(
        "Person Identification",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



video.release()
cv2.destroyAllWindows()