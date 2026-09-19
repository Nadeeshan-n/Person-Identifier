import cv2, pickle, threading, time
import face_recognition

EMBEDDINGS_FILE = "models/embeddings.pkl"
THRESHOLD = 0.55
SCALE = 0.5          # detection scale (0.5 finds faces at a longer range than 0.2)

with open(EMBEDDINGS_FILE, "rb") as f:
    data = pickle.load(f)
known_encodings = data["encodings"]
known_names = data["names"]

latest_frame = None
results = []
lock = threading.Lock()
running = True


def recognition_worker():
    global results
    while running:
        with lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.01)
            continue

        # Detect on a small image (fast)
        small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs_small = face_recognition.face_locations(rgb_small, model="hog")

        # Scale boxes back to full size
        locs = [(int(t / SCALE), int(r / SCALE), int(b / SCALE), int(l / SCALE))
                for (t, r, b, l) in locs_small]

        # Encode on the FULL image (better embeddings, same speed)
        rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_full, locs, num_jitters=1)

        new_results = []
        for (top, right, bottom, left), enc in zip(locs, encodings):
            distances = face_recognition.face_distance(known_encodings, enc)
            i = distances.argmin()
            name = known_names[i] if distances[i] < THRESHOLD else "Unknown"
            new_results.append((left, top, right, bottom, name, distances[i]))

        results = new_results


video = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # CAP_DSHOW is faster on Windows
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video.set(cv2.CAP_PROP_BUFFERSIZE, 1)        # avoid old frames piling up

if not video.isOpened():
    raise RuntimeError("Could not open webcam")

threading.Thread(target=recognition_worker, daemon=True).start()

while True:
    ret, frame = video.read()
    if not ret:
        break

    with lock:
        latest_frame = frame

    for (left, top, right, bottom, name, dist) in results:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"{dist:.3f}", (left, bottom + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.imshow("Person Identification", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

running = False
video.release()
cv2.destroyAllWindows()