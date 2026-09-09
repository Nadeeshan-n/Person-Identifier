import cv2

image = cv2.imread("test_images/test.jpeg")

if image is None:
    raise FileNotFoundError("Could not find test_images/test.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

faces = face_detector.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(50, 50)
)

for (x, y, w, h) in faces:
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

cv2.imshow("Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()