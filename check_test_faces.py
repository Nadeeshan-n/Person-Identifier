import os
import face_recognition

TEST_DIR = "dataset/test"

for person_name in os.listdir(TEST_DIR):

    person_dir = os.path.join(TEST_DIR, person_name)

    if not os.path.isdir(person_dir):
        continue

    print(f"\n===== {person_name} =====")

    for filename in os.listdir(person_dir):

        image_path = os.path.join(person_dir, filename)

        try:
            image = face_recognition.load_image_file(image_path)
            locations = face_recognition.face_locations(image)

            print(f"{filename}: {len(locations)} face(s) detected")

        except Exception as e:
            print(f"{filename}: ERROR -> {e}")