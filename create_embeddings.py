import os
import pickle
import face_recognition

DATASET_DIR = "dataset"
OUTPUT_FILE = "models/embeddings.pkl"

known_encodings = []
known_names = []

for person_name in os.listdir(DATASET_DIR):

    person_folder = os.path.join(DATASET_DIR, person_name)

    if not os.path.isdir(person_folder):
        continue

    print(f"\nProcessing: {person_name}")

    for filename in os.listdir(person_folder):

        image_path = os.path.join(person_folder, filename)

        try:
            image = face_recognition.load_image_file(image_path)

            face_locations = face_recognition.face_locations(image)

            if len(face_locations) != 1:
                print(f"Skipping {filename}: expected exactly one face.")
                continue

            encoding = face_recognition.face_encodings(
                image,
                face_locations
            )[0]

            known_encodings.append(encoding)
            known_names.append(person_name)

            print(f"  Added: {filename}")

        except Exception as error:
            print(f"  Error processing {filename}: {error}")

data = {
    "encodings": known_encodings,
    "names": known_names
}

os.makedirs("models", exist_ok=True)

with open(OUTPUT_FILE, "wb") as file:
    pickle.dump(data, file)

print("\nEmbeddings saved to:", OUTPUT_FILE)
print("Total faces:", len(known_encodings))