import os
import pickle

import face_recognition
import numpy as np
from PIL import Image, ImageOps

DATASET_DIR = "dataset/train"
OUTPUT_FILE = "models/embeddings.pkl"

known_encodings = []
known_names = []

for person_name in os.listdir(DATASET_DIR):

    person_folder = os.path.join(DATASET_DIR, person_name)

    if not os.path.isdir(person_folder):
        continue

    print(f"\nProcessing: {person_name}")

    for filename in os.listdir(person_folder):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        image_path = os.path.join(person_folder, filename)

        try:
            # Open image and correct phone-camera orientation
            pil_image = Image.open(image_path)
            pil_image = ImageOps.exif_transpose(pil_image)
            pil_image = pil_image.convert("RGB")

            # Resize large images
            max_dimension = 1600

            width, height = pil_image.size
            scale = min(
                1.0,
                max_dimension / max(width, height)
            )

            if scale < 1.0:
                new_size = (
                    int(width * scale),
                    int(height * scale)
                )
                pil_image = pil_image.resize(new_size)

            image = np.array(pil_image)

            # Detect faces
            face_locations = face_recognition.face_locations(
                image,
                model="hog"
            )

            print(
                f"{filename}: "
                f"{len(face_locations)} face(s)"
            )

            # We need exactly one face
            if len(face_locations) != 1:
                print("  Skipping image")
                continue

            # Generate 128-dimensional face embedding
            encoding = face_recognition.face_encodings(
                image,
                face_locations
            )[0]

            known_encodings.append(encoding)
            known_names.append(person_name)

            print("  Embedding created")

        except Exception as error:
            print(f"  Error: {error}")


# Save embeddings
data = {
    "encodings": known_encodings,
    "names": known_names
}

os.makedirs("models", exist_ok=True)

with open(OUTPUT_FILE, "wb") as file:
    pickle.dump(data, file)

print("\n==============================")
print("Embedding generation complete")
print("Total faces:", len(known_encodings))
print("People:", sorted(set(known_names)))
print("Saved:", OUTPUT_FILE)
print("==============================")