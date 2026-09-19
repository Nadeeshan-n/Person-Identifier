"""Shared helpers for Person-Identifier.

Every script loads images and identifies faces through these
functions, so training, evaluation and live recognition behave
the same way.
"""

import os
import pickle

import numpy as np
from PIL import Image, ImageOps

import config


# ---------------------------------------------------------------
# Images
# ---------------------------------------------------------------

def load_image(image_path, max_dimension=config.MAX_DIMENSION):
    """Load an image as an RGB numpy array.

    - Fixes phone-camera rotation (EXIF)
    - Converts to RGB
    - Shrinks very large images
    """
    pil_image = Image.open(image_path)
    pil_image = ImageOps.exif_transpose(pil_image)
    pil_image = pil_image.convert("RGB")

    width, height = pil_image.size
    scale = min(1.0, max_dimension / max(width, height))

    if scale < 1.0:
        new_size = (int(width * scale), int(height * scale))
        pil_image = pil_image.resize(new_size)

    return np.array(pil_image)


def list_person_folders(root_dir):
    """Return (person_name, folder_path) for each sub-folder of root_dir."""
    folders = []
    for name in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, name)
        if os.path.isdir(path):
            folders.append((name, path))
    return folders


def list_images(folder):
    """Return full paths of all image files in a folder."""
    paths = []
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(config.IMAGE_EXTENSIONS):
            paths.append(os.path.join(folder, filename))
    return paths


# ---------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------

def load_embeddings(path=config.EMBEDDINGS_FILE):
    """Load saved embeddings. Returns (encodings array, names list)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Run create_embeddings.py first."
        )

    with open(path, "rb") as file:
        data = pickle.load(file)

    encodings = np.array(data["encodings"])
    names = list(data["names"])

    if len(encodings) == 0:
        raise ValueError("The embeddings file has no faces in it.")

    return encodings, names


def save_embeddings(encodings, names, path=config.EMBEDDINGS_FILE):
    """Save embeddings to disk."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(path, "wb") as file:
        pickle.dump({"encodings": encodings, "names": names}, file)


# ---------------------------------------------------------------
# Identification
# ---------------------------------------------------------------

def nearest_person(encoding, known_encodings, known_names):
    """Find the closest saved face.

    Returns (name, distance) where BOTH come from the same saved
    face, so the name and the distance always match.
    """
    # Same formula as face_recognition.face_distance()
    distances = np.linalg.norm(known_encodings - encoding, axis=1)
    index = int(np.argmin(distances))
    return known_names[index], float(distances[index])


def identify(encoding, known_encodings, known_names,
             threshold=config.THRESHOLD):
    """Identify one face.

    Returns (final_name, nearest_name, distance).
    final_name is "Unknown" when the distance is not below threshold.
    """
    nearest_name, distance = nearest_person(
        encoding, known_encodings, known_names
    )

    if distance < threshold:
        return nearest_name, nearest_name, distance

    return config.UNKNOWN_LABEL, nearest_name, distance