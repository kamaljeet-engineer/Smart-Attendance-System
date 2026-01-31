import os
import cv2
import pickle
import face_recognition

# ================== PATHS ==================
DATASET_DIR = "dataset"
ENCODING_FILE = "encodings.pickle"

known_encodings = []
known_names = []

print("[INFO] Starting face encoding process...")

# Loop through each person folder
for person_name in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person_name)

    if not os.path.isdir(person_path):
        continue

    print(f"[INFO] Encoding faces for: {person_name}")

    # Loop through images of the person
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            continue

        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face locations
        boxes = face_recognition.face_locations(rgb, model="hog")

        # Encode faces
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(person_name)

print("[INFO] Saving encodings to file...")

data = {
    "encodings": known_encodings,
    "names": known_names
}

with open(ENCODING_FILE, "wb") as f:
    pickle.dump(data, f)

print("[SUCCESS] Face encodings completed and saved!")


