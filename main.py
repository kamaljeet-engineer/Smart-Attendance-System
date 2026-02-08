

import os
import cv2
import time
import csv
import pickle
from datetime import datetime
import face_recognition

# ================== PATHS ==================
DATASET_DIR = "dataset"
ATTENDANCE_DIR = "attendance"
ENCODING_FILE = "encodings.pickle"

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)


# ================== DAILY COUNT HELPER ==================
def get_today_count(file_path, name):
if not os.path.exists(file_path):
return 0

count = 0
with open(file_path, "r") as f:
reader = csv.reader(f)
next(reader)  # skip header
for row in reader:
if row[0] == name:
count += 1
return count


# ================== AUTO FACE CAPTURE ==================
def auto_capture_faces():
name = input("Enter student name: ").strip()
if name == "":
print("❌ Name required")
return

person_dir = os.path.join(DATASET_DIR, name)
os.makedirs(person_dir, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
print("❌ Camera not opened")
return

    MAX_IMAGES = 5
    MAX_IMAGES = 20
count = 0
last_time = time.time()

print("📸 Auto face capture started")
print("➡️ Look at camera | Auto close after capture")

while True:
ret, frame = cap.read()
if not ret:
break

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
locations = face_recognition.face_locations(rgb)

for (top, right, bottom, left) in locations:
cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

if len(locations) == 1:
if time.time() - last_time > 0.7:
img_path = os.path.join(person_dir, f"{count}.jpg")
cv2.imwrite(img_path, frame)
print(f"[SAVED] {img_path}")
count += 1
last_time = time.time()

cv2.putText(frame, f"Images: {count}/{MAX_IMAGES}",
(20, 40), cv2.FONT_HERSHEY_SIMPLEX,
1, (0, 255, 0), 2)

cv2.imshow("Auto Face Capture", frame)

if count >= MAX_IMAGES:
break

if cv2.waitKey(1) & 0xFF == ord('q'):
break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
print("✅ Face capture completed")


# ================== TRAIN MODEL ==================
def train_model():
encodings = []
names = []

print("[INFO] Training started...")

for person in os.listdir(DATASET_DIR):
person_path = os.path.join(DATASET_DIR, person)
if not os.path.isdir(person_path):
continue

for img in os.listdir(person_path):
img_path = os.path.join(person_path, img)
image = face_recognition.load_image_file(img_path)
locations = face_recognition.face_locations(image)

if len(locations) != 1:
continue

encoding = face_recognition.face_encodings(image, locations)[0]
encodings.append(encoding)
names.append(person)

if len(encodings) == 0:
print("❌ No data to train")
return

with open(ENCODING_FILE, "wb") as f:
pickle.dump({"encodings": encodings, "names": names}, f)

print("✅ Training completed")


# ================== MARK ATTENDANCE ==================
def mark_attendance():
if not os.path.exists(ENCODING_FILE):
print("❌ Train model first")
return

with open(ENCODING_FILE, "rb") as f:
data = pickle.load(f)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
print("❌ Camera not opened")
return

attendance_done = False

date = datetime.now().strftime("%Y-%m-%d")
file_path = os.path.join(ATTENDANCE_DIR, f"attendance_{date}.csv")

if not os.path.exists(file_path):
with open(file_path, "w", newline="") as f:
csv.writer(f).writerow(["Name", "Time"])

print("🟢 Attendance started (auto close after marking)")

while True:
ret, frame = cap.read()
if not ret:
break

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
locations = face_recognition.face_locations(rgb)
encodings = face_recognition.face_encodings(rgb, locations)

for encoding, (top, right, bottom, left) in zip(encodings, locations):
distances = face_recognition.face_distance(data["encodings"], encoding)
best_match = distances.argmin()

name = "Unknown"
if distances[best_match] < 0.45:
name = data["names"][best_match]

today_count = get_today_count(file_path, name)
if today_count >= 2:
print(f"⚠️ {name} attendance limit reached (2 times)")
continue

time_now = datetime.now().strftime("%H:%M:%S")
with open(file_path, "a", newline="") as f:
csv.writer(f).writerow([name, time_now])

attendance_done = True
print(f"[ATTENDANCE MARKED] {name}")

color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
cv2.putText(frame, name, (left, top - 10),
cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

cv2.imshow("Smart Attendance", frame)

if attendance_done:
cv2.putText(frame, "Attendance Done",
(50, 50), cv2.FONT_HERSHEY_SIMPLEX,
1, (0, 255, 0), 3)
cv2.imshow("Smart Attendance", frame)
cv2.waitKey(2000)
break

if cv2.waitKey(1) & 0xFF == ord('q'):
break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
print("✅ Camera closed automatically")


# ================== MAIN MENU ==================
def main():
while True:
print("\n===== SMART ATTENDANCE SYSTEM =====")
print("1. Auto Capture Face")
print("2. Train Face Model")
print("3. Mark Attendance")
print("4. Exit")

choice = input("Enter choice: ").strip()

if choice == "1":
auto_capture_faces()
elif choice == "2":
train_model()
elif choice == "3":
mark_attendance()
elif choice == "4":
print("👋 Exiting safely")
break
else:
print("❌ Invalid choice")


if __name__ == "__main__":
main()