import cv2
import os
import time

# ================== SETTINGS ==================
NAME = input("Enter student name: ").strip()
SAVE_DIR = os.path.join("dataset", NAME)
os.makedirs(SAVE_DIR, exist_ok=True)
MAX_IMAGES = 8       # kitni photos chahiye
DELAY = 0.5            # har photo ke beech ka gap (seconds)

# ==============================================

# Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
count = 0
last_capture = time.time()

print("📸 Auto face capture started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        # Time delay control
        if time.time() - last_capture > DELAY:
            img_path = os.path.join(SAVE_DIR, f"{count}.jpg")
            cv2.imwrite(img_path, face_img)
            print(f"Saved: {img_path}")
            count += 1
            last_capture = time.time()

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Auto Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if count >= MAX_IMAGES:
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Face capture completed")

