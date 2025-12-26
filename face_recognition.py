import os
import threading
import time

import cv2

PERSON_NAME = "Mohammed Alqarni"

# ============================
# Audio settings (your files)
# ============================
VOICES_DIR = "voices"
KNOWN_SOUND = "m1.mp3"        # plays when recognized
UNKNOWN_SOUND = "!.mp3"       # plays when face is Unknown (literal file name "!")

KNOWN_SOUND_PATH = os.path.join(VOICES_DIR, KNOWN_SOUND)
UNKNOWN_SOUND_PATH = os.path.join(VOICES_DIR, UNKNOWN_SOUND)

# Prevent sound spamming
SOUND_COOLDOWN_SEC = 2.5
_last_sound_time = 0.0
_sound_lock = threading.Lock()

# ============================
# Audio player (pygame mixer)
# ============================
_audio_ok = False
try:
    import pygame

    pygame.mixer.init()
    _audio_ok = True
except Exception as e:
    print(
        "[WARN] Audio disabled. Install pygame: pip install pygame\n"
        f"Reason: {e}"
    )


def play_sound(filepath: str):
    """Play an mp3/wav without blocking the camera loop."""
    global _last_sound_time

    if not _audio_ok:
        return
    if not os.path.exists(filepath):
        print(f"[WARN] Sound file not found: {filepath}")
        return

    with _sound_lock:
        now = time.time()
        if now - _last_sound_time < SOUND_COOLDOWN_SEC:
            return
        _last_sound_time = now

    def _worker():
        try:
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Failed to play sound: {exc}")

    threading.Thread(target=_worker, daemon=True).start()


# ============================
# Load reference image
# ============================
KNOWN_DIR = "known_faces"
REF_IMAGE_NAME = "mohammed.jpg"

ref_path = os.path.join(KNOWN_DIR, REF_IMAGE_NAME)

if not os.path.exists(ref_path):
    print(f"[ERROR] Reference image not found: {ref_path}")
    exit(1)

ref_img = cv2.imread(ref_path)
if ref_img is None:
    print("[ERROR] Unable to read the reference image.")
    exit(1)

ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
)

faces = face_cascade.detectMultiScale(ref_gray, scaleFactor=1.1, minNeighbors=6)
if len(faces) == 0:
    print("[ERROR] No face detected in the reference image.")
    exit(1)

(x, y, w, h) = faces[0]
ref_face = ref_gray[y : y + h, x : x + w]
ref_face = cv2.resize(ref_face, (100, 100))

print("[INFO] Reference image loaded successfully.")

# ============================
# Start camera
# ============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not access the camera.")
    exit(1)

# flags to avoid repeating sounds every frame
recognized_latched = False
unknown_latched = False

print("[INFO] Camera is running... Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame from camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_cam = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=6,
        minSize=(50, 50),
    )

    recognized_this_frame = False
    unknown_this_frame = False

    for (fx, fy, fw, fh) in faces_cam:
        face_roi = gray[fy : fy + fh, fx : fx + fw]
        face_roi_resized = cv2.resize(face_roi, (100, 100))

        diff = cv2.norm(ref_face, face_roi_resized, cv2.NORM_L2)
        diff_normalized = diff / float(ref_face.size)

        THRESHOLD = 25.0  # stricter match: lower threshold rejects more faces

        if diff_normalized < THRESHOLD:
            name = PERSON_NAME
            color = (0, 255, 0)
            recognized_this_frame = True
        else:
            name = "unknown"
            color = (0, 0, 255)
            unknown_this_frame = True

        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), color, 2)
        cv2.putText(
            frame,
            name,
            (fx, fy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    # ============================
    # Sound logic (only on state change)
    # ============================
    if recognized_this_frame:
        unknown_latched = False
        if not recognized_latched:
            play_sound(KNOWN_SOUND_PATH)
            recognized_latched = True
    else:
        recognized_latched = False

    if unknown_this_frame:
        if not unknown_latched:
            play_sound(UNKNOWN_SOUND_PATH)
            unknown_latched = True
    else:
        unknown_latched = False

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
