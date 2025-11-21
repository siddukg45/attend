# recognize.py (Hardcoded version with improved message formatting + time-window enforcement)

import os
import cv2
import pickle
import mysql.connector
import numpy as np
import datetime
import time
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

# timezone support (Python 3.9+)
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

# ---------------- CONFIG ----------------

MYSQL_HOST = "localhost"
MYSQL_USER = "root"
MYSQL_PASSWORD = "siddu@8276"
MYSQL_DB = "final"

EMBED_THRESHOLD = 0.6
COOLDOWN_SEC = 30
IMG_SIZE = (160, 160)

# Allowed attendance window (local Asia/Kolkata)
WORK_START = datetime.time(8, 0, 0)      # 08:00:00 inclusive
WORK_END = datetime.time(18, 30, 0)      # 18:30:00 inclusive
LOCAL_TZ_NAME = "Asia/Kolkata"

# ---- HARD CODED TWILIO CREDS ----
TWILIO_ACCOUNT_SID = "AC87623090c881a78388110fc072677480"
TWILIO_AUTH_TOKEN = "3ed1bf56f8c3b0dd16ecc6045ff2863c"
TWILIO_WHATSAPP_FROM = "whatsapp:+14155238886"

# Initialize Twilio
try:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    print("[INFO] Twilio initialized")
except Exception as e:
    print("[ERROR] Twilio init failed:", e)
    twilio_client = None


# ---------------- DB CONNECTION ----------------
def get_connection():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB
    )


# ---------------- SEND WHATSAPP ----------------
def send_whatsapp_message(to_phone: str, text: str):
    if twilio_client is None:
        print("[WARN] Twilio not ready.")
        return False

    to_phone = f"whatsapp:{to_phone}" if not str(to_phone).startswith("whatsapp:") else to_phone

    try:
        msg = twilio_client.messages.create(
            body=text,
            from_=TWILIO_WHATSAPP_FROM,
            to=to_phone
        )
        print("[INFO] WhatsApp sent:", getattr(msg, "sid", "<no-sid>"))
        return True
    except Exception as e:
        print("[ERROR] WhatsApp sending failed:", e)
        return False


# ---------------- LOAD REGISTERED FACES ----------------
def load_known_faces():
    known = []
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, name, embedding, contact_number FROM employees")
        rows = cur.fetchall()

        for emp_id, name, emb_blob, contact in rows:
            if emb_blob is None:
                continue
            try:
                emb = pickle.loads(emb_blob)
                emb = np.asarray(emb, dtype=np.float32)
            except:
                continue

            known.append({
                "id": emp_id,
                "name": name,
                "embedding": emb,
                "contact": contact
            })

        cur.close()
        conn.close()
    except Exception as e:
        print("[ERROR] load_known_faces:", e)

    return known


# ---------------- MODEL LOAD ----------------
embedder = FaceNet()
haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# ---------------- HELPERS ----------------
def compute_embedding_from_crop(face_crop_bgr):
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return None
    face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, IMG_SIZE)
    emb = embedder.embeddings([face_resized])[0]
    return np.asarray(emb, dtype=np.float32)


def find_best_match(embedding, known_list):
    best = None
    best_score = 0
    for rec in known_list:
        try:
            score = float(cosine_similarity([embedding], [rec["embedding"]])[0][0])
        except:
            continue

        if score > best_score:
            best_score = score
            best = rec
    return best, best_score


def normalize_phone(num):
    if not num:
        return None
    num = str(num).strip()
    digits = "".join(c for c in num if c.isdigit() or c == '+')
    if digits.startswith("+"):
        return digits
    if len(digits) == 10:
        return f"+91{digits}"
    return digits


# ---------------- ATTENDANCE ROW ----------------
def ensure_attendance_row(emp_id, date):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT id, in1, out1, in2, out2 FROM attendance WHERE emp_id=%s AND date=%s",
                (emp_id, date))
    row = cur.fetchone()

    if row is None:
        cur.execute("INSERT INTO attendance (emp_id, date) VALUES (%s, %s)", (emp_id, date))
        conn.commit()
        cur.execute("SELECT id, in1, out1, in2, out2 FROM attendance WHERE emp_id=%s AND date=%s",
                    (emp_id, date))
        row = cur.fetchone()

    cur.close()
    conn.close()
    return row


# ---------------- UPDATE FIELD (SAFE) ----------------
ALLOWED_FIELDS = {"in1", "out1", "in2", "out2"}


def update_field(att_id, field_name, time_str):
    if field_name not in ALLOWED_FIELDS:
        print("[ERROR] Invalid field:", field_name)
        return

    conn = get_connection()
    cur = conn.cursor()
    query = f"UPDATE attendance SET {field_name}=%s WHERE id=%s"
    cur.execute(query, (time_str, att_id))
    conn.commit()
    cur.close()
    conn.close()


# ---------------- TIME / WINDOW UTILITIES ----------------
def now_local():
    """
    Return timezone-aware datetime in Asia/Kolkata if zoneinfo available,
    otherwise return naive local datetime (best-effort).
    """
    if ZoneInfo is not None:
        try:
            return datetime.datetime.now(ZoneInfo(LOCAL_TZ_NAME))
        except Exception:
            # fallback to naive local time
            return datetime.datetime.now()
    else:
        return datetime.datetime.now()


def is_within_attendance_window(dt_local):
    """
    dt_local: datetime.datetime (ideally timezone-aware in LOCAL_TZ_NAME)
    Check if dt_local.time() is between WORK_START and WORK_END inclusive.
    """
    t = dt_local.time()
    return (t >= WORK_START) and (t <= WORK_END)


# ---------------- MARK ATTENDANCE & NOTIFY (updated formatting) ----------------

def notify_student(emp_id, event_label, time_str):
    """
    Fetch contact and name, then send a nicely formatted WhatsApp message.
    event_label: one of "IN1","OUT1","IN2","OUT2"
    """
    conn = None
    cur = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT name, contact_number FROM employees WHERE id=%s LIMIT 1", (emp_id,))
        row = cur.fetchone()
        if not row:
            print(f"[WARN] No employee row for {emp_id}")
            return False
        emp_name = row[0] if row[0] else emp_id
        contact_raw = row[1]
        if not contact_raw:
            print(f"[INFO] No contact number for {emp_id}, skipping notification.")
            return False
        phone = normalize_phone(contact_raw)
        if not phone:
            print(f"[WARN] Could not normalize contact '{contact_raw}' for {emp_id}")
            return False

        # Determine session (morning/afternoon) and whether it's check-in or check-out
        if event_label in ("IN1", "OUT1"):
            session = "morning"
        else:
            session = "afternoon"

        action = "checked in" if event_label.startswith("IN") else "checked out"

        # Use local date for message for clarity
        now_dt = now_local()
        date_str = now_dt.date().isoformat()
        # Final polished message
        text = (f"Dear {emp_name},\n\n"
                f"You have successfully {action} for the {session} session at {time_str} on {date_str}.")

        # send
        sent = send_whatsapp_message(phone, text)
        print(f"[INFO] notify_student -> emp={emp_id} to={phone} sent={sent}")
        return sent
    except Exception as e:
        print(f"[ERROR] notify_student exception for {emp_id}: {e}")
        return False
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def mark_attendance(emp_id):
    """
    Update attendance fields and call notify_student with the event label so notify_student
    decides the proper message text (morning/afternoon; check-in/check-out).
    Uses timezone-aware local time for stamps.
    """
    today = now_local().date()
    now = now_local()
    t = now.time()
    ts = now.strftime("%H:%M:%S")

    row = ensure_attendance_row(emp_id, today)
    if not row or row[0] is None:
        print(f"[ERROR] Could not ensure attendance row for {emp_id}.")
        return
    att_id, in1, out1, in2, out2 = row

    # Keep the same logic you had: morning slot until 13:45 then afternoon
    if t < datetime.time(13, 45):
        # morning slot
        if in1 is None:
            update_field(att_id, "in1", ts)
            print(f"[IN1] {emp_id} at {ts}")
            notify_student(emp_id, "IN1", ts)
            return
        else:
            update_field(att_id, "out1", ts)
            print(f"[OUT1] {emp_id} updated {ts}")
            notify_student(emp_id, "OUT1", ts)
            return
    else:
        # afternoon slot
        if in2 is None:
            update_field(att_id, "in2", ts)
            print(f"[IN2] {emp_id} at {ts}")
            notify_student(emp_id, "IN2", ts)
            return
        else:
            update_field(att_id, "out2", ts)
            print(f"[OUT2] {emp_id} updated {ts}")
            notify_student(emp_id, "OUT2", ts)
            return


# ---------------- MAIN LOOP ----------------
def main():
    known_faces = load_known_faces()
    print(f"[INFO] Loaded {len(known_faces)} faces.")

    last_seen = {}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera not found.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            # clamp coordinates just in case face near edge
            h_frame, w_frame = frame.shape[:2]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w_frame, x + w)
            y2 = min(h_frame, y + h)
            if x2 <= x1 or y2 <= y1:
                continue

            face_crop = frame[y1:y2, x1:x2]
            emb = compute_embedding_from_crop(face_crop)
            if emb is None:
                continue

            rec, score = find_best_match(emb, known_faces)
            if rec and score >= EMBED_THRESHOLD:
                emp_id = rec["id"]

                now_ts = time.time()
                # Check cooldown
                if now_ts - last_seen.get(emp_id, 0) > COOLDOWN_SEC:
                    # Check attendance window BEFORE recording
                    now_dt = now_local()
                    if is_within_attendance_window(now_dt):
                        # within allowed window -> record attendance
                        mark_attendance(emp_id)
                        last_seen[emp_id] = now_ts
                        print(f"[INFO] Recorded attendance for {emp_id} at {now_dt.time().strftime('%H:%M:%S')}")
                    else:
                        # Outside window: do not record. Optionally log or notify admin.
                        print(f"[INFO] Recognition for {emp_id} at {now_dt.time().strftime('%H:%M:%S')} - outside attendance window. Skipping DB write.")
                        # Optional: send notification to admin or store attempt in audit table
                        # Example (commented):
                        # notify_admin_of_outside_attempt(emp_id, now_dt)
                        last_seen[emp_id] = now_ts  # still set cooldown to avoid repeated spam

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{rec['name']} {score:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(frame, "Unknown", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
