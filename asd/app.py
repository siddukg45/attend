# app.py
import os
import pickle
import random
import numpy as np
import mysql.connector
from flask import Flask, render_template, request, jsonify, url_for, session, redirect, send_file, abort
from datetime import datetime, date, timedelta, time as dtime
import io
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash

# --- For face capture / embeddings ---
import cv2
import cv2.data
from keras_facenet import FaceNet

# ---------------- CONFIG ----------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "replace_this_with_strong_secret")

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "siddu@8276",
    "database": "final"
}

def get_db_conn():
    return mysql.connector.connect(**DB_CONFIG)

# ---------------- ensure tables exist ----------------
def ensure_tables():
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        id VARCHAR(50) PRIMARY KEY,
        name VARCHAR(100),
        embedding LONGBLOB,
        password_hash VARCHAR(255),
        contact_number VARCHAR(20)
    );
    """)
    conn.commit()
    cur.close()
    conn.close()

    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INT AUTO_INCREMENT PRIMARY KEY,
        emp_id VARCHAR(50),
        date DATE,
        in1 TIME,
        out1 TIME,
        in2 TIME,
        out2 TIME,
        FOREIGN KEY (emp_id) REFERENCES employees(id)
    );
    """)
    conn.commit()
    cur.close()
    conn.close()

ensure_tables()

# ---------------- utilities ----------------

import datetime as _dt

def time_to_str_safe(t):
    if t is None:
        return None
    if isinstance(t, str):
        return t
    if isinstance(t, _dt.time):
        try:
            return t.strftime("%H:%M:%S")
        except:
            return str(t)
    if isinstance(t, _dt.timedelta):
        total_seconds = int(t.total_seconds())
        hh = total_seconds // 3600
        mm = (total_seconds % 3600) // 60
        ss = total_seconds % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    return str(t)

def parse_time_str(t):
    if t is None:
        return None
    if isinstance(t, _dt.time):
        return t
    if isinstance(t, str):
        try:
            parts = [int(x) for x in t.split(":")]
            if len(parts) == 3:
                return _dt.time(parts[0], parts[1], parts[2])
            if len(parts) == 2:
                return _dt.time(parts[0], parts[1], 0)
        except:
            return None
    return None

def seconds_between(start_t, end_t):
    if not start_t or not end_t:
        return 0
    today = _dt.date.today()
    try:
        s_dt = _dt.datetime.combine(today, start_t)
        e_dt = _dt.datetime.combine(today, end_t)
        delta = (e_dt - s_dt).total_seconds()
        return int(delta) if delta > 0 else 0
    except:
        return 0

# ---------------- FACE MODEL ----------------
facenet_model = FaceNet()
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def face_distance(a, b):
    a = a.astype("float32")
    b = b.astype("float32")
    return float(np.linalg.norm(a - b))

def capture_face_embedding(num_images=50):
    if face_cascade.empty():
        return False, "Face detection model not loaded."

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return False, "Unable to access camera."

    collected = []
    try:
        while len(collected) < num_images:
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces):
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                pad = 10
                x1 = max(x - pad, 0)
                y1 = max(y - pad, 0)
                x2 = min(x + w + pad, frame.shape[1])
                y2 = min(y + h + pad, frame.shape[0])
                face_img = frame[y1:y2, x1:x2]
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_rgb = cv2.resize(face_rgb, (160, 160))
                emb = facenet_model.embeddings([face_rgb])[0]
                collected.append(emb)

            cv2.imshow("Face Capture - Press q to cancel", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return False, "Face capture cancelled."

        cap.release()
        cv2.destroyAllWindows()
        return True, np.mean(collected, axis=0)

    except Exception as e:
        cap.release()
        cv2.destroyAllWindows()
        return False, f"Error: {e}"

# ---------------- OTP / SMS helpers ----------------
# ---------------- OTP / SMS helpers ----------------
from twilio.rest import Client
import random

TWILIO_SID = "AC6f8a011f27b34f08f88a00f380d9c54d"
TWILIO_AUTH_TOKEN = "5354c85f6ad595a17692ace32be21bc5"

# Your Twilio SMS number from Console
TWILIO_SMS_FROM = "+13135133191"   # example, replace with yours

def send_otp_sms(phone_number, otp):
    client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
    message = client.messages.create(
        body=f"Your OTP is {otp}. It is valid for 5 minutes.",
        from_=TWILIO_SMS_FROM,            # SMS number, not WhatsApp
        to="+91" + phone_number[-10:]     # formats number correctly
    )
    return message.sid

def generate_otp(length: int = 6):
    return "".join(random.choices("0123456789", k=length))

# ---------------- REGISTER PAGE ----------------
@app.route("/", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    try:
        emp_id = request.form.get("emp_id", "").strip()
        emp_name = request.form.get("emp_name", "").strip()
        password = request.form.get("password", "").strip()
        confirm = request.form.get("confirm", "").strip()
        contact = request.form.get("contact", "").strip()

        # validations
        if not emp_id or not emp_name or not password or not confirm or not contact:
            return jsonify({"error": "All fields are required."})

        if password != confirm:
            return jsonify({"error": "Passwords do not match."})

        if len(password) < 6:
            return jsonify({"error": "Password must be at least 6 characters."})

        import re
        if not re.match(r"^CS\d{3}$", emp_id):
            return jsonify({"error": "Employee ID format must be like CS001."})

        if not re.match(r"^[A-Za-z ]+$", emp_name):
            return jsonify({"error": "Name must contain only letters."})

        if not re.match(r"^[6-9]\d{9}$", contact):
            return jsonify({"error": "Invalid 10-digit mobile number."})

        normalized_contact = "+91" + contact

        conn = get_db_conn()
        cur = conn.cursor()

        cur.execute("SELECT id FROM employees WHERE id=%s", (emp_id,))
        if cur.fetchone():
            cur.close()
            conn.close()
            return jsonify({"error": "Employee ID already registered."})

        ok, result = capture_face_embedding(num_images=50)
        if not ok:
            cur.close()
            conn.close()
            return jsonify({"error": result})

        new_emb = result

        cur.execute("SELECT id, name, embedding FROM employees WHERE embedding IS NOT NULL")
        for other_id, other_name, emb_blob in cur.fetchall():
            try:
                existing_emb = pickle.loads(emb_blob)
                if face_distance(new_emb, existing_emb) <= 0.9:
                    cur.close()
                    conn.close()
                    return jsonify({
                        "duplicate": True,
                        "error": f"Face already registered for {other_name} ({other_id})."
                    })
            except:
                continue

        password_hash = generate_password_hash(password)
        emb_blob = pickle.dumps(new_emb.astype(np.float32))

        cur.execute(
            "INSERT INTO employees (id, name, embedding, password_hash, contact_number) "
            "VALUES (%s, %s, %s, %s, %s)",
            (emp_id, emp_name, emb_blob, password_hash, normalized_contact)
        )
        conn.commit()
        cur.close()
        conn.close()

        # ---------------- SUCCESS PAGE FIXED ----------------
        return render_template(
            "success.html",
            emp_id=emp_id,
            emp_name=emp_name,
            count=50
        )

    except Exception as e:
        print("Registration error:", e)
        return jsonify({"error": "Server error. Please try again."}), 500

# ---------------- Authentication ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        emp_id = request.form.get("emp_id", "").strip()
        password = request.form.get("password", "").strip()

        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("SELECT name, password_hash FROM employees WHERE id=%s", (emp_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            return render_template("login.html", error="Employee ID not found.")

        emp_name, password_hash = row
        if not check_password_hash(password_hash, password):
            return render_template("login.html", error="Incorrect password.")

        session["emp_id"] = emp_id
        session["emp_name"] = emp_name
        return redirect(url_for("dashboard"))

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    if "emp_id" not in session:
        return redirect(url_for("login"))
    return render_template(
        "dashboard.html",
        emp_id=session["emp_id"],
        emp_name=session["emp_name"],
        current_date=date.today().isoformat()
    )

# ---------------- FORGOT PASSWORD FLOW ----------------

@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    """
    Step 1: Ask for emp_id + mobile number
    Step 2: Verify with DB, generate OTP, send SMS, show OTP form
    """
    if request.method == "GET":
        return render_template("forgot_password.html")

    emp_id = request.form.get("emp_id", "").strip()
    contact = request.form.get("contact", "").strip()

    if not emp_id or not contact:
        return render_template(
            "forgot_password.html",
            error="Employee ID and mobile number are required.",
            emp_id=emp_id,
            contact=contact
        )

    # Normalize contact: allow user to enter 10-digit or +91...
    digits_only = "".join(ch for ch in contact if ch.isdigit())
    if len(digits_only) == 10:
        normalized_contact = "+91" + digits_only
    elif contact.startswith("+") and len(digits_only) >= 10:
        normalized_contact = "+" + digits_only
    else:
        return render_template(
            "forgot_password.html",
            error="Please enter a valid 10-digit mobile number.",
            emp_id=emp_id,
            contact=contact
        )

    # Check ID + contact in DB
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT contact_number FROM employees WHERE id=%s", (emp_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        return render_template(
            "forgot_password.html",
            error="Employee ID not found.",
            emp_id=emp_id
        )

    db_contact = row[0]  # stored with +91

    if db_contact != normalized_contact:
        return render_template(
            "forgot_password.html",
            error="Mobile number does not match our records.",
            emp_id=emp_id
        )

    # Everything OK → generate OTP
    otp = generate_otp(6)
    expires_at = _dt.datetime.utcnow() + _dt.timedelta(minutes=5)

    session["fp_emp_id"] = emp_id
    session["fp_otp"] = otp
    session["fp_expires"] = expires_at.isoformat()

    # Send SMS (right now prints to console)
    send_otp_sms(db_contact, otp)
    # Mask contact for UI (e.g. +91******1234)
    last4 = db_contact[-4:]
    masked = f"+91******{last4}"

    return render_template(
        "verify_otp.html",
        emp_id=emp_id,
        masked_contact=masked
    )

@app.route("/verify_otp", methods=["POST"])
def verify_otp():
    """
    Step 3: Verify OTP. If correct, show reset password form.
    """
    emp_id = request.form.get("emp_id", "").strip()
    otp_input = request.form.get("otp", "").strip()

    if not otp_input:
        return render_template(
            "verify_otp.html",
            emp_id=emp_id,
            error="Please enter the OTP sent to your mobile."
        )

    session_emp = session.get("fp_emp_id")
    session_otp = session.get("fp_otp")
    exp_str = session.get("fp_expires")

    if not session_emp or not session_otp or not exp_str:
        return render_template(
            "forgot_password.html",
            error="OTP session expired. Please try again."
        )

    # Check expiry
    try:
        exp = _dt.datetime.fromisoformat(exp_str)
    except Exception:
        exp = None

    if exp and _dt.datetime.utcnow() > exp:
        session.pop("fp_emp_id", None)
        session.pop("fp_otp", None)
        session.pop("fp_expires", None)
        return render_template(
            "forgot_password.html",
            error="OTP expired. Please request a new one."
        )

    if emp_id != session_emp:
        return render_template(
            "verify_otp.html",
            emp_id=emp_id,
            error="Invalid Employee ID."
        )

    if otp_input != session_otp:
        return render_template(
            "verify_otp.html",
            emp_id=emp_id,
            error="Invalid OTP. Please try again."
        )

    # OTP correct → allow password reset
    session["allow_reset_for"] = session_emp
    session.pop("fp_otp", None)
    session.pop("fp_expires", None)

    return render_template("reset_password.html", emp_id=session_emp)

@app.route("/reset_password", methods=["POST"])
def reset_password():
    """
    Step 4: Update password after successful OTP verification.
    """
    emp_id = session.get("allow_reset_for")
    if not emp_id:
        # Direct access without OTP
        return redirect(url_for("forgot_password"))

    password = request.form.get("password", "").strip()
    confirm = request.form.get("confirm", "").strip()

    if not password or not confirm:
        return render_template(
            "reset_password.html",
            emp_id=emp_id,
            error="Please enter password in both fields."
        )

    if password != confirm:
        return render_template(
            "reset_password.html",
            emp_id=emp_id,
            error="Passwords do not match."
        )

    if len(password) < 6:
        return render_template(
            "reset_password.html",
            emp_id=emp_id,
            error="Password must be at least 6 characters."
        )

    password_hash = generate_password_hash(password)

    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE employees SET password_hash=%s WHERE id=%s",
        (password_hash, emp_id)
    )
    conn.commit()
    cur.close()
    conn.close()

    session.pop("allow_reset_for", None)

    return render_template(
        "login.html",
        success="Password reset successfully. Please log in."
    )

# ---------------- Attendance Range ----------------
@app.route("/api/attendance_range")
def attendance_range():
    if "emp_id" not in session:
        return jsonify({"rows": []}), 200

    emp_id = session["emp_id"]
    start = request.args.get("start")
    end = request.args.get("end")

    try:
        if not start:
            start_date = date.today() - timedelta(days=6)
        else:
            start_date = datetime.strptime(start, "%Y-%m-%d").date()

        if not end:
            end_date = date.today()
        else:
            end_date = datetime.strptime(end, "%Y-%m-%d").date()

    except:
        return jsonify({"error": "Invalid date format"}), 400

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT date, in1, out1, in2, out2
            FROM attendance
            WHERE emp_id=%s AND date BETWEEN %s AND %s
            ORDER BY date;
        """, (emp_id, start_date, end_date))

        rows = []
        for (d, in1, out1, in2, out2) in cur.fetchall():
            s_in1 = time_to_str_safe(in1)
            s_out1 = time_to_str_safe(out1)
            s_in2 = time_to_str_safe(in2)
            s_out2 = time_to_str_safe(out2)

            t_in1 = parse_time_str(s_in1)
            t_out1 = parse_time_str(s_out1)
            t_in2 = parse_time_str(s_in2)
            t_out2 = parse_time_str(s_out2)

            secs1 = seconds_between(t_in1, t_out1)
            secs2 = seconds_between(t_in2, t_out2)
            total_secs = secs1 + secs2
            total_hours = round(total_secs / 3600, 2)

            rows.append({
                "date": d.isoformat(),
                "in1": s_in1,
                "out1": s_out1,
                "in2": s_in2,
                "out2": s_out2,
                "present_seconds": total_secs,
                "present_hours": total_hours
            })

        cur.close()
        conn.close()
        return jsonify({"rows": rows})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- Download Attendance ----------------
@app.route("/download_attendance")
def download_attendance():
    if "emp_id" not in session:
        return abort(401)
    emp_id = session["emp_id"]

    start = request.args.get("start")
    end = request.args.get("end")

    try:
        if not start:
            start_date = date.today() - timedelta(days=6)
        else:
            start_date = datetime.strptime(start, "%Y-%m-%d").date()

        if not end:
            end_date = date.today()
        else:
            end_date = datetime.strptime(end, "%Y-%m-%d").date()
    except:
        return "Invalid date format", 400

    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT date, in1, out1, in2, out2
        FROM attendance
        WHERE emp_id=%s AND date BETWEEN %s AND %s
        ORDER BY date;
    """, (emp_id, start_date, end_date))

    rows = []
    for (d, in1, out1, in2, out2) in cur.fetchall():
        s_in1 = time_to_str_safe(in1) or ""
        s_out1 = time_to_str_safe(out1) or ""
        s_in2 = time_to_str_safe(in2) or ""
        s_out2 = time_to_str_safe(out2) or ""

        t_in1 = parse_time_str(s_in1)
        t_out1 = parse_time_str(s_out1)
        t_in2 = parse_time_str(s_in2)
        t_out2 = parse_time_str(s_out2)

        secs1 = seconds_between(t_in1, t_out1)
        secs2 = seconds_between(t_in2, t_out2)
        total_hours = round((secs1 + secs2) / 3600, 2)

        rows.append({
            "date": d.isoformat(),
            "in1": s_in1,
            "out1": s_out1,
            "in2": s_in2,
            "out2": s_out2,
            "present_hours": total_hours
        })

    cur.close()
    conn.close()

    df = pd.DataFrame(rows)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="attendance")
    output.seek(0)

    filename = f"attendance_{emp_id}_{start_date}_{end_date}.xlsx"

    try:
        return send_file(output, as_attachment=True,
                         attachment_filename=filename,
                         mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except:
        return send_file(output, as_attachment=True,
                         download_name=filename,
                         mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------- START SERVER ----------------
if __name__ == "__main__":
    port = 5000
    try:
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{port}/login")
    except:
        pass
    app.run(debug=True, port=port)
