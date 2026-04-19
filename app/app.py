# app/app.py
import os
import uuid
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from flask import session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from app.db import init_db
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from flask import send_file

from src.transfer_learning.predict import predict_image
from src.transfer_learning.xai import generate_heatmap

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)

app.secret_key = "supersecretkey"

init_db()

UPLOAD_FOLDER = "app/static/uploads"
OUTPUT_FOLDER = "app/static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])

        try:
            conn = sqlite3.connect("app/database.db")
            cursor = conn.cursor()

            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            conn.close()

            return redirect(url_for("login"))

        except:
            return render_template("register.html", error="Username already exists")

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("app/database.db")
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session["user_id"] = user[0]
            session["username"] = user[1]
            return redirect(url_for("home"))
        else:
            return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = sqlite3.connect("app/database.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT image_path, result, confidence, created_at
    FROM predictions
    WHERE user_id = ?
    ORDER BY created_at DESC
    """, (session["user_id"],))

    data = cursor.fetchall()
    conn.close()

    return render_template("dashboard.html", data=data)

@app.route("/download_report")
def download_report():

    from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from datetime import datetime

    image_path = request.args.get("image")
    result = request.args.get("result")
    confidence = float(request.args.get("confidence")) * 100
    date = request.args.get("date")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    pdf_filename = f"report_{uuid.uuid4()}.pdf"
    pdf_path = os.path.join(BASE_DIR, "static", pdf_filename)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(pdf_path)

    content = []

    # 🏥 HEADER
    content.append(Paragraph("🏥 AI Medical Imaging Center", styles['Title']))
    content.append(Spacer(1, 10))

    content.append(Paragraph("Brain Tumor Diagnosis Report", styles['Heading2']))
    content.append(Spacer(1, 15))

    # 📅 DETAILS
    content.append(Paragraph(f"<b>Date:</b> {date}", styles['Normal']))
    
    if "username" in session:
        content.append(Paragraph(f"<b>Patient:</b> {session['username']}", styles['Normal']))
    else:
        content.append(Paragraph("<b>Patient:</b> Guest User", styles['Normal']))

    content.append(Spacer(1, 10))

    # 📊 RESULT
    content.append(Paragraph(f"<b>Diagnosis:</b> {result}", styles['Normal']))
    content.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles['Normal']))
    content.append(Spacer(1, 15))

    # 🖼 IMAGE PATH FIX
    full_image_path = os.path.join(BASE_DIR, image_path.lstrip("/"))

    # 🔥 Generate heatmap TEMP for PDF
    heatmap_filename = f"pdf_heatmap_{uuid.uuid4()}.jpg"
    heatmap_path = os.path.join(BASE_DIR, "static", heatmap_filename)

    from src.transfer_learning.xai import generate_heatmap
    generate_heatmap(full_image_path, heatmap_path)

    # 🖼 ORIGINAL IMAGE
    if os.path.exists(full_image_path):
        content.append(Paragraph("<b>Original MRI Scan</b>", styles['Heading3']))
        content.append(Spacer(1, 5))
        content.append(RLImage(full_image_path, width=250, height=250))
        content.append(Spacer(1, 15))

    # 🔥 HEATMAP
    if os.path.exists(heatmap_path):
        content.append(Paragraph("<b>Model Attention (Grad-CAM)</b>", styles['Heading3']))
        content.append(Spacer(1, 5))
        content.append(RLImage(heatmap_path, width=250, height=250))

    content.append(Spacer(1, 20))

    # 📌 FOOTER
    content.append(Paragraph(
        "This report is generated using AI-based deep learning models and is intended for assistance purposes only.",
        styles['Italic']
    ))

    doc.build(content)

    return send_file(pdf_path, as_attachment=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")

    if file is None or file.filename == "":
        return render_template("index.html", error="No file selected")

    if not allowed_file(file.filename):
        return render_template("index.html", error="Invalid file type")

    original_filename = secure_filename(file.filename)
    unique_filename = str(uuid.uuid4()) + "_" + original_filename

    upload_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(upload_path)

    try:
        result, confidence = predict_image(upload_path)

        heatmap_filename = "heatmap_" + unique_filename
        heatmap_path = os.path.join(OUTPUT_FOLDER, heatmap_filename)

        generate_heatmap(upload_path, heatmap_path)

    except Exception as e:
        return f"Error processing image: {str(e)}"
    
    if "user_id" in session:
        conn = sqlite3.connect("app/database.db")
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO predictions (user_id, image_path, result, confidence)
        VALUES (?, ?, ?, ?)
        """, (
            session["user_id"],
            f"static/uploads/{unique_filename}",
            result,
            float(confidence)
        ))

        conn.commit()
        conn.close()

    return render_template(
        "result.html",
        result=result,
        confidence=round(confidence * 100, 2),
        image_path=f"static/uploads/{unique_filename}",
        heatmap_path=f"static/outputs/{heatmap_filename}"
    )

if __name__ == "__main__":
    app.run(debug=True)