
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import joblib
import sqlite3
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = "student_performance_secret"

# Load trained models
MODEL_PATH = "model/student_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found. Run python train_model.py first."
    )

bundle = joblib.load(MODEL_PATH)

models = bundle["models"]
best_model_name = bundle["best_model"]
model_results = bundle["results"]
features = bundle["features"]


# Database connection
def get_db():
    conn = sqlite3.connect("students.db")
    conn.row_factory = sqlite3.Row
    return conn


# Create database table
def create_table():
    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT,
            study_hours REAL,
            attendance REAL,
            previous_marks REAL,
            assignment_score REAL,
            sleep_hours REAL,
            predicted_marks REAL,
            performance TEXT,
            model_used TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


create_table()


# Home dashboard
@app.route("/")
def home():

    conn = get_db()

    total_predictions = conn.execute(
        "SELECT COUNT(*) FROM predictions"
    ).fetchone()[0]

    average_prediction = conn.execute(
        "SELECT AVG(predicted_marks) FROM predictions"
    ).fetchone()[0]

    recent = conn.execute("""
        SELECT * FROM predictions
        ORDER BY id DESC LIMIT 5
    """).fetchall()

    conn.close()

    return render_template(
        "index.html",
        total_students=bundle["total_students"],
        average_marks=bundle["average_marks"],
        total_predictions=total_predictions,
        average_prediction=round(average_prediction, 2)
            if average_prediction is not None else 0,
        results=model_results,
        best_model=best_model_name,
        recent=recent
    )


# Student prediction
@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        try:
            student_name = request.form["student_name"].strip()

            if not student_name:
                raise ValueError("Please enter student name.")

            student = {
                "StudyHours": float(request.form["StudyHours"]),
                "Attendance": float(request.form["Attendance"]),
                "PreviousMarks": float(request.form["PreviousMarks"]),
                "AssignmentScore": float(request.form["AssignmentScore"]),
                "SleepHours": float(request.form["SleepHours"])
            }

            # Validate input
            if not 0 <= student["StudyHours"] <= 24:
                raise ValueError("Study hours must be between 0 and 24.")

            if not 0 <= student["Attendance"] <= 100:
                raise ValueError("Attendance must be between 0 and 100.")

            if not 0 <= student["PreviousMarks"] <= 100:
                raise ValueError("Previous marks must be between 0 and 100.")

            if not 0 <= student["AssignmentScore"] <= 100:
                raise ValueError("Assignment score must be between 0 and 100.")

            if not 0 <= student["SleepHours"] <= 24:
                raise ValueError("Sleep hours must be between 0 and 24.")

            # Convert input into DataFrame
            input_data = pd.DataFrame(
                [student],
                columns=features
            )

            # Use best model
            model = models[best_model_name]

            predicted_marks = float(
                model.predict(input_data)[0]
            )

            predicted_marks = max(0, min(100, predicted_marks))
            predicted_marks = round(predicted_marks, 2)

            # Determine performance
            if predicted_marks >= 75:
                performance = "Excellent"
            elif predicted_marks >= 60:
                performance = "Good"
            elif predicted_marks >= 40:
                performance = "Average"
            else:
                performance = "Needs Improvement"

            # Save prediction
            conn = get_db()

            conn.execute("""
                INSERT INTO predictions (
                    student_name,
                    study_hours,
                    attendance,
                    previous_marks,
                    assignment_score,
                    sleep_hours,
                    predicted_marks,
                    performance,
                    model_used,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                student_name,
                student["StudyHours"],
                student["Attendance"],
                student["PreviousMarks"],
                student["AssignmentScore"],
                student["SleepHours"],
                predicted_marks,
                performance,
                best_model_name,
                datetime.now().strftime("%d-%m-%Y %H:%M")
            ))

            conn.commit()
            conn.close()

            return render_template(
                "predict.html",
                result=True,
                student_name=student_name,
                predicted_marks=predicted_marks,
                performance=performance,
                model_name=best_model_name
            )

        except (ValueError, KeyError) as error:
            flash(str(error), "error")

    return render_template("predict.html")


# Prediction reports
@app.route("/reports")
def reports():

    conn = get_db()

    records = conn.execute("""
        SELECT * FROM predictions
        ORDER BY id DESC
    """).fetchall()

    conn.close()

    return render_template(
        "reports.html",
        records=records
    )


if __name__ == "__main__":
    app.run(debug=True)
