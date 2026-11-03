
import pandas as pd
import numpy as np

np.random.seed(42)

students = 200

study_hours = np.random.uniform(1, 10, students)
attendance = np.random.uniform(50, 100, students)
previous_marks = np.random.uniform(30, 95, students)
assignment_score = np.random.uniform(40, 100, students)
sleep_hours = np.random.uniform(4, 10, students)

final_marks = (
    study_hours * 3.5
    + attendance * 0.20
    + previous_marks * 0.30
    + assignment_score * 0.15
    + sleep_hours * 0.8
    + np.random.normal(0, 5, students)
)

final_marks = np.clip(final_marks, 0, 100)

data = pd.DataFrame({
    "StudyHours": study_hours.round(1),
    "Attendance": attendance.round(1),
    "PreviousMarks": previous_marks.round(1),
    "AssignmentScore": assignment_score.round(1),
    "SleepHours": sleep_hours.round(1),
    "FinalMarks": final_marks.round(1)
})

data.to_csv("Student_Performance.csv", index=False)

print("Student dataset created successfully!")
print(data.head())
