import werkzeug.exceptions
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)


@app.route("/")
def home_page():
    return render_template("index.html")


@app.route("/heart-disease-ml-project")
def machine_learning_project():
    return render_template("heart_disease_project.html")


@app.route("/math-app-project")
def math_app_project():
    return render_template("math_app_project.html")


@app.route("/results", methods=["GET", "POST"])
def show_results():
    age_category = request.form["age_category"]
    try:
        difficulty_walking = request.form["difficulty_walking"]
    except werkzeug.exceptions.BadRequestKeyError:
        difficulty_walking = "off"
    try:
        diabetic = request.form["diabetic"]
    except werkzeug.exceptions.BadRequestKeyError:
        diabetic = "off"
    try:
        stroke = request.form["stroke"]
    except werkzeug.exceptions.BadRequestKeyError:
        stroke = "off"
    general_health = request.form["general_health"]

    at_risk_of_heart_disease = predict_heart_disease_risk(age_category, difficulty_walking, diabetic, stroke,
                                                          general_health)

    return render_template("heart_disease_project_results.html", at_risk_of_heart_disease=at_risk_of_heart_disease)


ml_model = joblib.load("./ml_model/model.joblib")

# These variables are used to convert the string values that we will receive from the user into numbers for the ml model
age_categories = [
    '18-24',
    '25-29',
    '30-34',
    '35-39',
    '40-44',
    '45-49',
    '50-54',
    '55-59',
    '60-64',
    '65-69',
    '70-74',
    '75-79',
    '80 or older'
]
len_age_categories = len(age_categories)
# The list below will end up having values from 0 to 1, with the same distance between each other [0.0, 0.083, 0.166...]
age_categories_as_numbers = [(1 / (len_age_categories - 1)) * idx for idx in range(len_age_categories)]

age_categories_to_numbers = {key: value for key, value in zip(age_categories, age_categories_as_numbers)}

general_health_markers_to_numbers = {
    "Poor": 0,
    "Fair": 0.25,
    "Good": 0.5,
    "Very good": 0.75,
    "Excellent": 1
}


def predict_heart_disease_risk(age_category, difficulty_walking, diabetic, stroke, general_health):
    age_category = age_categories_to_numbers[age_category]

    if difficulty_walking == "on":
        difficulty_walking = 1
    else:
        difficulty_walking = 0

    if diabetic == "on":
        diabetic = 1
    else:
        diabetic = 0

    if stroke == "on":
        stroke = 1
    else:
        stroke = 0

    general_health = general_health_markers_to_numbers[general_health]

    model_input = pd.DataFrame({
        "AgeCategory": [age_category],
        "DiffWalking": [difficulty_walking],
        "Diabetic": [diabetic],
        "Stroke": [stroke],
        "GenHealth": [general_health]
    })

    prediction = ml_model.predict(model_input)[0]

    return prediction


if __name__ == "__main__":
    app.run()
