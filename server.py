from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def home_page():
    return render_template("index.html")


@app.route("/heart-disease-ml-project")
def machine_learning_project():
    return render_template("heart_disease_project.html")


if __name__ == "__main__":
    app.run(debug=True)
