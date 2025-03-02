import gradio as gr
import joblib
import numpy as np
import pandas as pd

ALL_COLS = [
    ("age", gr.Slider(0, 130, step=1.0, label="Age")),
    ("sex", gr.Radio(["Male", "Female"], label="Gender")),
    ("cp", gr.Slider(1, 4, step=1.0, label="CP")),
    ("trestbps", gr.Slider(50, 250, step=1.0, label="Resting blood pressure (mm Hg)")),
    ("chol", gr.Slider(100, 600, step=1.0, label="Serum cholestoral (mg/dl)")),
    ("fbs", gr.Checkbox(label="Fasting blood sugar > 120 mg/dl")),
    ("restecg", gr.Slider(0, 2, step=1.0, label="Rest ECG")),
    ("thalach", gr.Slider(50, 250, step=1.0, label="Maximum heart rate achieved")),
    ("exang", gr.Checkbox(label="Exercise induced angina")),
    ("oldpeak", gr.Number(precision=1, minimum=0.0, maximum=7.0, label="ST depression induced by exercise relative to rest")),
    ("slope", gr.Slider(1, 3, step=1.0, label="Slope")),
    ("ca", gr.Slider(0, 3, step=1.0, label="Number of major vessels colored by flourosopy")),
    ("thal", gr.Radio([3.0, 6.0, 7.0], label="THAL")),
]


model = joblib.load("./data/pipelines/complete_binary_pipeline.jdp")

CAT_COLUMNS = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "thal",
]

NUM_COLUMNS = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "ca",
]


def predict(
    age,
    sex,
    cp,
    trestbps,
    chol,
    fbs,
    restecg,
    thalach,
    exang,
    oldpeak,
    slope,
    ca,
    thal,
):
    age = float(age)
    sex = 0.0 if "Male" else 1.0
    cp = float(cp)
    trestbps = float(trestbps)
    chol = float(chol)
    fbs = 1.0 if fbs else 0.0
    restecg = float(restecg)
    thalach = float(thalach)
    exang = 1.0 if exang else 0.0
    oldpeak = float(oldpeak)
    slope = float(slope)
    ca = float(ca)
    thal = float(thal) if thal is not None else np.nan

    features = pd.DataFrame(
        np.array([
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal,
        ]).reshape(1, -1),
        columns=[col[0] for col in ALL_COLS]
    )
    pred = model.predict_proba(features)[0]
    label = pred.argmax()
    positive = label == 1
    score = pred[pred.argmax()]
    if positive:
        return f"The model does predict heart disease for this patient with a probability of {100 * score:.2f}%"
    else:
        return f"The model does not predict heart disease for this patient with a probability of {100 * score:.2f}%"

demo = gr.Interface(
    fn=predict,
    inputs=[val[1] for val in ALL_COLS],
    outputs=["text"],
    title="Heart disease prediction",
)

demo.launch()

