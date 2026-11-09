"""Gradio app for a Hugging Face Space.

Before pushing to a Space, copy ../model.joblib and ../heart_features.py next to this file
(see space/README.md). Run locally from the repo root with:  python space/app.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]  # works inside the repo and inside a Space

import gradio as gr
import joblib
import pandas as pd

import heart_features as hf

bundle_path = HERE / "model.joblib" if (HERE / "model.joblib").exists() else HERE.parent / "model.joblib"
bundle = joblib.load(bundle_path)
model, params = bundle["model"], bundle["params"]
scaler = model.named_steps["standardscaler"]
lr = model.named_steps["logisticregression"]
GROUPS = {name: ("ST_Slope" if name.startswith("ST_Slope") else name.split("_")[0]) for name in hf.FEATURE_NAMES}


def predict(age, sex, cp, bp, chol, fbs, ecg, maxhr, angina, oldpeak, slope):
    raw = pd.DataFrame([{
        "Age": age, "Sex": sex, "ChestPainType": cp, "RestingBP": bp, "Cholesterol": chol,
        "FastingBS": int(fbs), "RestingECG": ecg, "MaxHR": maxhr, "ExerciseAngina": angina,
        "Oldpeak": oldpeak, "ST_Slope": slope,
    }])
    X = hf.transform(raw, params)
    p = float(model.predict_proba(X)[0, 1])
    z = (X.to_numpy()[0] - scaler.mean_) / scaler.scale_
    contrib = pd.Series(lr.coef_[0] * z, index=hf.FEATURE_NAMES).groupby(GROUPS).sum()
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(5)
    lines = [f"- **{k}** {'raises' if v > 0 else 'lowers'} the log-odds by {abs(v):.2f}" for k, v in top.items()]
    return {"Heart disease": p, "No heart disease": 1 - p}, "\n".join(lines)


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(18, 100, value=54, step=1, label="Age (years)"),
        gr.Radio(["M", "F"], value="M", label="Sex"),
        gr.Dropdown(["ASY", "NAP", "ATA", "TA"], value="ASY", label="Chest pain type"),
        gr.Slider(80, 200, value=130, step=1, label="Resting blood pressure (mm Hg)"),
        gr.Slider(0, 600, value=223, step=1, label="Serum cholesterol (mg/dl, 0 = not measured)"),
        gr.Checkbox(value=False, label="Fasting blood sugar > 120 mg/dl"),
        gr.Dropdown(["Normal", "ST", "LVH"], value="Normal", label="Resting ECG (not used by the model)"),
        gr.Slider(60, 210, value=138, step=1, label="Max heart rate achieved"),
        gr.Radio(["N", "Y"], value="N", label="Exercise-induced angina"),
        gr.Slider(-3, 6.5, value=0.6, step=0.1, label="Oldpeak (ST depression)"),
        gr.Dropdown(["Up", "Flat", "Down"], value="Flat", label="ST slope"),
    ],
    outputs=[gr.Label(label="Predicted probability"), gr.Markdown(label="Top contributing inputs")],
    title="Heart disease risk: an explainable classifier",
    description="Logistic regression trained on the Kaggle Heart Failure Prediction dataset. "
                "**Educational demo, not medical advice.**",
)

if __name__ == "__main__":
    demo.launch()
