---
title: Heart Disease Risk (educational)
emoji: 🫀
colorFrom: red
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Heart disease risk: Gradio Space

The same logistic-regression model as the [static demo](https://nardy11.github.io/Heart_Failure_prediction/), served with Gradio.

To publish it as a Hugging Face Space:

```bash
python train.py                      # writes model.joblib
cp model.joblib heart_features.py space/
cd space && git init && git remote add origin https://huggingface.co/spaces/<user>/heart-disease-risk
git add . && git commit -m "Initial Space" && git push origin main
```

Educational demo, not medical advice. See the model card in the main README.
