// Runs the exported Logistic Regression (docs/model.json) on one patient.
// Mirrors heart_features.transform() + StandardScaler + LogisticRegression.

const OLDPEAK = [[-0.01, "ST Elevation"], [0.0, "Normal"], [1.0, "Mild Depression"], [2.0, "Moderate Depression"]];
const CHOL = [[200, "Normal"], [240, "Borderline"]];

function bin(value, edges, last) {
  for (const [edge, label] of edges) if (value <= edge) return label;
  return last;
}

/** Raw 11-field patient -> {featureName: value} for the 23 model features. */
export function featurize(raw, params) {
  const clip = (col, v) => Math.min(Math.max(v, params.caps[col][0]), params.caps[col][1]);
  const bp = Number(raw.RestingBP) === 0 ? params.restingbp_median : Number(raw.RestingBP);
  const cat = {
    Sex: raw.Sex,
    ChestPainType: raw.ChestPainType,
    ExerciseAngina: raw.ExerciseAngina,
    ST_Slope: raw.ST_Slope,
    Oldpeak: bin(clip("Oldpeak", Number(raw.Oldpeak)), OLDPEAK, "Severe Depression"),
    Cholesterol: bin(clip("Cholesterol", Number(raw.Cholesterol)), CHOL, "High"),
  };
  const out = {
    Age: Number(raw.Age),
    RestingBP: clip("RestingBP", bp),
    FastingBS: Number(raw.FastingBS),
    MaxHR: clip("MaxHR", Number(raw.MaxHR)),
  };
  for (const [col, value] of Object.entries(cat)) out[`${col}_${value}`] = 1;
  return out;
}

/** Probability of heart disease plus per-input contributions to the log-odds. */
export function predict(raw, model) {
  const x = featurize(raw, model.params);
  let logit = model.intercept;
  const byInput = {};
  model.features.forEach((name, i) => {
    const z = ((x[name] ?? 0) - model.mean[i]) / model.scale[i];
    const c = model.coef[i] * z;
    logit += c;
    // one-hot columns are "<input>_<value>"; ST_Slope itself contains an underscore
    const key = name.startsWith("ST_Slope") ? "ST_Slope" : name.split("_")[0];
    byInput[key] = (byInput[key] ?? 0) + c;
  });
  const probability = 1 / (1 + Math.exp(-logit));
  const contributions = Object.entries(byInput)
    .map(([input, value]) => ({ input, value }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  return { probability, logit, contributions };
}
