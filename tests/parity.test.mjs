// node --test tests/
// Checks the browser model (docs/predict.js) against scikit-learn's predict_proba
// on 20 held-out rows written by train.py.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { predict } from "../docs/predict.js";

const load = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url), "utf8"));
const model = load("../docs/model.json");
const rows = load("./parity_rows.json");

test("JS and Python probabilities agree on 20 test rows", () => {
  assert.equal(rows.length, 20);
  for (const row of rows) {
    const { probability } = predict(row, model);
    assert.ok(Math.abs(probability - row.p_python) < 1e-9, `${probability} vs ${row.p_python}`);
  }
});

test("contributions sum to the logit", () => {
  const { logit, contributions } = predict(rows[0], model);
  const sum = contributions.reduce((s, c) => s + c.value, model.intercept);
  assert.ok(Math.abs(sum - logit) < 1e-12);
});
