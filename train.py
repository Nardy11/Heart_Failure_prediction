"""Reproduce the notebook's models with a fixed seed.

    python train.py              # cleaning parameters fitted on the training split only
    python train.py --notebook   # parameters fitted on all rows, exactly as the notebook did

Writes model.joblib (selected model), outputs/metrics.json, outputs/*.png and
docs/model.json (the Logistic Regression the browser demo runs).
"""
from __future__ import annotations

import argparse
import warnings
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import heart_features as hf

SEED = 42
warnings.filterwarnings("ignore", category=FutureWarning)  # SVC(probability=True) deprecation in sklearn 1.9
ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
WEB = "Logistic Regression (C=0.01, grid-search best)"  # C picked by 5-fold CV; saved and exported
SELECTED = WEB
NOTEBOOK_PICK = "Random Forest (tuned) + PCA"  # what the notebook selected (by test-set score)


def lr(c):
    return LogisticRegression(C=c, solver="lbfgs", random_state=SEED, max_iter=1000)


def rf():
    return RandomForestClassifier(n_estimators=200, criterion="gini", max_depth=10, random_state=SEED)


def svc():
    return SVC(kernel="linear", C=10, probability=True, random_state=SEED)


def pca():
    return PCA(n_components=0.95, random_state=SEED)


def models():
    return {
        WEB: make_pipeline(StandardScaler(), lr(0.01)),
        "Logistic Regression (C=10, notebook final table)": make_pipeline(StandardScaler(), lr(10)),
        "Random Forest (tuned)": make_pipeline(StandardScaler(), rf()),
        "Linear SVC (tuned)": make_pipeline(StandardScaler(), svc()),
        "Logistic Regression (C=10) + PCA": make_pipeline(StandardScaler(), pca(), lr(10)),
        NOTEBOOK_PICK: make_pipeline(StandardScaler(), pca(), rf()),
        "Linear SVC (tuned) + PCA": make_pipeline(StandardScaler(), pca(), svc()),
    }


def score(model, X, y):
    p = model.predict_proba(X)[:, 1]
    yhat = model.predict(X)
    return {
        "accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat),
        "recall": recall_score(y, yhat), "f1": f1_score(y, yhat), "roc_auc": roc_auc_score(y, p),
    }


def style(ax, title):
    ax.set_title(title, fontsize=12, loc="left", fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plots(fitted, results, X_tr, y_tr, X_te, y_te):
    plt.rcParams.update({"axes.edgecolor": "#888", "axes.labelcolor": "#333"})
    blue, orange, green, grey = "#2563eb", "#ea580c", "#16a34a", "#9ca3af"
    best = fitted[SELECTED]

    cm = confusion_matrix(y_te, best.predict(X_te))
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "#111")
    ax.set_xticks([0, 1], ["No disease", "Disease"])
    ax.set_yticks([0, 1], ["No disease", "Disease"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    style(ax, f"Confusion matrix (test set, n={len(y_te)})\nLogistic Regression, C=0.01")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, color in [(WEB, orange), (NOTEBOOK_PICK, blue), ("Linear SVC (tuned)", green)]:
        fpr, tpr, _ = roc_curve(y_te, fitted[name].predict_proba(X_te)[:, 1])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC {results[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color=grey, lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    style(ax, "ROC curve (test set)")
    fig.tight_layout()
    fig.savefig(OUT / "roc_curve.png", dpi=150)
    plt.close(fig)

    rf_plain = fitted["Random Forest (tuned)"].named_steps["randomforestclassifier"]
    imp = pd.Series(rf_plain.feature_importances_, index=hf.FEATURE_NAMES).sort_values().tail(12)
    coef = pd.Series(fitted[WEB].named_steps["logisticregression"].coef_[0], index=hf.FEATURE_NAMES)
    coef = coef.reindex(coef.abs().sort_values().tail(12).index)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5))
    a1.barh(imp.index, imp.values, color=blue)
    a1.set_xlabel("Mean decrease in impurity")
    style(a1, "Random Forest: feature importance")
    a2.barh(coef.index, coef.values, color=[orange if v > 0 else blue for v in coef.values])
    a2.axvline(0, color=grey, lw=1)
    a2.set_xlabel("Coefficient on standardised feature (orange raises risk)")
    style(a2, "Logistic Regression: coefficients")
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=150)
    plt.close(fig)

    pca2 = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=SEED)).fit(X_tr)
    Z = pca2.transform(X_tr)
    ev = pca2.named_steps["pca"].explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(6, 5))
    yv = y_tr.to_numpy()
    for cls, color, label in [(0, blue, "No heart disease"), (1, orange, "Heart disease")]:
        ax.scatter(Z[yv == cls, 0], Z[yv == cls, 1], s=14, alpha=0.6, color=color, label=label)
    ax.set_xlabel(f"PC1 ({ev[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({ev[1]:.0%} of variance)")
    ax.legend(frameon=False)
    style(ax, "PCA of the training set (2 components)")
    fig.tight_layout()
    fig.savefig(OUT / "pca_scatter.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", action="store_true",
                    help="fit cleaning params on all rows, as the notebook did (prints metrics only)")
    args = ap.parse_args()

    df = pd.read_csv(ROOT / "heart.csv")
    y = df[hf.TARGET]
    raw_tr, raw_te, y_tr, y_te = train_test_split(df[hf.RAW_FEATURES], y, test_size=0.2,
                                                  random_state=SEED, stratify=y)
    params = hf.fit_params(df if args.notebook else raw_tr)
    X_tr, X_te = hf.transform(raw_tr, params), hf.transform(raw_te, params)

    results, fitted = {}, {}
    for name, m in models().items():
        cv = cross_val_score(m, X_tr, y_tr, cv=5, scoring="f1")
        m.fit(X_tr, y_tr)
        results[name] = {**score(m, X_te, y_te), "cv5_f1_train": float(cv.mean())}
        fitted[name] = m
        r = results[name]
        print(f"{name:50s} acc {r['accuracy']:.4f} prec {r['precision']:.4f} rec {r['recall']:.4f} "
              f"f1 {r['f1']:.4f} auc {r['roc_auc']:.4f} | train cv5 f1 {r['cv5_f1_train']:.4f}")

    if args.notebook:
        return

    best = fitted[SELECTED]
    joblib.dump({"model": best, "params": params, "features": hf.FEATURE_NAMES}, ROOT / "model.joblib")

    OUT.mkdir(exist_ok=True)
    meta = {"notebook_pick": NOTEBOOK_PICK, "seed": SEED, "train_size": len(y_tr), "test_size": len(y_te), "selected_model": SELECTED,
            "pca_components": int(fitted[NOTEBOOK_PICK].named_steps["pca"].n_components_), "results": results}
    (OUT / "metrics.json").write_text(json.dumps(meta, indent=2))

    web = fitted[WEB]
    sc, clf = web.named_steps["standardscaler"], web.named_steps["logisticregression"]
    export = {
        "model": WEB, "features": hf.FEATURE_NAMES,
        "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
        "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]), "params": params,
        "test_metrics": results[WEB], "rf_pca_test_metrics": results[NOTEBOOK_PICK],
    }
    (ROOT / "docs" / "model.json").write_text(json.dumps(export, indent=1))

    plots(fitted, results, X_tr, y_tr, X_te, y_te)

    sample = raw_te.head(20).copy()
    sample["p_python"] = web.predict_proba(hf.transform(sample, params))[:, 1]
    (ROOT / "tests" / "parity_rows.json").write_text(sample.to_json(orient="records", indent=1))
    print("saved model.joblib, outputs/, docs/model.json, tests/parity_rows.json")


if __name__ == "__main__":
    main()
