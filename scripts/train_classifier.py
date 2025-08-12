#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--components", required=True, help="CCA_components.csv")
    p.add_argument("--labels", required=True, help="labels_aligned.csv with sample_id,label")
    p.add_argument("--out", required=True)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    args = p.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    X = pd.read_csv(args.components)
    y_df = pd.read_csv(args.labels)
    # assume same order; if not, user should merge by id and reorder
    y = y_df["label"].astype(str).values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X.values, y_enc, test_size=args.test_size, random_state=args.random_state, stratify=y_enc)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rep = classification_report(y_test, y_pred, target_names=list(le.classes_), output_dict=True)

    joblib.dump(clf, outdir / "logreg.joblib")
    joblib.dump(le, outdir / "label_encoder.joblib")
    (outdir / "metrics.json").write_text(json.dumps({"accuracy": acc, "report": rep}, indent=2))

    print("Saved model + metrics to", outdir)

if __name__ == "__main__":
    main()
