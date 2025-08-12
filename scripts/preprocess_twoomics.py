#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expr", required=True)
    p.add_argument("--meth", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--id_col", default="sample_id")
    p.add_argument("--label_col", default="label")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    expr = pd.read_csv(args.expr)
    meth = pd.read_csv(args.meth)
    lab  = pd.read_csv(args.labels)

    if args.id_col not in expr.columns or args.id_col not in meth.columns or args.id_col not in lab.columns:
        raise ValueError("sample_id column must exist in all input files")

    # Align by intersection of samples
    ids = set(expr[args.id_col]).intersection(set(meth[args.id_col])).intersection(set(lab[args.id_col]))
    expr = expr[expr[args.id_col].isin(ids)].copy()
    meth = meth[meth[args.id_col].isin(ids)].copy()
    lab  = lab[lab[args.id_col].isin(ids)].copy()

    # Sort to same order
    expr = expr.sort_values(args.id_col).reset_index(drop=True)
    meth = meth.sort_values(args.id_col).reset_index(drop=True)
    lab = lab.sort_values(args.id_col).reset_index(drop=True)

    # Extract feature matrices
    X1 = expr.drop(columns=[args.id_col])
    X2 = meth.drop(columns=[args.id_col])

    # Transform (log1p for expression only)
    X1 = np.log1p(X1.astype(float))
    X2 = X2.astype(float)

    # Scale
    s1 = StandardScaler()
    s2 = StandardScaler()
    X1s = s1.fit_transform(X1)
    X2s = s2.fit_transform(X2)

    # Save processed
    pd.DataFrame(X1s, columns=[f"g{i}" for i in range(X1s.shape[1])]).to_csv(out / "X_expr.csv", index=False)
    pd.DataFrame(X2s, columns=[f"cpg{i}" for i in range(X2s.shape[1])]).to_csv(out / "X_meth.csv", index=False)
    lab[[args.id_col, args.label_col]].to_csv(out / "labels_aligned.csv", index=False)

    print("Processed and saved to", out)

if __name__ == "__main__":
    main()
