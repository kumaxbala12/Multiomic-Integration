#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cross_decomposition import CCA
import joblib

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--X1", required=True, help="processed expression matrix CSV")
    p.add_argument("--X2", required=True, help="processed methylation matrix CSV")
    p.add_argument("--k", type=int, default=20, help="number of CCA components")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X1 = pd.read_csv(args.X1).values
    X2 = pd.read_csv(args.X2).values

    k = min(args.k, X1.shape[1], X2.shape[1])
    cca = CCA(n_components=k, max_iter=1000)
    U, V = cca.fit_transform(X1, X2)  # U: X1 comps, V: X2 comps

    # Concatenate joint representation
    comps = np.hstack([U, V])
    cols = [f"cca_expr_{i+1}" for i in range(k)] + [f"cca_meth_{i+1}" for i in range(k)]
    pd.DataFrame(comps, columns=cols).to_csv(out / "CCA_components.csv", index=False)

    # Save loadings for inspection
    load_expr = pd.DataFrame(cca.x_weights_, columns=[f"cca_expr_{i+1}" for i in range(k)])
    load_meth = pd.DataFrame(cca.y_weights_, columns=[f"cca_meth_{i+1}" for i in range(k)])
    load_expr.to_csv(out / "CCA_loadings_expr.csv", index=False)
    load_meth.to_csv(out / "CCA_loadings_meth.csv", index=False)

    joblib.dump(cca, out / "cca_model.joblib")
    print("Saved CCA components and loadings to", out)

if __name__ == "__main__":
    main()
