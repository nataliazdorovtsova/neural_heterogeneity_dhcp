"""Stage 2: turn connectome stacks into (n, 90, 3) feature tables.

    python scripts/extract_features.py

Writes results/features/<variant>.npz holding RAW features. Standardisation is
deliberately NOT baked in: it must use training-fold moments only, so it happens
at fit time via features.standardise().
"""

import numpy as np
import pandas as pd
import networkx as nx

import paths
from features import regional_features, clustering_onnela, MEASURES

OUT = paths.RESULTS / "features"
OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = {
    "perbin25_mst":        (paths.STACK, False),           # primary
    "perbin25_mst_binary": (paths.STACK, True),
    "unthresholded":       (paths.STACK_UNTHRESH, False),
    "unthresholded_binary":(paths.STACK_UNTHRESH, True),
}


def verify_clustering(A: np.ndarray, n: int = 5) -> None:
    """Our vectorised Onnela must match networkx before we trust it."""
    for i in range(n):
        W = np.asarray(A[:, :, i], dtype=float)
        ours = clustering_onnela(W)
        theirs = np.array(list(nx.clustering(nx.from_numpy_array(W), weight="weight").values()))
        if not np.allclose(ours, theirs, atol=1e-10):
            raise AssertionError(f"clustering mismatch on session {i}: "
                                 f"max abs diff {np.abs(ours-theirs).max():.3g}")
    print(f"  clustering verified against networkx on {n} sessions")


def build(stack_path, binarise: bool) -> np.ndarray:
    A = np.load(stack_path, mmap_mode="r")
    n = A.shape[2]
    X = np.empty((n, 90, 3))
    for i in range(n):
        W = np.asarray(A[:, :, i], dtype=float)
        if binarise:
            W = (W > 0).astype(float)
        X[i] = regional_features(W)
    return X


def main():
    cohort = pd.read_csv(paths.RESULTS / "cohort" / "cohort.csv")
    print("verifying the clustering implementation")
    verify_clustering(np.load(paths.STACK, mmap_mode="r"))

    for name, (stack, binarise) in VARIANTS.items():
        X = build(stack, binarise)
        assert len(X) == len(cohort), f"{name}: {len(X)} rows vs cohort {len(cohort)}"
        assert np.isfinite(X).all(), f"{name}: non-finite features"

        flat = X.reshape(len(X), -1)
        zero_var = (flat.std(0) == 0).sum()
        from scipy.stats import skew
        sk = skew(flat.reshape(len(X), 90, 3), axis=0)

        np.savez_compressed(OUT / f"{name}.npz", X=X, measures=np.array(MEASURES),
                            dhcp_id=cohort.dhcp_id.values)
        print(f"\n{name}: {X.shape}  -> flattened D = {flat.shape[1]}")
        print(f"  zero-variance coordinates: {zero_var}")
        for j, m in enumerate(MEASURES):
            print(f"  {m:20s} mean {X[:,:,j].mean():11.4g}  sd {X[:,:,j].std():10.4g}"
                  f"  median |skew| {np.median(np.abs(sk[:,j])):.2f}")

    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
