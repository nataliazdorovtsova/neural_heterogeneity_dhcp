"""Feature sets for the measure-set checks (communicability replaced or dropped).

    python scripts/extract_alt_features.py

  perbin25_mst_eig   strength, log clustering, eigenvector centrality  (753, 90, 3)
  perbin25_mst_sc    strength, log clustering                           (753, 90, 2)

Strength and clustering are copied unchanged from the primary feature file, so
the only difference from the primary set is the third measure. Eigenvector
centrality is computed on the same primary (per-bin 25% + MST) networks.

Its transform follows the rule used for the other measures (DECISIONS.md D10):
take the log only if it lowers the median |skewness| across regions, judged on
the training infants. Centrality is strictly positive in connected networks
(the MST guarantees connectivity), so the log is defined.
"""

import numpy as np
import pandas as pd
from scipy import stats

import bct
import paths

OUT = paths.RESULTS / "features"


def eigenvector_all(stack):
    A = np.load(stack, mmap_mode="r")
    out = np.empty((A.shape[2], A.shape[0]))
    for i in range(A.shape[2]):
        W = np.array(A[:, :, i], dtype=float)
        np.fill_diagonal(W, 0)
        out[i] = bct.eigenvector_centrality_und(W)
    return out


def main():
    coh = pd.read_csv(paths.RESULTS / "cohort" / "cohort.csv")
    prim = np.load(OUT / "perbin25_mst.npz")
    X = prim["X"]                                             # (753, 90, 3)
    assert list(prim["measures"]) == ["strength", "log_clustering", "communicability"]
    eig = eigenvector_all(paths.STACK)
    assert eig.shape == (len(coh), 90) and np.isfinite(eig).all()
    assert (eig > 0).all(), "eigenvector centrality must be positive to take its log"

    train = (coh.in_primary & (coh.split == "train")).values
    sk_raw = np.median(np.abs(stats.skew(eig[train], axis=0)))
    sk_log = np.median(np.abs(stats.skew(np.log(eig[train]), axis=0)))
    use_log = sk_log < sk_raw
    third = np.log(eig) if use_log else eig
    name3 = "log_eigenvector" if use_log else "eigenvector"
    print(f"eigenvector centrality: median |skew| raw {sk_raw:.2f}, log {sk_log:.2f} -> {name3}")

    np.savez_compressed(OUT / "perbin25_mst_eig.npz",
                        X=np.concatenate([X[:, :, :2], third[:, :, None]], axis=2),
                        measures=np.array(["strength", "log_clustering", name3]),
                        dhcp_id=coh.dhcp_id.values)
    np.savez_compressed(OUT / "perbin25_mst_sc.npz", X=X[:, :, :2],
                        measures=np.array(["strength", "log_clustering"]),
                        dhcp_id=coh.dhcp_id.values)
    print("wrote perbin25_mst_eig.npz (753, 90, 3) and perbin25_mst_sc.npz (753, 90, 2)")


if __name__ == "__main__":
    main()
