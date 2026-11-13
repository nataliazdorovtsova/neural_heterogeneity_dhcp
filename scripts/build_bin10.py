"""Binarised networks at a fixed density, and lobe features (run bin10, step 1).

    python scripts/build_bin10.py

Each session's unthresholded mu-weighted network is binarised to the same
number of edges for every infant: the maximum-weight spanning tree (89 edges,
so no region is cut off), then the strongest remaining edges. 10% of the 4,005
possible edges is the primary density; 8% and 12% are built for the density
check, and 8-12% in 1% steps for the averaged features (D19). Ties are broken
by edge index, so the result is deterministic.

Two candidate measures are computed per region on the full 90-region network,
then averaged within the 12 lobe x hemisphere groups (atlas.LOBE, atlas.HEMI):
  communicability     mean off-diagonal expm(D^-1/2 A D^-1/2) (features.py)
  between-lobe share  fraction of a region's edges leaving its lobe x hemisphere
                      group; interhemispheric edges count as leaving

Writes results/features/bin10_mst_nodes.npz (node level, all densities),
bin10_mst_{comm,bls}_lobe.npz (n, 12) at 10%, and bin10_comm_avg_lobe.npz:
communicability averaged over 8-12% for each region, then per lobe (the
primary features, D19). Raw values: standardisation happens at fit time.
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.csgraph import minimum_spanning_tree

import paths
import atlas
from features import communicability

OUT = paths.PROJECT / "results" / "features"
N = 90
IU = np.triu_indices(N, 1)                       # the 4,005 possible edges
DENSITIES = {8: 320, 9: 360, 10: 400, 11: 441, 12: 481}   # % -> edges (round(p * 4005))
PRIMARY = 10

GROUPS = [(l, h) for l in atlas.LOBES for h in ("L", "R")]
GROUP_NAMES = np.array([f"{l}_{h}" for l, h in GROUPS])
GROUP_OF = np.array([GROUPS.index((l, h)) for l, h in zip(atlas.LOBE, atlas.HEMI)])


def max_spanning_tree(W):
    """Edge mask (4,005,) of the maximum-weight spanning tree."""
    pos = W > 0
    inv = np.where(pos, W.max() + 1.0 - W, 0.0)  # positive, order reversed
    T = minimum_spanning_tree(inv).toarray()
    T = (T + T.T) > 0
    if T[IU].sum() != N - 1:
        raise ValueError("network is disconnected: the spanning tree has "
                         f"{T[IU].sum()} edges, not {N - 1}")
    return T[IU]


def binarise(W, n_edges, tree):
    w = W[IU]
    keep = tree.copy()
    rest = np.where(~tree & (w > 0))[0]
    order = rest[np.lexsort((rest, -w[rest]))]   # strongest first, ties by index
    keep[order[:n_edges - keep.sum()]] = True
    A = np.zeros((N, N))
    A[IU] = keep
    return A + A.T, keep


def between_lobe_share(A):
    other = GROUP_OF[:, None] != GROUP_OF[None, :]
    return (A * other).sum(1) / A.sum(1)


def lobe_means(x):
    """(..., 90) -> (..., 12), unweighted mean of each group's regions."""
    return np.stack([x[..., GROUP_OF == g].mean(-1) for g in range(len(GROUPS))], -1)


def verify_tree(S, n=5):
    """Our spanning tree must match networkx's maximum spanning tree."""
    for i in range(n):
        W = np.asarray(S[:, :, i], float)
        ours = W[IU][max_spanning_tree(W)].sum()
        G = nx.maximum_spanning_tree(nx.from_numpy_array(W))
        theirs = sum(d["weight"] for *_, d in G.edges(data=True))
        assert np.isclose(ours, theirs), f"tree weight differs on session {i}"
    print(f"spanning tree verified against networkx on {n} sessions")


def main():
    coh = pd.read_csv(paths.PROJECT / "results" / "cohort" / "cohort.csv")
    S = np.load(paths.STACK_UNTHRESH, mmap_mode="r")
    assert S.shape[2] == len(coh), "stack and cohort differ in length"
    verify_tree(S)

    n, D = len(coh), list(DENSITIES)
    comm = np.empty((n, len(D), N)); bls = np.empty_like(comm); deg = np.empty_like(comm)
    weakest = np.empty((n, len(D))); edges = np.zeros((n, len(IU[0])), bool)
    for i in range(n):
        W = np.asarray(S[:, :, i], float)
        assert np.allclose(W, W.T) and (np.diag(W) == 0).all()
        tree = max_spanning_tree(W)
        for j, d in enumerate(D):
            A, keep = binarise(W, DENSITIES[d], tree)
            assert keep.sum() == DENSITIES[d] and nx.is_connected(nx.from_numpy_array(A))
            comm[i, j], bls[i, j], deg[i, j] = communicability(A), between_lobe_share(A), A.sum(1)
            weakest[i, j] = W[IU][keep].min()
            if d == PRIMARY:
                edges[i] = keep
    assert np.isfinite(comm).all() and np.isfinite(bls).all()

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "bin10_mst_nodes.npz", communicability=comm,
                        between_lobe_share=bls, degree=deg, weakest_weight=weakest,
                        edges_10=edges, densities=np.array(D),
                        n_edges=np.array([DENSITIES[d] for d in D]),
                        dhcp_id=coh.dhcp_id.values, group_of=GROUP_OF,
                        group_names=GROUP_NAMES)
    j = D.index(PRIMARY)
    for tag, x, name in (("comm", comm, "communicability"),
                         ("bls", bls, "between_lobe_share")):
        X = lobe_means(x[:, j])
        np.savez_compressed(OUT / f"bin10_mst_{tag}_lobe.npz", X=X,
                            features=GROUP_NAMES, measure=name, dhcp_id=coh.dhcp_id.values)
        print(f"{name}: lobe features {X.shape}, mean {X.mean():.4g}, sd {X.std():.4g}")
    X = lobe_means(comm.mean(1))                 # mean over 8-12% per region, then per lobe
    np.savez_compressed(OUT / "bin10_comm_avg_lobe.npz", X=X, features=GROUP_NAMES,
                        measure="communicability, mean over 8-12% density",
                        densities=np.array(D), dhcp_id=coh.dhcp_id.values)
    print(f"communicability averaged over {D}%: lobe features {X.shape}")
    print(f"weakest retained weight at 10%: median {np.median(weakest[:, j]):.4g}, "
          f"min {weakest[:, j].min():.4g}")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
