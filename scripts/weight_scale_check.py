"""Is axis 2 (strength everywhere) an overall per-infant weight scale?

    python scripts/weight_scale_check.py

Strength is in raw edge-weight units, so any per-infant scaling of the whole
connectome (biological, such as brain size, or acquisition-related) moves
every region's strength together. This correlates the leading axes of the
selected model with per-infant quantities, all adjusted for age:

  log total edge weight     sum of the upper triangle of W
  log max edge weight       the Onnela normaliser (AUDIT.md finding A)
  head circumference        at scan (sentinel codes removed; AUDIT.md finding I)
  birth weight              adjusted for GA at birth as well as PMA
  network density           post-threshold

PMA is removed with the same penalised spline as the model's trajectories.
n varies with missing covariates and is reported.

Writes results/significance/weight_scale.json.
"""

import json

import numpy as np

import paths
from model import splines
from scripts.run_spec import load
from scripts.compare_variant_axes import axes_of

Q = 70


def residualise(y, coh, extra=None):
    ok = np.isfinite(y)
    B = splines.basis(coh.scan_age.values)
    if extra is not None:
        B = np.column_stack([B, extra])
    P = np.zeros((B.shape[1], B.shape[1]))
    P[:20, :20] = splines.difference_penalty(20)
    coef = np.linalg.solve(B[ok].T @ B[ok] + P, B[ok].T @ y[ok])
    r = np.full_like(y, np.nan, dtype=float)
    r[ok] = y[ok] - B[ok] @ coef
    return r


def main():
    coh, _ = load()
    A = np.load(paths.STACK, mmap_mode="r")
    idx = coh.subject_index.values
    iu = np.triu_indices(90, 1)
    total = np.log(np.array([A[:, :, i][iu].sum() for i in idx]))
    maxw = np.log(np.array([A[:, :, i].max() for i in idx]))
    hc = coh.head_circumference_scan.values.astype(float)
    bw = coh.birth_weight.values.astype(float)
    ga = coh.birth_age.values.astype(float)
    cov = {
        "log total edge weight": residualise(total, coh),
        "log max edge weight": residualise(maxw, coh),
        "head circumference": residualise(hc, coh),
        "birth weight (adj. GA)": residualise(bw, coh, extra=ga),
        "network density": residualise(coh.density_thr.values.astype(float), coh),
    }
    a = axes_of(paths.RESULTS / "final" / f"K1_q{Q}.npz")
    out = {"q": Q, "axes": {}, "between": {}}
    for name, v in cov.items():
        ok = np.isfinite(v)
        out["axes"][name] = dict(n=int(ok.sum()),
                                 r=[round(float(np.corrcoef(a["scores"][ok, j], v[ok])[0, 1]), 3)
                                    for j in range(3)])
    names = list(cov)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            ok = np.isfinite(cov[x]) & np.isfinite(cov[y])
            out["between"][f"{x} ~ {y}"] = round(float(np.corrcoef(cov[x][ok], cov[y][ok])[0, 1]), 3)
    (paths.RESULTS / "significance" / "weight_scale.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
