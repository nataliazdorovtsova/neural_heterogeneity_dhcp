"""Who makes frontal communicability skewed? (run bin10, step 1 follow-up)

    python scripts/check_bin10_frontal.py      # after scripts/build_bin10.py

The pilot found frontal communicability strongly left-skewed (skew -2.35 left,
-1.72 right, at 10%): a few infants sit far below their age curve. This finds
them (residual z < -3 on the age curve, in the 8-12% averaged features, D19),
asks which frontal regions carry the low values and whether those infants
differ in scan quality, radiology score or age, and reports the skew with them
set aside and after a log transform. Training infants only.

Writes results/bin10/pilot/frontal_check.json and figures/bin10/b4b_frontal_check.
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, mannwhitneyu

import paths
import atlas
import figstyle as fs
from scripts.build_bin10 import GROUP_NAMES, GROUP_OF
from scripts.pilot_bin10 import Curve

FEAT = paths.PROJECT / "results" / "features"
OUT = paths.PROJECT / "results" / "bin10" / "pilot"
FIG = paths.PROJECT / "figures" / "bin10"
Z_CUT = -3.0
COVARIATES = ["scan_age", "birth_age", "qc_dmri_eddy_rel_motion", "radiology_score",
              "density_unthresh"]


def main():
    coh = pd.read_csv(paths.PROJECT / "results" / "cohort" / "cohort.csv")
    tr = coh.in_primary.values.astype(bool) & (coh.split == "train").values
    c = coh[tr].reset_index(drop=True)
    t = c.scan_age.values
    X = np.load(FEAT / "bin10_comm_avg_lobe.npz")["X"][tr]
    nodes = np.load(FEAT / "bin10_mst_nodes.npz")
    node = nodes["communicability"][tr].mean(1)            # (n, 90), mean over 8-12%
    deg = nodes["degree"][tr].mean(1)

    out = dict(n=int(tr.sum()), z_cut=Z_CUT, skew_all_lobes=np.round(skew(X, axis=0), 2).tolist(),
               lobes=GROUP_NAMES.tolist(), frontal={})
    zs, flagged_any = {}, np.zeros(len(t), bool)
    for h in ("L", "R"):
        g = list(GROUP_NAMES).index(f"frontal_{h}")
        f = Curve(t, X[:, g])
        z = f.resid / f.resid.std(ddof=1)
        zs[h] = z
        flag = z < Z_CUT
        flagged_any |= flag
        logf = Curve(t, np.log(X[:, g]))
        out["frontal"][h] = dict(
            skew=round(float(skew(f.resid)), 2),
            skew_without_flagged=round(float(skew(f.resid[~flag])), 2),
            skew_log=round(float(skew(logf.resid)), 2),
            n_flagged=int(flag.sum()), z_flagged=np.round(np.sort(z[flag]), 2).tolist())

    # which frontal regions carry the low values: per-region residual z, flagged infants
    fr = np.where(atlas.LOBE == "frontal")[0]
    reg_z = np.column_stack([Curve(t, node[:, r]).resid for r in fr])
    reg_z /= reg_z.std(0, ddof=1)
    F = flagged_any
    region_mean_z = reg_z[F].mean(0) if F.any() else np.zeros(len(fr))
    worst = np.argsort(region_mean_z)[:5]
    out["regions_lowest_in_flagged"] = [dict(region=atlas.NAMES[fr[i]], mean_z=round(float(region_mean_z[i]), 2),
                                             degree_flagged=round(float(deg[F, fr[i]].mean()), 1),
                                             degree_others=round(float(deg[~F, fr[i]].mean()), 1))
                                        for i in worst]
    # are the flagged infants different?
    cmp = {}
    for col in COVARIATES:
        v = c[col].values.astype(float)
        a, b = v[F & np.isfinite(v)], v[~F & np.isfinite(v)]
        p = float(mannwhitneyu(a, b).pvalue) if len(a) and len(b) else None
        cmp[col] = dict(flagged_median=float(np.median(a)) if len(a) else None,
                        others_median=float(np.median(b)), p=p)
    cmp["preterm_share"] = dict(flagged=float(c.preterm.values[F].mean()) if F.any() else None,
                                others=float(c.preterm.values[~F].mean()))
    out["flagged_vs_others"] = cmp
    out["n_flagged_any"] = int(F.sum())
    out["flagged_infants"] = [dict(dhcp_id=str(c.dhcp_id[i]), pma=round(float(t[i]), 1),
                                   ga=round(float(c.birth_age[i]), 1), z_L=round(float(zs["L"][i]), 2),
                                   z_R=round(float(zs["R"][i]), 2),
                                   radiology=c.radiology_score[i],
                                   motion=round(float(c.qc_dmri_eddy_rel_motion[i]), 3),
                                   lowest_region=atlas.NAMES[fr[int(np.argmin(reg_z[i]))]],
                                   lowest_region_z=round(float(reg_z[i].min()), 2))
                              for i in np.where(F)[0]]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frontal_check.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=1, default=str))

    fs.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1, 1.6]))
    fig.subplots_adjust(wspace=0.3, bottom=0.42)
    ax = axes[0]
    bins = np.linspace(min(zs["L"].min(), zs["R"].min()) - 0.2, max(zs["L"].max(), zs["R"].max()) + 0.2, 45)
    for h, col in (("L", fs.COLORS["orange"]), ("R", fs.COLORS["blue"])):
        ax.hist(zs[h], bins=bins, color=col, alpha=0.55, edgecolor=fs.TEXT_PRIMARY, linewidth=0.4,
                label=f"frontal {h}")
    fs.reference_line(ax, Z_CUT, axis="x")
    ax.set_xlabel("residual z (distance from the age curve)"); ax.set_ylabel("infants")
    fs.panel_title(ax, "A", "Frontal communicability")
    ax.legend(loc="upper left")
    ax = axes[1]
    order = np.argsort(region_mean_z)
    x = np.arange(len(fr))
    ax.set_xlim(-0.6, len(fr) - 0.4)
    lo = min(region_mean_z.min() * 1.15, -0.5)
    ax.set_ylim(lo, max(0.5, region_mean_z.max() * 1.15))
    fs.rounded_bars(ax, x, region_mean_z[order], width=0.7,
                    color=[fs.NEGATIVE_COLOUR if v < 0 else fs.POSITIVE_COLOUR for v in region_mean_z[order]])
    fs.reference_line(ax, 0)
    ax.set_xticks(x, [atlas.NAMES[fr[i]].replace("_", " ") for i in order], rotation=70, ha="right",
                  fontsize=9)
    ax.set_ylabel("mean residual z")
    fs.panel_title(ax, "B", f"Which regions, in the {int(F.sum())} low infants")
    fig.text(0.07, -0.08, f"{out['n']} training infants; communicability averaged over 8-12% density. "
             f"A: dashed line = z {Z_CUT:g}, the cut for 'low'. B: residual z of each frontal region,\n"
             "averaged over the infants below the cut in either hemisphere.",
             fontsize=fs.NOTE_SIZE, color=fs.TEXT_SECONDARY, va="top")
    fs.save_figure(fig, FIG / "b4b_frontal_check")


if __name__ == "__main__":
    main()
