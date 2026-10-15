"""AAL90 region names, lobes and centroids, in connectome node order.

Node i of every connectome is AAL label i (data/atlases/AAL90_node_index.csv,
copied from ab15's review bundle; identity mapping 1..90). Labels follow the
standard AAL order (Tzourio-Mazoyer et al., 2002): left and right alternate,
odd = left.

Feature coordinates are region-major: coordinate d = 3 * region + measure.
"""

import numpy as np
import pandas as pd

import paths

_BILATERAL = [
    "Precentral", "Frontal_Sup", "Frontal_Sup_Orb", "Frontal_Mid", "Frontal_Mid_Orb",
    "Frontal_Inf_Oper", "Frontal_Inf_Tri", "Frontal_Inf_Orb", "Rolandic_Oper",
    "Supp_Motor_Area", "Olfactory", "Frontal_Sup_Medial", "Frontal_Med_Orb", "Rectus",
    "Insula", "Cingulum_Ant", "Cingulum_Mid", "Cingulum_Post", "Hippocampus",
    "ParaHippocampal", "Amygdala", "Calcarine", "Cuneus", "Lingual", "Occipital_Sup",
    "Occipital_Mid", "Occipital_Inf", "Fusiform", "Postcentral", "Parietal_Sup",
    "Parietal_Inf", "SupraMarginal", "Angular", "Precuneus", "Paracentral_Lobule",
    "Caudate", "Putamen", "Pallidum", "Thalamus", "Heschl", "Temporal_Sup",
    "Temporal_Pole_Sup", "Temporal_Mid", "Temporal_Pole_Mid", "Temporal_Inf",
]
NAMES = [f"{n}_{h}" for n in _BILATERAL for h in ("L", "R")]

LOBES = ("frontal", "parietal", "temporal", "occipital", "limbic", "subcortical")
_LOBE_OF = {
    "frontal": ["Precentral", "Frontal_Sup", "Frontal_Sup_Orb", "Frontal_Mid",
                "Frontal_Mid_Orb", "Frontal_Inf_Oper", "Frontal_Inf_Tri", "Frontal_Inf_Orb",
                "Rolandic_Oper", "Supp_Motor_Area", "Olfactory", "Frontal_Sup_Medial",
                "Frontal_Med_Orb", "Rectus", "Paracentral_Lobule"],
    "parietal": ["Postcentral", "Parietal_Sup", "Parietal_Inf", "SupraMarginal",
                 "Angular", "Precuneus"],
    "temporal": ["Heschl", "Temporal_Sup", "Temporal_Pole_Sup", "Temporal_Mid",
                 "Temporal_Pole_Mid", "Temporal_Inf", "Fusiform"],
    "occipital": ["Calcarine", "Cuneus", "Lingual", "Occipital_Sup", "Occipital_Mid",
                  "Occipital_Inf"],
    "limbic": ["Insula", "Cingulum_Ant", "Cingulum_Mid", "Cingulum_Post", "Hippocampus",
               "ParaHippocampal", "Amygdala"],
    "subcortical": ["Caudate", "Putamen", "Pallidum", "Thalamus"],
}
LOBE = np.array([next(l for l, ns in _LOBE_OF.items() if n.rsplit("_", 1)[0] in ns)
                 for n in NAMES])
HEMI = np.array([n[-1] for n in NAMES])

assert len(NAMES) == 90 and sum(len(v) for v in _LOBE_OF.values()) == 45


def check_node_index():
    """The connectome nodes must map 1:1 onto atlas labels in order."""
    idx = pd.read_csv(paths.ATLAS_DIR / "AAL90_node_index.csv")
    assert (idx.node_index.values == idx.aal_label.values).all(), "node order != AAL order"
    assert len(idx) == 90


def load_atlas():
    import nibabel as nib
    img = nib.load(paths.AAL90)
    return np.asarray(img.dataobj).astype(int), img.affine


def centroids(lab=None, affine=None):
    """(90, 3) world-space centroids, label order. Checks hemisphere parity."""
    from nibabel.affines import apply_affine
    if lab is None:
        lab, affine = load_atlas()
    c = np.zeros((90, 3))
    for r in range(90):
        c[r] = apply_affine(affine, np.argwhere(lab == r + 1).mean(0))
    # Odd labels are left. In RAS world space every left region of a pair should
    # sit at smaller x than its right partner.
    left_smaller = (c[0::2, 0] < c[1::2, 0]).mean()
    assert left_smaller > 0.95, f"hemisphere parity check failed ({left_smaller:.2f})"
    return c
