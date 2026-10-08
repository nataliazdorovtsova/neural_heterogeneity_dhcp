"""Central pointers to this project's data.

Everything lives inside data/ as a local copy owned by nz01 (copied 2026-10-02
from /imaging/astle/ab15/Project 1). Nothing here reads from a colleague's
folder, so their reorganising cannot break this project and we cannot touch
their files. See DATA_PROVENANCE.md for sources and checksums.
"""

from pathlib import Path

PROJECT = Path(__file__).resolve().parent

DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
FIGURES = PROJECT / "figures"

# --- connectomes: AAL90 streamline-count (mu) stacks, shape (90, 90, 753) ----
CONNECTOMES = DATA / "connectomes" / "perbin25_mst_n753"      # primary
CONNECTOMES_UNTHRESH = DATA / "connectomes" / "unthresholded_n753"

STACK = CONNECTOMES / "weighted_perbin_retain25_mst.npy"
STACK_UNTHRESH = CONNECTOMES_UNTHRESH / "mu_unthresholded_stack.npy"

# subject_index aligns with the third axis of the stack it sits beside
PARTICIPANTS = CONNECTOMES / "participants_with_index.csv"
PARTICIPANTS_UNTHRESH = CONNECTOMES_UNTHRESH / "participants_with_index.csv"

# QC for the thresholding variants
QC_SUMMARY = CONNECTOMES / "qc_summary_with_nbins.csv"
SESSION_STATS = CONNECTOMES / "session_connectivity_stats.csv"

# --- behaviour and demographics ---------------------------------------------
BEHAVIOUR = DATA / "behaviour_outcomes"
DEMOGRAPHICS = BEHAVIOUR / "demographics.csv"   # birth weight, head circ., dMRI QC
CBCL = BEHAVIOUR / "cbcl_clean.csv"             # 18-month CBCL t-scores
ECBQ = BEHAVIOUR / "ecbq_clean.csv"             # ECBQ temperament scales

# --- topology descriptives for the final stack (ab15's, for cross-checking) --
DESCRIPTIVES = DATA / "descriptives" / "descriptives_topology_summary.csv"
DESCRIPTIVES_SUBJECTS = DATA / "descriptives" / "combined_df_dhcp_with_ids.csv"

# --- atlases and templates ---------------------------------------------------
ATLAS_DIR = DATA / "atlases"
AAL90 = ATLAS_DIR / "AAL90_in_GA40.nii.gz"

# --- polygenic scores: READ IN PLACE, DO NOT COPY ---------------------------
# These live behind a restricted five-person access group
# (nz01, ab15, da02, fp02, yh03). This project sits on /imaging/astle, readable
# by the whole cbsu-fs-astle-lab group, which is far wider. Copying genotype-
# derived data across that boundary would widen access beyond who is authorised
# under the dHCP 2026 agreement. Read from these paths; do not mirror them into
# data/. See DECISIONS.md.
DHCP2026 = Path("/imaging/projects/external/dHCP_2026")
PRS = DHCP2026 / "data" / "genetics_qc" / "prs" / "dhcp_prs_all.csv"

# Wider dHCP 2026 release, also read-in-place: participant enrolment, imaging,
# genomics (incl. methylation), 18-month follow-up at source (CBCL + ECBQ,
# possibly more complete than the cleaned copies in data/behaviour_outcomes),
# eye tracking, clinical details.
DHCP2026_DATA = DHCP2026 / "data"
