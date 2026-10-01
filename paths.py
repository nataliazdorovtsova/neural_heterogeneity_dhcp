"""Central pointers to data that lives outside this workspace.

Nothing here is copied into the project: the connectome stacks and behavioural
tables stay in ab15's dHCP project tree and are reached through symlinks in
data/.  Treat everything under AB15 as read-only.
"""

from pathlib import Path

PROJECT = Path(__file__).resolve().parent

DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
FIGURES = PROJECT / "figures"

# --- external sources (read-only) -------------------------------------------
AB15 = Path("/imaging/astle/ab15/dHCP project")

# AAL90 streamline-count (mu) stacks, shape (90, 90, 753)
CONNECTOMES = DATA / "connectomes_perbin25_mst_n753"     # per-bin 25% + MST, nbins=10
CONNECTOMES_UNTHRESH = DATA / "connectomes_unthresholded_n753"

STACK = CONNECTOMES / "weighted_perbin_retain25_mst.npy"
STACK_UNTHRESH = CONNECTOMES_UNTHRESH / "mu_unthresholded_stack.npy"

# subject_index aligns with the third axis of the stacks
PARTICIPANTS = CONNECTOMES / "participants_with_index.csv"

# behaviour and QC
DEMOGRAPHICS = DATA / "behavior" / "demographics.csv"
CBCL = DATA / "behavior" / "cbcl_clean.csv"
ECBQ = DATA / "behavior" / "ecbq_clean.csv"

ATLAS = DATA / "atlases_templates"
