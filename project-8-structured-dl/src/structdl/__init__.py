"""Structured-Data DL — FT-Transformer vs XGBoost on tabular NIDS data.

NOTE (macOS OpenMP): XGBoost and PyTorch each ship their OWN OpenMP runtime
(two `libomp` copies). When both are resident in one process, whichever runs its
threaded kernels second segfaults (SIGSEGV) — reproducible in XGBoost CV *and* in
the FT-Transformer's forward pass. Import order only moves the crash between them.
The robust fix is process isolation: `compare.py` runs each model in its own
subprocess (they never co-reside), and because `build_bundle` is deterministic the
split is identical across processes. So this package deliberately imports NEITHER
torch nor xgboost at import time — each submodule pulls in only what it needs.
"""
