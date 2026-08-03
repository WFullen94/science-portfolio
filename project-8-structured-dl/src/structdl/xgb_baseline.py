"""Stage 1a — the baseline: XGBoost with an Optuna CV search.

The classical champion for tabular data, and the bar the FT-Transformer has to clear.
Optuna proposes hyperparameters; k-fold CV on a train subsample scores each proposal
(the same nested pattern as P1). The winning config is refit on the full train split
and scored once on the held-out test set. Categoricals ride in as pandas `category`
dtype via XGBoost's native `enable_categorical` — no one-hot.
"""

from __future__ import annotations

import json

import mlflow
import numpy as np
import optuna
import xgboost as xgb

from structdl.config import load_config, resolve
from structdl.data import DataBundle, build_bundle
from structdl.metrics import binary_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _scale_pos_weight(y: np.ndarray) -> float:
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return neg / pos if pos else 1.0


def _search(b: DataBundle, cfg) -> dict:
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

    xcfg = cfg["xgb"]
    spw = _scale_pos_weight(b.y_train)

    # Subsample the train split for a fast CV search; final model uses all of it.
    n = min(xcfg["optuna_sample_rows"], len(b.y_train))
    Xs, _, ys, _ = train_test_split(
        b.X_train_df, b.y_train, train_size=n, stratify=b.y_train,
        random_state=xcfg["random_state"],
    )
    skf = StratifiedKFold(n_splits=xcfg["cv_folds"], shuffle=True,
                          random_state=xcfg["random_state"])

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 150, 500),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            tree_method="hist",
            enable_categorical=True,
            eval_metric="auc",
            scale_pos_weight=spw,
            n_jobs=-1,
            random_state=xcfg["random_state"],
        )
        clf = xgb.XGBClassifier(**params)
        auc = cross_val_score(clf, Xs, ys, cv=skf, scoring="roc_auc").mean()
        return float(auc)

    study = optuna.create_study(
        direction="maximize", study_name="xgb-unsw",
        sampler=optuna.samplers.TPESampler(seed=xcfg["random_state"]),
    )
    study.optimize(objective, n_trials=xcfg["optuna_trials"], show_progress_bar=False)
    print(f"[xgb] best CV ROC-AUC: {study.best_value:.4f}")

    best = study.best_params
    best.update(tree_method="hist", enable_categorical=True, eval_metric="auc",
                scale_pos_weight=spw, n_jobs=-1, random_state=xcfg["random_state"])
    return best


def run(b: DataBundle | None = None, cfg=None, log_to_mlflow: bool = True) -> dict:
    """Train + evaluate the XGBoost baseline. Returns test metrics."""
    cfg = cfg or load_config()
    b = b or build_bundle(cfg)

    params = _search(b, cfg)
    clf = xgb.XGBClassifier(**params)
    clf.fit(b.X_train_df, b.y_train)

    scores = clf.predict_proba(b.X_test_df)[:, 1]
    metrics = binary_metrics(b.y_test, scores)
    print(f"[xgb] test ROC-AUC={metrics['roc_auc']:.4f} "
          f"PR-AUC={metrics['pr_auc']:.4f} F1={metrics['f1']:.4f}")

    if log_to_mlflow:
        mcfg = cfg["mlflow"]
        mlflow.set_tracking_uri(mcfg["tracking_uri"])
        mlflow.set_experiment(mcfg["experiment"])
        with mlflow.start_run(run_name="xgboost-baseline"):
            mlflow.log_param("model", "xgboost")
            mlflow.log_params({k: v for k, v in params.items()
                               if k in {"n_estimators", "max_depth", "learning_rate"}})
            mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
    return metrics


def main() -> int:
    # Runs as its own process (see compare.py) — no torch here, so no OpenMP clash.
    cfg = load_config()
    metrics = run(cfg=cfg)
    out = resolve(cfg["paths"]["reports"]) / "xgb_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(f"[xgb] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
