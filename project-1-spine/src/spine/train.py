"""Stage 3 — Train: XGBoost + Optuna search, tracked in MLflow, explained with
SHAP, and registered + promoted in the MLflow Model Registry.

Flow:
  gold Delta -> pandas
  -> Optuna CV search (each trial logged as a nested MLflow run)
  -> final fit with early stopping (MLflow autolog captures params/metrics/model)
  -> held-out test evaluation (ROC-AUC, PR-AUC, precision/recall/F1)
  -> SHAP global explanation (logged as an artifact)
  -> register best model; promote to the 'staging' alias iff it clears the bar.
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import shap
import xgboost as xgb
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from spine.config import get_spark, load_config, resolve
from spine.features import feature_columns, to_model_matrix
from spine.schema import LABEL_BINARY


# ----------------------------------------------------------------------------- data
def load_gold(cfg):
    spark = get_spark("spine-train")
    try:
        paths = cfg["paths"]
        train = spark.read.format("delta").load(str(resolve(paths["gold_train"])))
        test = spark.read.format("delta").load(str(resolve(paths["gold_test"])))
        feats = feature_columns(train)  # excludes Label + Attack
        cols = feats + [LABEL_BINARY]
        train_pdf = train.select(*cols).toPandas()
        test_pdf = test.select(*cols).toPandas()
    finally:
        spark.stop()

    X_train = to_model_matrix(train_pdf, feats)
    y_train = train_pdf[LABEL_BINARY].astype(int)
    X_test = to_model_matrix(test_pdf, feats)
    y_test = test_pdf[LABEL_BINARY].astype(int)
    return X_train, y_train, X_test, y_test, feats


# ----------------------------------------------------------------------- Optuna search
def optuna_search(X, y, cfg, scale_pos_weight: float):
    tcfg = cfg["train"]
    # Subsample for a fast CV search; final model trains on all rows.
    n = min(tcfg.get("optuna_sample_rows", len(X)), len(X))
    Xs, _, ys, _ = train_test_split(
        X, y, train_size=n, stratify=y, random_state=tcfg["random_state"]
    )
    skf = StratifiedKFold(n_splits=tcfg["cv_folds"], shuffle=True,
                          random_state=tcfg["random_state"])

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 150, 500),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            tree_method="hist",
            eval_metric="auc",
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=tcfg["random_state"],
        )
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            clf = xgb.XGBClassifier(**params)
            auc = cross_val_score(clf, Xs, ys, cv=skf, scoring="roc_auc").mean()
            mlflow.log_metric("cv_roc_auc", float(auc))
            return auc

    study = optuna.create_study(direction="maximize",
                                study_name="xgb-nids",
                                sampler=optuna.samplers.TPESampler(seed=tcfg["random_state"]))
    study.optimize(objective, n_trials=tcfg["optuna_trials"], show_progress_bar=False)
    print(f"[train] best CV ROC-AUC: {study.best_value:.4f}")
    return study


# -------------------------------------------------------------------------- final fit
def fit_final(best_params, X_train, y_train, scale_pos_weight, cfg):
    tcfg = cfg["train"]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train,
        random_state=tcfg["random_state"],
    )
    params = dict(
        **best_params,
        tree_method="hist",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=30,
        n_jobs=-1,
        random_state=tcfg["random_state"],
    )
    model = xgb.XGBClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    print(f"[train] final model best_iteration={model.best_iteration}")
    return model


# ------------------------------------------------------------------------- evaluation
def evaluate(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred)
    metrics = {
        "test_roc_auc": float(roc_auc_score(y_test, proba)),
        "test_pr_auc": float(average_precision_score(y_test, proba)),
        "test_f1": float(f1_score(y_test, pred)),
        "test_precision": float(precision_score(y_test, pred)),
        "test_recall": float(recall_score(y_test, pred)),
        "test_tn": int(cm[0, 0]), "test_fp": int(cm[0, 1]),
        "test_fn": int(cm[1, 0]), "test_tp": int(cm[1, 1]),
    }
    print("[train] test metrics: " + ", ".join(
        f"{k}={v:.4f}" for k, v in metrics.items() if k.endswith(("auc", "f1", "precision", "recall"))
    ))
    return metrics


def shap_explain(model, X_test, feats, out_dir):
    sample = X_test.sample(min(2000, len(X_test)), random_state=0)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "shap_summary.png"
    plt.figure()
    shap.summary_plot(shap_values, sample, feature_names=feats, show=False,
                      plot_type="bar", max_display=15)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = sorted(zip(feats, mean_abs.tolist()), key=lambda t: t[1], reverse=True)
    top = {f: v for f, v in ranking[:15]}
    (out_dir / "shap_top_features.json").write_text(json.dumps(top, indent=2))
    print("[train] top SHAP features: " + ", ".join(list(top)[:5]))
    return plot_path, out_dir / "shap_top_features.json"


# -------------------------------------------------------------- registry + promotion
def register_and_promote(run_id, metrics, cfg):
    name = cfg["mlflow"]["registered_model"]
    client = MlflowClient()
    try:
        client.create_registered_model(name)
    except Exception:
        pass  # already exists

    version = mlflow.register_model(f"runs:/{run_id}/model", name).version
    client.set_registered_model_alias(name, "candidate", version)

    bar = cfg["train"]["promote_min_roc_auc"]
    promoted = metrics["test_roc_auc"] >= bar
    if promoted:
        client.set_registered_model_alias(name, "staging", version)
    print(f"[train] registered {name} v{version} "
          f"({'promoted to @staging' if promoted else f'held (roc_auc<{bar})'})")
    return version, promoted


# -------------------------------------------------------------------------------- main
def main() -> int:
    cfg = load_config()
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    exp = cfg["mlflow"]["experiment"]
    if mlflow.get_experiment_by_name(exp) is None:
        mlflow.create_experiment(exp, artifact_location=str(resolve(cfg["mlflow"]["artifact_location"])))
    mlflow.set_experiment(exp)

    X_train, y_train, X_test, y_test, feats = load_gold(cfg)
    print(f"[train] loaded train={X_train.shape} test={X_test.shape}")
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    with mlflow.start_run(run_name="xgb-nids") as run:
        mlflow.set_tag("domain", "network-intrusion-detection")
        mlflow.log_param("n_features", len(feats))
        mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 4))

        study = optuna_search(X_train, y_train, cfg, scale_pos_weight)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_cv_roc_auc", study.best_value)

        # Autolog the final fit only (keeps the 25 search trials out of the model log).
        mlflow.xgboost.autolog(log_models=True, silent=True)
        model = fit_final(study.best_params, X_train, y_train, scale_pos_weight, cfg)
        mlflow.xgboost.autolog(disable=True)

        metrics = evaluate(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        reports = resolve(cfg["paths"]["reports"])
        plot_path, json_path = shap_explain(model, X_test, feats, reports)
        mlflow.log_artifact(str(plot_path), artifact_path="shap")
        mlflow.log_artifact(str(json_path), artifact_path="shap")

        version, promoted = register_and_promote(run.info.run_id, metrics, cfg)
        mlflow.set_tag("registered_version", version)
        mlflow.set_tag("promoted", promoted)

    print("[train] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
