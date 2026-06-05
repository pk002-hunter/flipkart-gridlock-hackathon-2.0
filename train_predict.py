"""
Traffic Demand Prediction — Main Training Script
=================================================
Trains an ensemble of LightGBM + XGBoost + CatBoost models
with cross-validation and generates submission.csv.

Usage:
    python train_predict.py

Output:
    submission.csv  (41778 rows × 2 columns)
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# ---------------------------------------------
# Paths
# ---------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "dataset", "train.csv")
TEST_PATH  = os.path.join(BASE_DIR, "dataset", "test.csv")
SUBMISSION_PATH = os.path.join(BASE_DIR, "submission_robust.csv")

# ---------------------------------------------
# Hyperparameters
# ---------------------------------------------
N_FOLDS = 5
SEED = 42

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "num_leaves": 127,           # Reduced from 255
    "max_depth": 8,              # Reduced from -1 (unlimited)
    "min_child_samples": 50,     # Increased from 20
    "feature_fraction": 0.7,     # Reduced from 0.8
    "bagging_fraction": 0.7,     # Reduced from 0.8
    "bagging_freq": 1,
    "reg_alpha": 0.5,            # Increased from 0.1
    "reg_lambda": 0.5,           # Increased from 0.1
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "max_depth": 6,              # Reduced from 8
    "min_child_weight": 10,      # Increased from 5
    "subsample": 0.7,            # Reduced from 0.8
    "colsample_bytree": 0.7,     # Reduced from 0.8
    "reg_alpha": 0.5,            # Increased from 0.1
    "reg_lambda": 2.0,           # Increased from 1.0
    "tree_method": "hist",
    "early_stopping_rounds": 100,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": 0,
}

CATBOOST_PARAMS = {
    "loss_function": "RMSE",
    "iterations": 3000,
    "learning_rate": 0.03,
    "depth": 6,                  # Reduced from 8
    "l2_leaf_reg": 5.0,          # Increased from 3.0
    "random_strength": 1.5,      # Increased from 1.0
    "bagging_temperature": 0.8,  # Increased from 0.5
    "od_type": "Iter",
    "od_wait": 100,
    "random_seed": SEED,
    "verbose": 0,
    "thread_count": -1,
}

# Ensemble weights (tune based on CV R²)
MODEL_WEIGHTS = {
    "lgbm": 0.40,
    "xgb": 0.30,
    "catboost": 0.30,
}


# ---------------------------------------------
# Helpers
# ---------------------------------------------

def print_banner(msg: str):
    print("\n" + "=" * 60)
    print(f"  {msg}")
    print("=" * 60)


def score_str(r2: float) -> str:
    if r2 >= 0.9:
        return f"R²={r2:.5f} [OK] Excellent"
    elif r2 >= 0.7:
        return f"R²={r2:.5f} [WARN] Good"
    else:
        return f"R²={r2:.5f} [LOW] Needs improvement"


# ---------------------------------------------
# Model Training Functions
# ---------------------------------------------

def train_lgbm(X_train, y_train, X_val, y_val, params):
    import lightgbm as lgb
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)],
    )
    best_iter = model.best_iteration_
    return model, best_iter


def train_xgb(X_train, y_train, X_val, y_val, params):
    import xgboost as xgb
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    best_iter = getattr(model, "best_iteration", model.n_estimators)
    return model, best_iter


def train_catboost(X_train, y_train, X_val, y_val, params):
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
    )
    return model, model.best_iteration_


# ---------------------------------------------
# Cross-Validation Loop
# ---------------------------------------------

def cross_validate_ensemble(X, y, X_test, n_folds=5):
    """
    Run N-fold cross-validation for all 3 models.
    Returns:
        oof_preds: out-of-fold predictions (shape: [n_train])
        test_preds: averaged test predictions (shape: [n_test])
        fold_scores: per-fold R² scores
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    oof_lgbm     = np.zeros(len(X))
    oof_xgb      = np.zeros(len(X))
    oof_catboost = np.zeros(len(X))

    test_lgbm     = np.zeros((len(X_test), n_folds))
    test_xgb      = np.zeros((len(X_test), n_folds))
    test_catboost = np.zeros((len(X_test), n_folds))

    fold_r2_lgbm = []
    fold_r2_xgb  = []
    fold_r2_cat  = []

    X_arr = X.values if hasattr(X, "values") else X
    y_arr = y.values if hasattr(y, "values") else y
    X_test_arr = X_test.values if hasattr(X_test, "values") else X_test

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_arr)):
        print(f"\n  -- Fold {fold + 1}/{n_folds} --")
        X_tr, X_val = X_arr[tr_idx], X_arr[val_idx]
        y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]

        # -- LightGBM --
        t0 = time.time()
        lgbm_model, lgbm_iters = train_lgbm(X_tr, y_tr, X_val, y_val, LGBM_PARAMS)
        lgbm_val_pred = lgbm_model.predict(X_val)
        lgbm_test_pred = lgbm_model.predict(X_test_arr)
        r2_lgbm = r2_score(y_val, lgbm_val_pred)
        oof_lgbm[val_idx] = lgbm_val_pred
        test_lgbm[:, fold] = lgbm_test_pred
        fold_r2_lgbm.append(r2_lgbm)
        print(f"    LightGBM  {score_str(r2_lgbm)}  |  iters={lgbm_iters}  |  {time.time()-t0:.1f}s")

        # -- XGBoost --
        t0 = time.time()
        xgb_model, xgb_iters = train_xgb(X_tr, y_tr, X_val, y_val, XGB_PARAMS)
        xgb_val_pred = xgb_model.predict(X_val)
        xgb_test_pred = xgb_model.predict(X_test_arr)
        r2_xgb = r2_score(y_val, xgb_val_pred)
        oof_xgb[val_idx] = xgb_val_pred
        test_xgb[:, fold] = xgb_test_pred
        fold_r2_xgb.append(r2_xgb)
        print(f"    XGBoost   {score_str(r2_xgb)}  |  iters={xgb_iters}  |  {time.time()-t0:.1f}s")

        # -- CatBoost --
        t0 = time.time()
        cat_model, cat_iters = train_catboost(X_tr, y_tr, X_val, y_val, CATBOOST_PARAMS)
        cat_val_pred = cat_model.predict(X_val)
        cat_test_pred = cat_model.predict(X_test_arr)
        r2_cat = r2_score(y_val, cat_val_pred)
        oof_catboost[val_idx] = cat_val_pred
        test_catboost[:, fold] = cat_test_pred
        fold_r2_cat.append(r2_cat)
        print(f"    CatBoost  {score_str(r2_cat)}  |  iters={cat_iters}  |  {time.time()-t0:.1f}s")

    # Average test predictions across folds
    test_lgbm_avg = test_lgbm.mean(axis=1)
    test_xgb_avg  = test_xgb.mean(axis=1)
    test_cat_avg  = test_catboost.mean(axis=1)

    # Ensemble OOF predictions
    w = MODEL_WEIGHTS
    oof_ensemble = (
        w["lgbm"] * oof_lgbm +
        w["xgb"]  * oof_xgb  +
        w["catboost"] * oof_catboost
    )
    test_ensemble = (
        w["lgbm"] * test_lgbm_avg +
        w["xgb"]  * test_xgb_avg  +
        w["catboost"] * test_cat_avg
    )

    # Print summary
    print_banner("Cross-Validation Summary")
    print(f"  LightGBM  CV R²: {np.mean(fold_r2_lgbm):.5f} ± {np.std(fold_r2_lgbm):.5f}")
    print(f"  XGBoost   CV R²: {np.mean(fold_r2_xgb):.5f}  ± {np.std(fold_r2_xgb):.5f}")
    print(f"  CatBoost  CV R²: {np.mean(fold_r2_cat):.5f}  ± {np.std(fold_r2_cat):.5f}")
    ensemble_r2 = r2_score(y_arr, oof_ensemble)
    print(f"\n  Ensemble  OOF R²: {score_str(ensemble_r2)}")
    print(f"  Final Score: {max(0, 100 * ensemble_r2):.2f} / 100")

    return oof_ensemble, test_ensemble, {
        "lgbm": fold_r2_lgbm,
        "xgb": fold_r2_xgb,
        "catboost": fold_r2_cat,
        "ensemble_r2": ensemble_r2,
    }


# ---------------------------------------------
# Optional: Post-processing
# ---------------------------------------------

def clip_predictions(preds: np.ndarray) -> np.ndarray:
    """Clip predictions to valid demand range [0, 1]."""
    return np.clip(preds, 0.0, 1.0)


# ---------------------------------------------
# Main
# ---------------------------------------------

def main():
    total_start = time.time()

    print_banner("Traffic Demand Prediction — Training Pipeline")

    # -- 1. Load Data --
    print("\n[Loading data...]")
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw  = pd.read_csv(TEST_PATH)
    print(f"  Train: {train_raw.shape}   Test: {test_raw.shape}")

    # Column rename alignment (train has 'demand' in position 5)
    # Verify columns match problem statement
    print(f"  Train columns: {list(train_raw.columns)}")
    print(f"  Test columns:  {list(test_raw.columns)}")

    # -- 2. Feature Engineering --
    print_banner("Feature Engineering")
    from feature_engineering import build_features
    X_train, y_train, X_test, feature_names, target_encoder = build_features(
        train_raw, test_raw, target_col="demand"
    )

    print(f"\n  Feature list ({len(feature_names)} features):")
    for i, f in enumerate(feature_names):
        print(f"    [{i+1:02d}] {f}")

    # -- 3. Cross-Validation & Training --
    print_banner(f"Training Ensemble ({N_FOLDS}-Fold CV)")
    oof_preds, test_preds, scores = cross_validate_ensemble(
        X_train, y_train, X_test, n_folds=N_FOLDS
    )

    # -- 4. Post-process predictions --
    test_preds = clip_predictions(test_preds)

    # -- 5. Create Submission --
    print_banner("Creating Submission")
    submission = pd.DataFrame({
        "Index": test_raw["Index"],
        "demand": test_preds
    })

    # Verify shape
    assert submission.shape == (41778, 2), f"Unexpected shape: {submission.shape}"
    assert list(submission.columns) == ["Index", "demand"], f"Unexpected columns: {list(submission.columns)}"

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"  [OK] Saved to: {SUBMISSION_PATH}")
    print(f"  Shape: {submission.shape}")
    print(f"  Demand stats:")
    print(f"    min   = {submission['demand'].min():.6f}")
    print(f"    max   = {submission['demand'].max():.6f}")
    print(f"    mean  = {submission['demand'].mean():.6f}")
    print(f"    std   = {submission['demand'].std():.6f}")
    print(f"\n  Preview (first 5 rows):")
    print(submission.head(5).to_string(index=False))

    elapsed = time.time() - total_start
    print_banner(f"Done! Total time: {elapsed/60:.1f} minutes")
    print(f"  Final Ensemble OOF R²: {scores['ensemble_r2']:.5f}")
    print(f"  Competition Score:     {max(0, 100 * scores['ensemble_r2']):.2f} / 100")


if __name__ == "__main__":
    main()
