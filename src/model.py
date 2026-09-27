import time
from typing import Dict, List, Set, Tuple, Optional, Any
from pathlib import Path
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

class EntityResolutionModel:
    """Unified wrapper for LightGBM and Gradient Boosting models for entity resolution."""
    def __init__(
        self,
        model_type: str = "lightgbm",
        lgb_params: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
    ):
        self.model_type = model_type
        self.feature_names = feature_names or []
        self.lgb_booster: Optional[lgb.Booster] = None
        self.hgb_model: Optional[HistGradientBoostingClassifier] = None
        
        default_lgb = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": 0.04,
            "num_leaves": 45,
            "max_depth": 7,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "min_child_samples": 20,
            "verbose": -1,
            "n_jobs": -1,
            "random_state": 42,
        }
        self.lgb_params = lgb_params or default_lgb

    def train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        num_boost_round: int = 400,
        early_stopping_rounds: int = 30,
    ) -> Dict[str, float]:
        """Train LightGBM binary classifier."""
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        valid_sets = [train_data]
        valid_names = ["train"]
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=self.feature_names)
            valid_sets.append(val_data)
            valid_names.append("valid")
            
        callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)] if (X_val is not None and y_val is not None) else []
        
        self.lgb_booster = lgb.train(
            self.lgb_params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        
        metrics = {}
        if X_val is not None and y_val is not None:
            val_preds = self.lgb_booster.predict(X_val)
            metrics["val_roc_auc"] = float(roc_auc_score(y_val, val_preds))
            metrics["val_pr_auc"] = float(average_precision_score(y_val, val_preds))
        return metrics

    def train_hist_gradient_boosting(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Train scikit-learn HistGradientBoostingClassifier."""
        self.hgb_model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.04,
            max_leaf_nodes=45,
            max_depth=7,
            min_samples_leaf=20,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )
        self.hgb_model.fit(X_train, y_train)
        
        metrics = {}
        if X_val is not None and y_val is not None:
            val_preds = self.hgb_model.predict_proba(X_val)[:, 1]
            metrics["val_roc_auc"] = float(roc_auc_score(y_val, val_preds))
            metrics["val_pr_auc"] = float(average_precision_score(y_val, val_preds))
        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict match probabilities."""
        if self.model_type == "lightgbm" and self.lgb_booster is not None:
            return self.lgb_booster.predict(X)
        elif self.model_type == "hist_gb" and self.hgb_model is not None:
            return self.hgb_model.predict_proba(X)[:, 1]
        elif self.model_type == "ensemble":
            p_lgb = self.lgb_booster.predict(X) if self.lgb_booster else np.zeros(len(X))
            p_hgb = self.hgb_model.predict_proba(X)[:, 1] if self.hgb_model else np.zeros(len(X))
            return 0.70 * p_lgb + 0.30 * p_hgb
        else:
            raise ValueError(f"Model not trained or unknown type: {self.model_type}")

    def save_model(self, out_path: Path):
        """Save LightGBM model text file."""
        if self.lgb_booster is not None:
            self.lgb_booster.save_model(str(out_path))

    def load_model(self, in_path: Path):
        """Load LightGBM model file."""
        self.lgb_booster = lgb.Booster(model_file=str(in_path))
        self.feature_names = self.lgb_booster.feature_name()
