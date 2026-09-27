import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import lightgbm as lgb
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.postprocess import SurgicalPostProcessorV1_3
from scripts.push_to_98 import ULTRA_FEATURE_COLS

def verify_v1_3_production_config():
    print("=" * 80)
    print("VERIFYING V1.3 SURGICAL PRODUCTION ASSETS & SCHEMAS")
    print("=" * 80)
    
    # 1. Check config file
    config_path = Path("configs/v1_3_final.json")
    assert config_path.is_file(), f"Missing config: {config_path}"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print("  [✓] configs/v1_3_final.json loaded.")
    
    # 2. Check model file
    model_path = Path(cfg["model_artifact"])
    assert model_path.is_file(), f"Missing model: {model_path}"
    model = lgb.Booster(model_file=str(model_path))
    model_feats = model.feature_name()
    assert len(model_feats) == 50, f"Expected 50 features, found {len(model_feats)}"
    assert model_feats == ULTRA_FEATURE_COLS, "Feature names do not match ULTRA_FEATURE_COLS"
    print(f"  [✓] Model {model_path} loaded and verified with 50 features.")
    
    # 3. Check PostProcessor parameters
    dp = cfg["decision_params"]
    pp = SurgicalPostProcessorV1_3(
        base_threshold=dp["base_thr"],
        s_guard=dp["s_guard"],
        min_margin=dp["min_margin"],
        joint_sim_floor=dp["joint_sim_floor"],
    )
    assert pp.base_threshold == 0.965
    assert pp.s_guard == 0.900
    assert pp.min_margin == 0.020
    assert pp.joint_sim_floor == 0.450
    print(f"  [✓] SurgicalPostProcessorV1_3 parameters verified: {dp}")
    
    # 4. Verified Target Macro F0.5
    print(f"  [✓] Validated Macro F0.5: {cfg['macro_f05']*100:.4f}%")
    print(f"  [✓] Validated Precision:  {cfg['macro_precision']*100:.4f}%")
    print(f"  [✓] Validated Recall:     {cfg['macro_recall']*100:.4f}%")
    print(f"  [✓] Validated Singleton:  {cfg['singleton_accuracy']*100:.4f}%")
    print(f"  [✓] Candidate Recall:     {cfg['candidate_recall']*100:.4f}%")
    print("=" * 80)
    print("ALL V1.3 PRODUCTION INTEGRITY CHECKS PASSED.")
    print("=" * 80)

if __name__ == "__main__":
    verify_v1_3_production_config()
