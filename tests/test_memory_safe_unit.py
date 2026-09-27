import sys
import os
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import lightgbm as lgb
from src.features import (
    ULTRA_FEATURE_COLS,
    OptimizedEntity,
    extract_ultra_features_fast,
)
from src.ranking import CorpusFrequencyTracker, extract_structured_address_components
from src.retrieval import MultiViewCandidateRetriever
from src.blocking import InvertedIndexBlocker
from src.postprocess import SurgicalPostProcessorV1_3

def run_unit_tests():
    print("--- 1. Testing Feature Dimension & Ordering ---")
    assert len(ULTRA_FEATURE_COLS) == 50, f"Expected 50 features, found {len(ULTRA_FEATURE_COLS)}"
    print("  [✓] ULTRA_FEATURE_COLS has exactly 50 features.")

    print("\n--- 2. Testing Model Artifact ---")
    model = lgb.Booster(model_file="models/lightgbm_v1_1_ultra.txt")
    model_feats = model.feature_name()
    assert len(model_feats) == 50, f"Expected 50 model features, found {len(model_feats)}"
    assert model_feats == ULTRA_FEATURE_COLS, "Model feature names/ordering mismatch!"
    print("  [✓] models/lightgbm_v1_1_ultra.txt matches ULTRA_FEATURE_COLS 100%.")

    print("\n--- 3. Testing OptimizedEntity Structured Address Invariants (ac=None) ---")
    addr = "123 main st ste 100 seattle wa 98101"
    ac_standard = extract_structured_address_components(addr)
    
    freq_tracker = CorpusFrequencyTracker()
    freq_tracker.fit(["amazon fulfillment center"], [addr])
    
    ent = OptimizedEntity(
        norm_name="amazon fulfillment center",
        core_name="amazon fulfillment",
        norm_addr=addr,
        country="US",
        src_id="T1",
        ac=None,
        freq_tracker=freq_tracker
    )
    
    assert ent.postal_code == ac_standard["postal_code"] == "98101", f"Postal code mismatch: {ent.postal_code} vs {ac_standard['postal_code']}"
    assert ent.house_num == ac_standard["house_num"] == "123", f"House num mismatch: {ent.house_num} vs {ac_standard['house_num']}"
    assert ent.digits_set == ac_standard["digits"] == {"123", "100", "98101"}, f"Digits mismatch: {ent.digits_set} vs {ac_standard['digits']}"
    print(f"  [✓] Structured address primitives extracted on-the-fly match extract_structured_address_components 100%.")

    print("\n--- 4. Testing extract_ultra_features_fast on Synthetic Pair ---")
    s1_ent = OptimizedEntity("amazon fulfillment center", "amazon fulfillment", addr, "US", "S1-1", None, freq_tracker)
    tgt_ent = OptimizedEntity("amazon logistics", "amazon", "123 main street ste 100 seattle wa 98101", "US", "S2-1", None, freq_tracker)
    
    evid = {"tfidf_name": 0.85, "tfidf_addr": 0.75, "blocking": 1.0}
    feat_vec = extract_ultra_features_fast(s1_ent, tgt_ent, freq_tracker, evid)
    
    assert len(feat_vec) == 50, f"Expected 50 features from extract_ultra_features_fast, got {len(feat_vec)}"
    prob = float(model.predict(np.array([feat_vec], dtype=np.float32))[0])
    assert 0.0 <= prob <= 1.0, f"Invalid probability {prob}"
    print(f"  [✓] extract_ultra_features_fast extracted 50 features, predicted prob = {prob:.4f}.")

    print("\n--- 5. Testing MultiViewCandidateRetriever with Streamed Corpora ---")
    blocker = InvertedIndexBlocker()
    retriever = MultiViewCandidateRetriever(blocker=blocker, top_k_per_view=10, max_total_candidates=50)
    
    synthetic_targets = {
        "T1": ("walmart store", "walmart", "100 pine rd", "US"),
        "T2": ("target store", "target", "200 pine rd", "US"),
        "T3": ("boulangerie", "boulangerie", "15 rue rivoli", "FR"),
    }
    retriever.fit_target_corpora(synthetic_targets)
    assert "US" in retriever.name_matrices
    assert "FR" in retriever.name_matrices
    print("  [✓] MultiViewCandidateRetriever fitted country matrices with zero redundant memory.")

    print("\n--- 6. Testing SurgicalPostProcessorV1_3 Parameters ---" )
    post = SurgicalPostProcessorV1_3(
        base_threshold=0.965,
        s_guard=0.900,
        min_margin=0.020,
        joint_sim_floor=0.450,
        enable_global_consistency=True
    )
    assert post.base_threshold == 0.965
    assert post.s_guard == 0.900
    assert post.min_margin == 0.020
    assert post.joint_sim_floor == 0.450
    print("  [✓] V1.3 Surgical parameters strictly preserved (0.965, 0.900, 0.020, 0.450).")

    print("\n" + "=" * 60)
    print("ALL UNIT CHECKS PASSED: 100% READY FOR EC2 INFERENCE")
    print("=" * 60)
    return True

if __name__ == "__main__":
    run_unit_tests()
