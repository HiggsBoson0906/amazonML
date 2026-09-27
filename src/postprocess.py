from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import numpy as np
from src.evaluate import evaluate_predictions

def optimize_decision_threshold(
    ground_truth: Dict[str, Set[str]],
    pair_predictions: List[Tuple[str, str, float]],
    s1_all_ids: List[str],
    thresholds: Optional[List[float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Search threshold grid to maximize official Macro F0.5 on the S1 universe."""
    if thresholds is None:
        thresholds = [0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995]
        
    s1_pairs = defaultdict(list)
    for s1_id, tid, prob in pair_predictions:
        s1_pairs[s1_id].append((tid, prob))
        
    best_thr = 0.980
    best_metrics = {}
    best_f05 = -1.0
    
    for thr in thresholds:
        preds = {}
        for s1_id in s1_all_ids:
            matches = {tid for tid, prob in s1_pairs.get(s1_id, []) if prob >= thr}
            preds[s1_id] = matches
            
        metrics = evaluate_predictions(ground_truth, preds)
        if metrics["macro_f05"] > best_f05:
            best_f05 = metrics["macro_f05"]
            best_thr = thr
            best_metrics = metrics
            
    return best_thr, best_metrics

class PostProcessor:
    """Post-processing and decision layer for singleton protection and target exclusivity."""
    def __init__(
        self,
        base_threshold: float = 0.980,
        s2_threshold: Optional[float] = None,
        s3_threshold: Optional[float] = None,
        min_margin: float = 0.0,
        enable_singleton_guard: bool = False,
        enable_global_consistency: bool = False,
    ):
        self.base_threshold = base_threshold
        self.s2_threshold = s2_threshold if s2_threshold is not None else base_threshold
        self.s3_threshold = s3_threshold if s3_threshold is not None else base_threshold
        self.min_margin = min_margin
        self.enable_singleton_guard = enable_singleton_guard
        self.enable_global_consistency = enable_global_consistency

    def apply(
        self,
        s1_ids: List[str],
        pair_probs: Dict[str, List[Tuple[str, float]]],
        candidate_sets: Dict[str, Set[str]],
    ) -> Dict[str, Set[str]]:
        """Apply thresholds, singleton safety filters, and target exclusivity conflict resolution.

        
        pair_probs: s1_id -> list of (target_id, prob)
        candidate_sets: s1_id -> set of candidate target_ids (guarantees subset invariant)
        """
        initial_matches: Dict[str, List[Tuple[str, float]]] = {}
        
        for s1_id in s1_ids:
            cands = candidate_sets.get(s1_id, set())
            pairs = pair_probs.get(s1_id, [])
            
            # Sort descending by probability
            sorted_pairs = sorted(pairs, key=lambda x: -x[1])
            selected = []
            
            for tid, p in sorted_pairs:
                # Invariant: must exist in candidate set
                if tid not in cands:
                    continue
                    
                # Source-specific threshold
                thr = self.s2_threshold if tid.startswith("S2-") else self.s3_threshold
                if p >= thr:
                    selected.append((tid, p))
                    
            if self.enable_singleton_guard and selected:
                # Singleton protection: if top match probability is weak or ambiguity is high
                top_p = selected[0][1]
                if len(selected) > 1:
                    margin = top_p - selected[1][1]
                    if margin < self.min_margin and top_p < 0.995:
                        # Ambiguous multi-candidate collision: keep only ultra-confident match
                        selected = [selected[0]] if top_p >= 0.992 else []
                        
            initial_matches[s1_id] = selected

        # Global Target Consistency (Exclusivity Check)
        if self.enable_global_consistency:
            # Check if any single target ID was matched to multiple distinct S1 entities
            target_to_s1 = defaultdict(list)
            for s1_id, matches in initial_matches.items():
                for tid, p in matches:
                    target_to_s1[tid].append((s1_id, p))
                    
            final_matches: Dict[str, Set[str]] = {s1_id: set() for s1_id in s1_ids}
            for tid, s1_list in target_to_s1.items():
                if len(s1_list) == 1:
                    final_matches[s1_list[0][0]].add(tid)
                else:
                    # Multiple S1 entities claimed the same target ID: assign greedily to the highest probability
                    s1_list.sort(key=lambda x: -x[1])
                    best_s1, best_p = s1_list[0]
                    final_matches[best_s1].add(tid)
        else:
            final_matches = {
                s1_id: {tid for tid, _ in matches}
                for s1_id, matches in initial_matches.items()
            }
            
        return final_matches
