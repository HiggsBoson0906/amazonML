from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
from rapidfuzz.distance import JaroWinkler

from src.evaluate import evaluate_predictions
from src.features import PrecomputedEntity
from src.ranking import extract_structured_address_components

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


class SurgicalPostProcessorV1_3:
    """Exact V1.3 Surgical Post-Processing and Decision Layer.
    
    Implements calibrated base thresholding (0.965), dynamic exact-evidence
    overrides, candidate competition margin gating, singleton protection,
    and global target exclusivity conflict resolution.
    """
    def __init__(
        self,
        base_threshold: float = 0.965,
        s_guard: float = 0.900,
        min_margin: float = 0.020,
        joint_sim_floor: float = 0.450,
        enable_global_consistency: bool = True,
    ):
        self.base_threshold = base_threshold
        self.s_guard = s_guard
        self.min_margin = min_margin
        self.joint_sim_floor = joint_sim_floor
        self.enable_global_consistency = enable_global_consistency

    def apply(
        self,
        s1_ids: List[str],
        pair_probs: Dict[str, List[Tuple[str, float]]],
        candidate_sets: Dict[str, Set[str]],
        s1_objects: Dict[str, PrecomputedEntity],
        target_lookup: Dict[str, PrecomputedEntity],
        s1_addr_comps: Optional[Dict[str, Any]] = None,
        tgt_addr_comps: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Set[str]]:
        """Apply exact V1.3 multi-tier surgical decision rules."""
        if s1_addr_comps is None:
            s1_addr_comps = {eid: extract_structured_address_components(e.norm_addr) for eid, e in s1_objects.items()}
        if tgt_addr_comps is None:
            tgt_addr_comps = {}

        initial_matches: Dict[str, List[Tuple[str, float]]] = {}

        for s1_id in s1_ids:
            cands = candidate_sets.get(s1_id, set())
            pairs = pair_probs.get(s1_id, [])
            if not pairs:
                initial_matches[s1_id] = []
                continue

            s1_obj = s1_objects[s1_id]
            s1_ac = s1_addr_comps.get(s1_id)
            if s1_ac is None:
                s1_ac = extract_structured_address_components(s1_obj.norm_addr)

            # Build enriched candidate list
            cand_items = []
            for tid, prob in pairs:
                if tid not in cands:
                    continue
                tgt_obj = target_lookup.get(tid)
                if not tgt_obj:
                    continue

                tgt_ac = tgt_addr_comps.get(tid)
                if tgt_ac is None:
                    if hasattr(tgt_obj, "postal_code"):
                        tgt_ac = {"postal_code": tgt_obj.postal_code, "house_num": tgt_obj.house_num, "digits": getattr(tgt_obj, "digits_set", set())}
                    else:
                        tgt_ac = extract_structured_address_components(tgt_obj.norm_addr)

                name_jw = JaroWinkler.similarity(s1_obj.norm_name, tgt_obj.norm_name)
                addr_jw = JaroWinkler.similarity(s1_obj.norm_addr, tgt_obj.norm_addr) if (s1_obj.norm_addr and tgt_obj.norm_addr) else 0.0
                joint_sim = name_jw * 0.6 + addr_jw * 0.4

                exact_name = bool(s1_obj.core_name and s1_obj.core_name == tgt_obj.core_name)
                exact_addr = bool(s1_obj.norm_addr and s1_obj.norm_addr == tgt_obj.norm_addr)
                postal_match = bool(s1_ac["postal_code"] and s1_ac["postal_code"] == tgt_ac["postal_code"])
                hnum_match = bool(s1_ac["house_num"] and s1_ac["house_num"] == tgt_ac["house_num"])

                cand_items.append({
                    "tid": tid,
                    "p": float(prob),
                    "name_jw": name_jw,
                    "addr_jw": addr_jw,
                    "joint_sim": joint_sim,
                    "exact_name": exact_name,
                    "exact_addr": exact_addr,
                    "postal_match": postal_match,
                    "hnum_match": hnum_match,
                })

            cand_items.sort(key=lambda x: -x["p"])

            selected = []
            for item in cand_items:
                p = item["p"]
                # Dynamic Evidence Gating
                if item["exact_name"] and (item["exact_addr"] or (item["postal_match"] and item["hnum_match"])):
                    thr = 0.800
                elif item["exact_name"]:
                    thr = 0.875
                elif item["exact_addr"] and item["postal_match"]:
                    thr = 0.895
                elif item["joint_sim"] >= 0.88 and p >= 0.930:
                    thr = 0.930
                else:
                    thr = self.base_threshold

                if p >= thr and item["joint_sim"] >= self.joint_sim_floor:
                    selected.append(item)

            # Enhanced Singleton Protection Guard
            if selected:
                top_item = selected[0]
                top_p = top_item["p"]
                if len(selected) > 1:
                    margin = top_p - selected[1]["p"]
                    if top_p < 0.980 and margin < self.min_margin and not top_item["exact_name"]:
                        selected = []
                else:
                    if top_p < self.s_guard and not (top_item["exact_name"] or top_item["exact_addr"]):
                        if top_item["joint_sim"] < 0.75:
                            selected = []

            initial_matches[s1_id] = [(it["tid"], it["p"]) for it in selected]

        # Global Target Exclusivity Conflict Resolution
        if self.enable_global_consistency:
            target_to_s1 = defaultdict(list)
            for s1_id, matches in initial_matches.items():
                for tid, p in matches:
                    target_to_s1[tid].append((s1_id, p))

            final_matches = {s1_id: set() for s1_id in s1_ids}
            for tid, s1_list in target_to_s1.items():
                if len(s1_list) == 1:
                    final_matches[s1_list[0][0]].add(tid)
                else:
                    s1_list.sort(key=lambda x: -x[1])
                    best_s1, _ = s1_list[0]
                    final_matches[best_s1].add(tid)
        else:
            final_matches = {
                s1_id: {tid for tid, _ in matches}
                for s1_id, matches in initial_matches.items()
            }

        return final_matches


# Backward compatibility alias
PostProcessor = SurgicalPostProcessorV1_3
