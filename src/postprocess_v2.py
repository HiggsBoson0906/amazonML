"""
V2 Post-Processor — Eliminates redundant JaroWinkler computation
================================================================

The V1.3 postprocessor re-computes JaroWinkler similarity for every candidate pair.
Since we already compute name_jw and addr_jw during feature extraction (Stage B),
V2 passes these through to avoid redundant work.

All V1.3 decision logic is preserved exactly.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict


class PostProcessorV2:
    """V2 surgical post-processor that reuses feature-stage JW scores.
    
    Preserves exact V1.3 decision semantics:
    - base threshold 0.965
    - dynamic evidence gating (exact name, exact addr, postal+hnum)
    - joint similarity floor
    - singleton protection guard
    - competition margin gating
    - global target exclusivity conflict resolution
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
        pair_results: Dict[str, List[dict]],
    ) -> Dict[str, Set[str]]:
        """Apply V1.3 decision logic using pre-computed feature data.
        
        pair_results: s1_id -> list of {
            'tid': str,
            'prob': float,           # LightGBM probability
            'name_jw': float,        # from feature extraction
            'addr_jw': float,        # from feature extraction
            'exact_core': bool,      # s1.cn == tgt.cn
            'exact_addr': bool,      # s1.na == tgt.na
            'postal_match': bool,    # s1.pc == tgt.pc
            'hnum_match': bool,      # s1.hn == tgt.hn
        }
        """
        initial_matches: Dict[str, List[Tuple[str, float]]] = {}
        
        for s1_id in s1_ids:
            items = pair_results.get(s1_id, [])
            if not items:
                initial_matches[s1_id] = []
                continue
            
            # Sort by probability descending
            items.sort(key=lambda x: -x['prob'])
            
            selected = []
            for item in items:
                p = item['prob']
                joint_sim = item['name_jw'] * 0.6 + item['addr_jw'] * 0.4
                exact_name = item['exact_core']
                exact_addr = item['exact_addr']
                postal_match = item['postal_match']
                hnum_match = item['hnum_match']
                
                # Dynamic Evidence Gating (exact V1.3)
                if exact_name and (exact_addr or (postal_match and hnum_match)):
                    thr = 0.800
                elif exact_name:
                    thr = 0.875
                elif exact_addr and postal_match:
                    thr = 0.895
                elif joint_sim >= 0.88 and p >= 0.930:
                    thr = 0.930
                else:
                    thr = self.base_threshold
                
                if p >= thr and joint_sim >= self.joint_sim_floor:
                    selected.append({
                        'tid': item['tid'],
                        'p': p,
                        'joint_sim': joint_sim,
                        'exact_name': exact_name,
                        'exact_addr': exact_addr,
                    })
            
            # Enhanced Singleton Protection Guard (exact V1.3)
            if selected:
                top_item = selected[0]
                top_p = top_item['p']
                if len(selected) > 1:
                    margin = top_p - selected[1]['p']
                    if top_p < 0.980 and margin < self.min_margin and not top_item['exact_name']:
                        selected = []
                else:
                    if top_p < self.s_guard and not (top_item['exact_name'] or top_item['exact_addr']):
                        if top_item['joint_sim'] < 0.75:
                            selected = []
            
            initial_matches[s1_id] = [(it['tid'], it['p']) for it in selected]
        
        # Global Target Exclusivity Conflict Resolution (exact V1.3)
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
