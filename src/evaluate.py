from typing import Dict, List, Set, Tuple
import numpy as np

def compute_f05(precision: float, recall: float, beta: float = 0.5) -> float:
    """Compute F_beta score (default beta=0.5)."""
    if precision + recall == 0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)

def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
) -> Dict[str, float]:
    """Calculate macro-averaged precision, recall, F0.5 across all entities in ground_truth.

    
    Evaluation Rules from Challenge:
    - Macro-average: F0.5 calculated per Source 1 entity, then averaged across all S1 entities.
    - Singletons (0 true matches):
        - Correctly predicting empty set -> Precision=1.0, Recall=1.0, F0.5=1.0
        - Predicting any false match -> Precision=0.0, Recall=0.0, F0.5=0.0
    - Non-singletons:
        - Predicting empty set -> Precision=0.0, Recall=0.0, F0.5=0.0
        - Predicting matches -> standard Precision, Recall, F0.5
    """
    total_entities = len(ground_truth)
    if total_entities == 0:
        return {"macro_f05": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "singleton_acc": 0.0}
    
    f05_scores: List[float] = []
    precision_scores: List[float] = []
    recall_scores: List[float] = []
    
    singletons_total = 0
    singletons_correct = 0
    
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        
        # Singleton entity (no true matches)
        if len(true_set) == 0:
            singletons_total += 1
            if len(pred_set) == 0:
                singletons_correct += 1
                f05_scores.append(1.0)
                precision_scores.append(1.0)
                recall_scores.append(1.0)
            else:
                f05_scores.append(0.0)
                precision_scores.append(0.0)
                recall_scores.append(0.0)
            continue
            
        # Non-singleton entity
        if len(pred_set) == 0:
            f05_scores.append(0.0)
            precision_scores.append(0.0)
            recall_scores.append(0.0)
            continue
            
        true_positives = len(pred_set.intersection(true_set))
        if true_positives == 0:
            f05_scores.append(0.0)
            precision_scores.append(0.0)
            recall_scores.append(0.0)
            continue
            
        prec = true_positives / len(pred_set)
        rec = true_positives / len(true_set)
        f05 = compute_f05(prec, rec, beta=0.5)
        
        precision_scores.append(prec)
        recall_scores.append(rec)
        f05_scores.append(f05)
        
    singleton_acc = singletons_correct / singletons_total if singletons_total > 0 else 0.0
    
    return {
        "macro_f05": float(np.mean(f05_scores)),
        "macro_precision": float(np.mean(precision_scores)),
        "macro_recall": float(np.mean(recall_scores)),
        "singleton_accuracy": float(singleton_acc),
        "total_evaluated": total_entities,
    }

def evaluate_candidate_recall(
    ground_truth: Dict[str, Set[str]],
    candidates: Dict[str, Set[str]],
) -> Dict[str, float]:
    """Calculate the candidate recall ceiling (what percentage of all ground-truth matches

    were successfully captured by the blocking stage).
    """
    total_true_matches = 0
    captured_matches = 0
    candidate_counts = []
    
    for s1_id, true_set in ground_truth.items():
        cand_set = candidates.get(s1_id, set())
        candidate_counts.append(len(cand_set))
        
        if len(true_set) > 0:
            total_true_matches += len(true_set)
            captured_matches += len(cand_set.intersection(true_set))
            
    recall_ceiling = captured_matches / total_true_matches if total_true_matches > 0 else 0.0
    
    return {
        "candidate_recall_ceiling": float(recall_ceiling),
        "total_true_matches": total_true_matches,
        "captured_matches": captured_matches,
        "avg_candidates_per_s1": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "median_candidates_per_s1": float(np.median(candidate_counts)) if candidate_counts else 0.0,
    }
