import time
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

class MultiViewCandidateRetriever:
    """Multi-view candidate retrieval engine combining Stage-5 InvertedIndexBlocking

    with sparse TF-IDF and char-ngram vector spaces.
    """
    def __init__(
        self,
        blocker,
        enable_tfidf_name: bool = True,
        enable_tfidf_addr: bool = True,
        enable_tfidf_char: bool = True,
        top_k_per_view: int = 40,
        max_total_candidates: int = 160,
    ):
        self.blocker = blocker
        self.enable_tfidf_name = enable_tfidf_name
        self.enable_tfidf_addr = enable_tfidf_addr
        self.enable_tfidf_char = enable_tfidf_char
        self.top_k_per_view = top_k_per_view
        self.max_total_candidates = max_total_candidates
        
        # Country -> Vectorizer & Matrix
        self.name_vectorizers: Dict[str, TfidfVectorizer] = {}
        self.name_matrices: Dict[str, sp.csr_matrix] = {}
        self.addr_vectorizers: Dict[str, TfidfVectorizer] = {}
        self.addr_matrices: Dict[str, sp.csr_matrix] = {}
        self.char_vectorizers: Dict[str, TfidfVectorizer] = {}
        self.char_matrices: Dict[str, sp.csr_matrix] = {}
        
        # Country -> List of target IDs corresponding to matrix rows
        self.country_target_ids: Dict[str, List[str]] = defaultdict(list)

    def fit_target_corpora(self, target_records: Dict[str, Tuple[str, str, str, str]]):
        """Fit TF-IDF vectorizers and transform target matrices partitioned by country.

        
        target_records: tid -> (norm_name, core_name, norm_addr, country)
        """
        # Group by country
        country_names = defaultdict(list)
        country_addrs = defaultdict(list)
        
        for tid, (n_name, c_name, n_addr, country) in target_records.items():
            c = country.strip() if country else "UNKNOWN"
            self.country_target_ids[c].append(tid)
            country_names[c].append(n_name or " ")
            country_addrs[c].append(n_addr or " ")
            
        for c, tids in self.country_target_ids.items():
            if len(tids) == 0:
                continue
            names = country_names[c]
            addrs = country_addrs[c]
            
            if self.enable_tfidf_name:
                v_name = TfidfVectorizer(max_features=50000, token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
                self.name_matrices[c] = v_name.fit_transform(names).tocsr()
                self.name_vectorizers[c] = v_name
                
            if self.enable_tfidf_addr:
                v_addr = TfidfVectorizer(max_features=50000, token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
                self.addr_matrices[c] = v_addr.fit_transform(addrs).tocsr()
                self.addr_vectorizers[c] = v_addr
                
            if self.enable_tfidf_char:
                v_char = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), max_features=40000, dtype=np.float32)
                self.char_matrices[c] = v_char.fit_transform(names).tocsr()
                self.char_vectorizers[c] = v_char

    def retrieve_candidates(
        self,
        s1_id: str,
        n_name: str,
        c_name: str,
        n_addr: str,
        country: str,
        blocking_cands: Optional[Set[str]] = None,
    ) -> Tuple[Set[str], Dict[str, Dict[str, float]]]:
        """Retrieve candidate set by taking union of Stage-5 blocking and TF-IDF top-K views.

        
        Returns:
            (final_candidate_ids, retrieval_evidence_dict)
            where retrieval_evidence_dict maps tid -> {view_name: score}
        """
        c = country.strip() if country else "UNKNOWN"
        cand_set: Set[str] = set(blocking_cands) if blocking_cands is not None else set()
        evidence: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Mark blocking candidates
        for tid in cand_set:
            evidence[tid]["blocking"] = 1.0
            
        tids = self.country_target_ids.get(c, [])
        n_tids = len(tids)
        if n_tids == 0:
            return cand_set, evidence
            
        k = min(self.top_k_per_view, n_tids)
        
        # 1. Name TF-IDF Top-K
        if self.enable_tfidf_name and c in self.name_vectorizers and n_name:
            v_name = self.name_vectorizers[c]
            q_vec = v_name.transform([n_name])
            if q_vec.nnz > 0:
                sims = self.name_matrices[c].dot(q_vec.T).toarray().ravel()
                if k < n_tids:
                    top_idx = np.argpartition(sims, -k)[-k:]
                    top_idx = top_idx[np.argsort(-sims[top_idx])]
                else:
                    top_idx = np.argsort(-sims)
                for idx in top_idx:
                    score = float(sims[idx])
                    if score > 0.15:  # meaningful similarity threshold
                        tid = tids[idx]
                        evidence[tid]["tfidf_name"] = score
                        if len(cand_set) < self.max_total_candidates:
                            cand_set.add(tid)
                            
        # 2. Address TF-IDF Top-K
        if self.enable_tfidf_addr and c in self.addr_vectorizers and n_addr:
            v_addr = self.addr_vectorizers[c]
            q_vec = v_addr.transform([n_addr])
            if q_vec.nnz > 0:
                sims = self.addr_matrices[c].dot(q_vec.T).toarray().ravel()
                if k < n_tids:
                    top_idx = np.argpartition(sims, -k)[-k:]
                    top_idx = top_idx[np.argsort(-sims[top_idx])]
                else:
                    top_idx = np.argsort(-sims)
                for idx in top_idx:
                    score = float(sims[idx])
                    if score > 0.25:
                        tid = tids[idx]
                        evidence[tid]["tfidf_addr"] = score
                        if len(cand_set) < self.max_total_candidates:
                            cand_set.add(tid)
                            
        # 3. Char 3-gram TF-IDF Top-K
        if self.enable_tfidf_char and c in self.char_vectorizers and n_name:
            v_char = self.char_vectorizers[c]
            q_vec = v_char.transform([n_name])
            if q_vec.nnz > 0:
                sims = self.char_matrices[c].dot(q_vec.T).toarray().ravel()
                if k < n_tids:
                    top_idx = np.argpartition(sims, -k)[-k:]
                    top_idx = top_idx[np.argsort(-sims[top_idx])]
                else:
                    top_idx = np.argsort(-sims)
                for idx in top_idx:
                    score = float(sims[idx])
                    if score > 0.30:
                        tid = tids[idx]
                        evidence[tid]["tfidf_char"] = score
                        if len(cand_set) < self.max_total_candidates:
                            cand_set.add(tid)
                            
        return cand_set, evidence
