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
        top_k_per_view: int = 45,
        max_total_candidates: int = 200,
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
        """Retrieve candidate set by taking union of Stage-5 blocking and TF-IDF top-K views."""
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
                top_idx = np.argsort(-sims, kind='stable')[:k]
                for idx in top_idx:
                    score = float(sims[idx])
                    if score > 0.15:
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
                top_idx = np.argsort(-sims, kind='stable')[:k]
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
                top_idx = np.argsort(-sims, kind='stable')[:k]
                for idx in top_idx:
                    score = float(sims[idx])
                    if score > 0.30:
                        tid = tids[idx]
                        evidence[tid]["tfidf_char"] = score
                        if len(cand_set) < self.max_total_candidates:
                            cand_set.add(tid)
                            
        return cand_set, evidence

    def retrieve_candidates_batch(
        self,
        s1_by_country: Dict[str, List[Tuple[str, str, str, str, str]]],
        blocking_cands_map: Dict[str, Set[str]],
    ) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, Dict[str, float]]]]:
        """Vectorized country-sharded batch candidate retriever using sparse matrix-matrix multiplication (GEMM)."""
        final_cands: Dict[str, Set[str]] = {}
        evidence_map: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

        for c, s1_list in s1_by_country.items():
            if not s1_list:
                continue
                
            tids = self.country_target_ids.get(c, [])
            n_tids = len(tids)
            
            # Initialize with blocking candidates
            for sid, rn, rc, ra, rco in s1_list:
                b_cands = blocking_cands_map.get(sid, set())
                final_cands[sid] = set(b_cands)
                for tid in b_cands:
                    evidence_map[sid][tid]["blocking"] = 1.0

            if n_tids == 0:
                continue

            k = min(self.top_k_per_view, n_tids)
            
            # 1. BATCH NAME TF-IDF
            if self.enable_tfidf_name and c in self.name_vectorizers:
                v_name = self.name_vectorizers[c]
                M_name = self.name_matrices[c]
                names = [item[1] if item[1] else " " for item in s1_list]
                Q_name = v_name.transform(names)
                sim_matrix = M_name.dot(Q_name.T).tocsc()
                
                for col_idx, (sid, rn, rc, ra, rco) in enumerate(s1_list):
                    if not rn:
                        continue
                    col = sim_matrix.getcol(col_idx)
                    if col.nnz == 0:
                        continue
                    row_indices = col.indices
                    row_data = col.data
                    
                    top_local = np.argsort(-row_data, kind='stable')[:k]
                    for loc in top_local:
                        score = float(row_data[loc])
                        if score > 0.15:
                            tid = tids[row_indices[loc]]
                            if len(final_cands[sid]) < self.max_total_candidates:
                                final_cands[sid].add(tid)
                            evidence_map[sid][tid]["tfidf_name"] = score

            # 2. BATCH ADDR TF-IDF
            if self.enable_tfidf_addr and c in self.addr_vectorizers:
                v_addr = self.addr_vectorizers[c]
                M_addr = self.addr_matrices[c]
                addrs = [item[3] if item[3] else " " for item in s1_list]
                Q_addr = v_addr.transform(addrs)
                sim_matrix = M_addr.dot(Q_addr.T).tocsc()
                
                for col_idx, (sid, rn, rc, ra, rco) in enumerate(s1_list):
                    if not ra:
                        continue
                    col = sim_matrix.getcol(col_idx)
                    if col.nnz == 0:
                        continue
                    row_indices = col.indices
                    row_data = col.data
                    
                    top_local = np.argsort(-row_data, kind='stable')[:k]
                    for loc in top_local:
                        score = float(row_data[loc])
                        if score > 0.25:
                            tid = tids[row_indices[loc]]
                            if len(final_cands[sid]) < self.max_total_candidates:
                                final_cands[sid].add(tid)
                            evidence_map[sid][tid]["tfidf_addr"] = score

            # 3. BATCH CHAR 3-GRAM TF-IDF
            if self.enable_tfidf_char and c in self.char_vectorizers:
                v_char = self.char_vectorizers[c]
                M_char = self.char_matrices[c]
                names = [item[1] if item[1] else " " for item in s1_list]
                Q_char = v_char.transform(names)
                sim_matrix = M_char.dot(Q_char.T).tocsc()
                
                for col_idx, (sid, rn, rc, ra, rco) in enumerate(s1_list):
                    if not rn:
                        continue
                    col = sim_matrix.getcol(col_idx)
                    if col.nnz == 0:
                        continue
                    row_indices = col.indices
                    row_data = col.data
                    
                    top_local = np.argsort(-row_data, kind='stable')[:k]
                    for loc in top_local:
                        score = float(row_data[loc])
                        if score > 0.30:
                            tid = tids[row_indices[loc]]
                            if len(final_cands[sid]) < self.max_total_candidates:
                                final_cands[sid].add(tid)
                            evidence_map[sid][tid]["tfidf_char"] = score

        return final_cands, evidence_map
