import time
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

def _extract_sparse_topk(sims: sp.csr_matrix, k: int, min_score: float) -> List[Tuple[int, float]]:
    data = sims.data
    if len(data) == 0:
        return []
    indices = sims.indices
    mask = data > min_score
    if not np.any(mask):
        return []
    val_d = data[mask]
    val_i = indices[mask]
    n_val = len(val_d)
    if n_val <= k:
        top_order = np.argsort(-val_d, kind='stable')
    else:
        top_part = np.argpartition(-val_d, k)[:k]
        top_order = top_part[np.argsort(-val_d[top_part], kind='stable')]
    return [(int(val_i[i]), float(val_d[i])) for i in top_order]

class MultiViewCandidateRetriever:
    """Multi-view candidate retrieval engine combining Stage-5 InvertedIndexBlocking
    with high-speed sparse TF-IDF and char-ngram vector spaces.
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
        
        # Country -> Vectorizer & Transposed CSR Matrix (V x N)
        self.name_vectorizers: Dict[str, TfidfVectorizer] = {}
        self.name_matrices: Dict[str, sp.csr_matrix] = {}
        self.addr_vectorizers: Dict[str, TfidfVectorizer] = {}
        self.addr_matrices: Dict[str, sp.csr_matrix] = {}
        self.char_vectorizers: Dict[str, TfidfVectorizer] = {}
        self.char_matrices: Dict[str, sp.csr_matrix] = {}
        
        # Country -> List of target IDs corresponding to matrix columns
        self.country_target_ids: Dict[str, List[str]] = defaultdict(list)

    def fit_target_corpora(self, target_records: Dict[str, Tuple[str, str, str, str]], n_jobs: int = 1):
        """Fit TF-IDF vectorizers and transform target matrices partitioned by country.
        
        target_records: tid -> (norm_name, core_name, norm_addr, country)
        """
        import concurrent.futures
        
        # Index target IDs by country
        self.country_target_ids = defaultdict(list)
        for tid, (_, _, _, country) in target_records.items():
            c = country.strip() if country else "UNKNOWN"
            self.country_target_ids[c].append(tid)
            
        def _fit_country(c, tids):
            names = [target_records[tid][0] or " " for tid in tids]
            addrs = [target_records[tid][2] or " " for tid in tids]
            results = {}
            
            if self.enable_tfidf_name:
                v_name = TfidfVectorizer(max_features=50000, token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
                M_name = v_name.fit_transform(names)
                results['name'] = (v_name, M_name.T.tocsr())
                
            if self.enable_tfidf_addr:
                v_addr = TfidfVectorizer(max_features=50000, token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
                M_addr = v_addr.fit_transform(addrs)
                results['addr'] = (v_addr, M_addr.T.tocsr())
                
            if self.enable_tfidf_char:
                v_char = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), max_features=40000, dtype=np.float32)
                M_char = v_char.fit_transform(names)
                results['char'] = (v_char, M_char.T.tocsr())
                
            return c, results

        tasks = [(c, tids) for c, tids in self.country_target_ids.items() if len(tids) > 0]
        
        if n_jobs > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
                futures = [executor.submit(_fit_country, c, tids) for c, tids in tasks]
                for future in concurrent.futures.as_completed(futures):
                    c, res = future.result()
                    if 'name' in res:
                        self.name_vectorizers[c], self.name_matrices[c] = res['name']
                    if 'addr' in res:
                        self.addr_vectorizers[c], self.addr_matrices[c] = res['addr']
                    if 'char' in res:
                        self.char_vectorizers[c], self.char_matrices[c] = res['char']
        else:
            for c, tids in tasks:
                _, res = _fit_country(c, tids)
                if 'name' in res:
                    self.name_vectorizers[c], self.name_matrices[c] = res['name']
                if 'addr' in res:
                    self.addr_vectorizers[c], self.addr_matrices[c] = res['addr']
                if 'char' in res:
                    self.char_vectorizers[c], self.char_matrices[c] = res['char']

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
                sims = q_vec.dot(self.name_matrices[c])
                for idx, score in _extract_sparse_topk(sims, k, 0.15):
                    tid = tids[idx]
                    evidence[tid]["tfidf_name"] = score
                    if len(cand_set) < self.max_total_candidates:
                        cand_set.add(tid)
                            
        # 2. Address TF-IDF Top-K
        if self.enable_tfidf_addr and c in self.addr_vectorizers and n_addr:
            v_addr = self.addr_vectorizers[c]
            q_vec = v_addr.transform([n_addr])
            if q_vec.nnz > 0:
                sims = q_vec.dot(self.addr_matrices[c])
                for idx, score in _extract_sparse_topk(sims, k, 0.25):
                    tid = tids[idx]
                    evidence[tid]["tfidf_addr"] = score
                    if len(cand_set) < self.max_total_candidates:
                        cand_set.add(tid)
                            
        # 3. Char 3-gram TF-IDF Top-K
        if self.enable_tfidf_char and c in self.char_vectorizers and n_name:
            v_char = self.char_vectorizers[c]
            q_vec = v_char.transform([n_name])
            if q_vec.nnz > 0:
                sims = q_vec.dot(self.char_matrices[c])
                for idx, score in _extract_sparse_topk(sims, k, 0.30):
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
        """Bounded-memory country-sharded candidate retriever using fast sparse matrix operations."""
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
            has_name = self.enable_tfidf_name and (c in self.name_vectorizers)
            has_addr = self.enable_tfidf_addr and (c in self.addr_vectorizers)
            has_char = self.enable_tfidf_char and (c in self.char_vectorizers)

            v_name = self.name_vectorizers.get(c) if has_name else None
            M_name = self.name_matrices.get(c) if has_name else None
            
            v_addr = self.addr_vectorizers.get(c) if has_addr else None
            M_addr = self.addr_matrices.get(c) if has_addr else None
            
            v_char = self.char_vectorizers.get(c) if has_char else None
            M_char = self.char_matrices.get(c) if has_char else None

            for sid, rn, rc, ra, rco in s1_list:
                cand_set = final_cands[sid]
                
                # 1. Name TF-IDF Top-K
                if has_name and rn:
                    q_vec = v_name.transform([rn])
                    if q_vec.nnz > 0:
                        sims = q_vec.dot(M_name)
                        for idx, score in _extract_sparse_topk(sims, k, 0.15):
                            tid = tids[idx]
                            evidence_map[sid][tid]["tfidf_name"] = score
                            if len(cand_set) < self.max_total_candidates:
                                cand_set.add(tid)

                # 2. Address TF-IDF Top-K
                if has_addr and ra:
                    q_vec = v_addr.transform([ra])
                    if q_vec.nnz > 0:
                        sims = q_vec.dot(M_addr)
                        for idx, score in _extract_sparse_topk(sims, k, 0.25):
                            tid = tids[idx]
                            evidence_map[sid][tid]["tfidf_addr"] = score
                            if len(cand_set) < self.max_total_candidates:
                                cand_set.add(tid)

                # 3. Char 3-gram TF-IDF Top-K
                if has_char and rn:
                    q_vec = v_char.transform([rn])
                    if q_vec.nnz > 0:
                        sims = q_vec.dot(M_char)
                        for idx, score in _extract_sparse_topk(sims, k, 0.30):
                            tid = tids[idx]
                            evidence_map[sid][tid]["tfidf_char"] = score
                            if len(cand_set) < self.max_total_candidates:
                                cand_set.add(tid)

        return final_cands, evidence_map
