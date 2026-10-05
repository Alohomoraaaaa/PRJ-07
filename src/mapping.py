"""
Deterministic Schema Mapping & Column Inspection Engine (Phase 6 & 7).
Uses canonical synonym lookup and RapidFuzz token matching to suggest field mappings
with explainable confidence scores and user review/override validation.
"""

import re
from typing import Dict, List, Tuple, Any, Optional
from rapidfuzz import fuzz
from src.config import CANONICAL_SYNONYMS, CANONICAL_FIELDS


class SchemaMappingEngine:
    """
    Deterministic schema inspector and mapper.
    Suggests canonical field mappings without requiring LLMs or embeddings.
    """

    def __init__(self, synonym_dict: Optional[Dict[str, List[str]]] = None):
        self.synonyms = synonym_dict or CANONICAL_SYNONYMS

    def clean_header_name(self, col_name: str) -> str:
        """Standardize column header string for matching."""
        if not col_name:
            return ""
        s = str(col_name).strip().lower()
        s = re.sub(r"[^a-z0-9_]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s

    def suggest_single_column_mapping(self, raw_col_name: str) -> Dict[str, Any]:
        """
        Suggest canonical field mapping for a single raw source column.
        Returns dictionary with suggested_canonical, confidence, and reason.
        """
        clean_col = self.clean_header_name(raw_col_name)

        if not clean_col:
            return {
                "raw_column": raw_col_name,
                "suggested_canonical": "unmapped",
                "confidence": 0.0,
                "reason": "Empty column name",
                "is_approved": False,
            }

        # Rule 1: Exact canonical field name match
        if clean_col in self.synonyms:
            return {
                "raw_column": raw_col_name,
                "suggested_canonical": clean_col,
                "confidence": 1.0,
                "reason": f"Exact canonical match on '{clean_col}'",
                "is_approved": False,
            }

        # Rule 2: Exact synonym dictionary match
        for canonical, syn_list in self.synonyms.items():
            if clean_col in syn_list:
                return {
                    "raw_column": raw_col_name,
                    "suggested_canonical": canonical,
                    "confidence": 1.0,
                    "reason": f"Exact synonym match ('{clean_col}' -> '{canonical}')",
                    "is_approved": False,
                }

        # Rule 3: Token / Substring / RapidFuzz similarity
        best_canonical = "unmapped"
        highest_score = 0.0
        best_reason = ""
        generic_tokens = {"user", "id", "no", "number", "code", "val", "data", "txt"}

        clean_tokens = set(clean_col.split("_"))
        distinctive_clean = clean_tokens - generic_tokens

        for canonical, syn_list in self.synonyms.items():
            for syn in syn_list:
                syn_tokens = set(syn.split("_"))
                distinctive_syn = syn_tokens - generic_tokens

                # 3a. Distinctive token intersection
                common_distinctive = distinctive_clean.intersection(distinctive_syn)
                if common_distinctive:
                    jaccard = len(clean_tokens.intersection(syn_tokens)) / len(clean_tokens.union(syn_tokens))
                    token_score = 0.90 + 0.08 * jaccard
                    if token_score > highest_score:
                        highest_score = token_score
                        best_canonical = canonical
                        best_reason = f"Keyword match on '{list(common_distinctive)[0]}' (Score: {token_score*100:.0f}%)"

                # 3b. RapidFuzz token similarity
                sim_ratio = fuzz.token_set_ratio(clean_col, syn)
                if sim_ratio >= 80:
                    normalized_score = round(sim_ratio / 100.0, 2)
                    if normalized_score > highest_score:
                        highest_score = normalized_score
                        best_canonical = canonical
                        best_reason = f"Semantic token similarity with '{syn}' (Score: {sim_ratio}%)"

                # 3c. Substring containment (if syn is at least 4 chars and distinctive)
                elif len(syn) >= 4 and syn not in generic_tokens and (syn in clean_col or clean_col in syn):
                    sub_score = 0.85 + min(0.08, 0.01 * len(syn))
                    if sub_score > highest_score:
                        highest_score = sub_score
                        best_canonical = canonical
                        best_reason = f"Substring match ('{clean_col}' contains '{syn}')"

        if highest_score >= 0.70:
            return {
                "raw_column": raw_col_name,
                "suggested_canonical": best_canonical,
                "confidence": highest_score,
                "reason": best_reason,
                "is_approved": False,
            }

        return {
            "raw_column": raw_col_name,
            "suggested_canonical": "unmapped",
            "confidence": 0.0,
            "reason": "No confident match found",
            "is_approved": False,
        }

    def suggest_mappings_for_table(self, column_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Inspect all columns in a table and return mapping proposals.
        """
        results = {}
        assigned_canonicals = set()

        # First pass: map exact and confident matches
        for col in column_names:
            proposal = self.suggest_single_column_mapping(col)
            results[col] = proposal

        return results

    def validate_user_mapping(
        self,
        user_mapping: Dict[str, str],
        source_columns: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        Validate user-approved mapping dictionary.
        Returns (is_valid, list_of_error_messages).
        """
        errors = []
        mapped_canonicals = set()

        for raw_col, target in user_mapping.items():
            if raw_col not in source_columns:
                errors.append(f"Column '{raw_col}' does not exist in source schema.")
            if target != "unmapped" and target not in CANONICAL_FIELDS:
                errors.append(f"Target '{target}' is not a valid canonical field.")
            if target != "unmapped":
                if target in mapped_canonicals:
                    errors.append(f"Multiple source columns mapped to the same canonical field '{target}'.")
                mapped_canonicals.add(target)

        # Check that at least one identifier is mapped
        identifier_mapped = any(
            CANONICAL_FIELDS.get(t, {}).get("is_identifier", False)
            for t in user_mapping.values()
            if t != "unmapped"
        )
        if not identifier_mapped:
            errors.append("At least one unique identifier (email, phone, username, member_id) must be mapped.")

        return len(errors) == 0, errors

    def check_cross_dataset_compatibility(
        self,
        dataset_mappings: Dict[str, Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Check whether multiple mapped datasets share common identity fields for entity resolution.
        Returns compatibility status, shared identifier list, and explainable message.
        """
        if len(dataset_mappings) < 2:
            return {
                "is_compatible": True,
                "shared_identifiers": [],
                "message": "Single dataset provided; duplicate resolution will occur internally.",
            }

        # Collect mapped identifier canonical fields per dataset
        dataset_identifiers: Dict[str, set] = {}
        for src_name, mapping in dataset_mappings.items():
            mapped_ids = {
                target for col, target in mapping.items()
                if target != "unmapped" and CANONICAL_FIELDS.get(target, {}).get("is_identifier", False)
            }
            dataset_identifiers[src_name] = mapped_ids

        # Compute pairwise intersection of identifiers
        all_srcs = list(dataset_mappings.keys())
        shared_any_pair = False
        all_shared_identifiers = set()

        for i in range(len(all_srcs)):
            for j in range(i + 1, len(all_srcs)):
                s1, s2 = all_srcs[i], all_srcs[j]
                inter = dataset_identifiers[s1].intersection(dataset_identifiers[s2])
                if inter:
                    shared_any_pair = True
                    all_shared_identifiers.update(inter)

        if not shared_any_pair:
            return {
                "is_compatible": False,
                "shared_identifiers": [],
                "message": (
                    "⚠️ No common identity fields (Email, Phone, Username, Member ID) were mapped across the datasets. "
                    "Records from disconnected datasets cannot be linked without shared identifiers, and will be preserved as standalone entities."
                ),
            }

        return {
            "is_compatible": True,
            "shared_identifiers": sorted(list(all_shared_identifiers)),
            "message": f"✅ Compatible identity linkers found across datasets: {', '.join([s.title() for s in all_shared_identifiers])}.",
        }


if __name__ == "__main__":
    engine = SchemaMappingEngine()
    test_cols = ["email_id", "contact_no", "full_name", "emp_code", "residence_addr", "random_notes_field"]
    suggestions = engine.suggest_mappings_for_table(test_cols)
    for col, res in suggestions.items():
        print(f"[{col}] -> {res['suggested_canonical']} (Confidence: {res['confidence']:.2f}) | {res['reason']}")

