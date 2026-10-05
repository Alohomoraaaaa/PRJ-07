"""
Hierarchical Matching & Progressive Enrichment Edge Logger (Phases 3 & 4).
Uses inverted identifier indexing and configurable matching rules to link records,
update the Union-Find graph, and record explainable enrichment edges.
Guarantees that NULL/BLANK/WHITESPACE values are never indexed or matched.
"""

import re
import json
from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np
import pandas as pd
from src.config import IDENTIFIER_HIERARCHY
from src.clustering import UnionFind


def is_blank_or_null(val: Any) -> bool:
    """
    Check whether a value is None, empty string, whitespace-only, or a null representation.
    Returns True for null/blank values, False for valid populated identifiers.
    """
    if val is None:
        return True
    if isinstance(val, float) and (np.isnan(val) or pd.isna(val)):
        return True
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "null", "undefined", "n/a", "na", "-", ""):
        return True
    return False


class HierarchicalMatcher:
    """
    Matches records across heterogeneous datasets using a configurable hierarchy of identifiers.
    Logs each successful link to the enrichment edges table.
    """

    def __init__(
        self,
        hierarchy: Optional[List[Dict[str, Any]]] = None,
        union_find: Optional[UnionFind] = None,
    ):
        self.hierarchy = hierarchy or IDENTIFIER_HIERARCHY
        self.uf = union_find or UnionFind()
        # In-memory inverted index for fast chunk processing:
        # { (identifier_type, normalized_value): [ (record_uuid, source_id) ] }
        self.inverted_index: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
        self.step_counter = 1

    def register_record_identifiers(
        self,
        record_uuid: str,
        source_id: str,
        norm_fields: Dict[str, str],
    ):
        """Index all valid (non-blank, non-null) identifiers for a record."""
        self.uf.add(record_uuid)

        for rule in self.hierarchy:
            ident_type = rule["identifier_type"]
            canon_field = rule.get("canonical_field")
            canon_fields = rule.get("canonical_fields")

            # Single field identifier rule
            if canon_field and canon_field in norm_fields:
                norm_val = norm_fields[canon_field]
                if not is_blank_or_null(norm_val):
                    key = (ident_type, str(norm_val).strip())
                    if key not in self.inverted_index:
                        self.inverted_index[key] = []
                    if (record_uuid, source_id) not in self.inverted_index[key]:
                        self.inverted_index[key].append((record_uuid, source_id))

            # Composite identifier rule (e.g. name + address)
            elif canon_fields:
                # All fields in composite rule must be strictly non-blank
                all_valid = all(
                    f in norm_fields and not is_blank_or_null(norm_fields[f])
                    for f in canon_fields
                )
                if all_valid:
                    comp_val = "|".join([str(norm_fields[f]).strip().lower() for f in canon_fields])
                    key = (ident_type, comp_val)
                    if key not in self.inverted_index:
                        self.inverted_index[key] = []
                    if (record_uuid, source_id) not in self.inverted_index[key]:
                        self.inverted_index[key].append((record_uuid, source_id))

    def match_and_link_record(
        self,
        record_uuid: str,
        source_id: str,
        norm_fields: Dict[str, str],
        import_batch_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Match a record against existing indexed records using the identifier hierarchy.
        Skips any blank/null values to prevent incorrect universal entity collapse.
        Performs Union-Find merging and returns generated enrichment edges.
        """
        discovered_edges = []
        self.uf.add(record_uuid)

        # Iterate through hierarchy rules by priority order
        for rule in sorted(self.hierarchy, key=lambda x: x["priority"]):
            ident_type = rule["identifier_type"]
            canon_field = rule.get("canonical_field")
            canon_fields = rule.get("canonical_fields")
            confidence = rule.get("confidence", 1.0)

            lookup_keys = []

            # 1. Single-attribute identifier matching
            if canon_field and canon_field in norm_fields:
                val = norm_fields[canon_field]
                if is_blank_or_null(val):
                    continue
                lookup_keys.append(((ident_type, str(val).strip()), str(val).strip(), confidence))

            # 2. Composite-attribute identifier matching (e.g. Name + Address)
            elif canon_fields:
                all_valid = all(
                    f in norm_fields and not is_blank_or_null(norm_fields[f])
                    for f in canon_fields
                )
                if not all_valid:
                    continue
                comp_val = "|".join([str(norm_fields[f]).strip().lower() for f in canon_fields])
                comp_conf = rule.get("confidence_threshold", 0.90)
                lookup_keys.append(((ident_type, comp_val), comp_val, comp_conf))

            # Perform lookups in inverted index
            for lookup_key, display_val, match_conf in lookup_keys:
                existing_matches = self.inverted_index.get(lookup_key, [])

                for match_uuid, match_source_id in existing_matches:
                    # Do not link to self
                    if match_uuid == record_uuid:
                        continue

                    # Union in disjoint-set graph
                    is_new_link = self.uf.union(record_uuid, match_uuid)

                    # Create enrichment edge
                    edge_id = f"EDGE_{self.step_counter:06d}"
                    edge_data = {
                        "edge_id": edge_id,
                        "record_uuid_a": record_uuid,
                        "record_uuid_b": match_uuid,
                        "source_id_a": source_id,
                        "source_id_b": match_source_id,
                        "identifier_type": ident_type,
                        "identifier_value": display_val,
                        "match_confidence": match_conf,
                        "import_batch_id": import_batch_id,
                        "step_order": self.step_counter,
                    }
                    self.step_counter += 1
                    discovered_edges.append(edge_data)

        # Register this record's non-blank identifiers for subsequent records to match against
        self.register_record_identifiers(record_uuid, source_id, norm_fields)

        return discovered_edges

    def replay_progressive_enrichment(
        self,
        seed_identifier_type: str,
        seed_value: str,
        edges_df: Any,
        records_df: Any,
    ) -> Dict[str, Any]:
        """
        Replay chronological progressive entity enrichment for a given seed identifier.
        """
        if is_blank_or_null(seed_value):
            return {
                "found": False,
                "message": "Invalid or blank seed identifier provided.",
                "steps": [],
                "linked_records": [],
            }

        # Clean seed value
        clean_seed = str(seed_value).strip().lower()

        # Find initial matching records from records table
        matching_recs_list = []
        if not records_df.empty:
            for _, r in records_df.iterrows():
                try:
                    norm_d = json.loads(r["norm_fields_json"]) if isinstance(r.get("norm_fields_json"), str) else (r.get("norm_fields_json") or {})
                except Exception:
                    norm_d = {}
                try:
                    raw_d = json.loads(r["raw_fields_json"]) if isinstance(r.get("raw_fields_json"), str) else (r.get("raw_fields_json") or {})
                except Exception:
                    raw_d = {}

                # Check norm_fields, raw_fields, and direct columns
                target_val = str(norm_d.get(seed_identifier_type) or raw_d.get(seed_identifier_type) or r.get(f"{seed_identifier_type}_norm") or r.get(seed_identifier_type) or "").strip()
                
                # Normalization matching
                if seed_identifier_type in ("email", "username"):
                    if target_val.lower() == clean_seed:
                        matching_recs_list.append(r)
                elif seed_identifier_type in ("phone", "aadhaar"):
                    digits_target = re.sub(r"\D", "", target_val)
                    digits_seed = re.sub(r"\D", "", str(seed_value))
                    if (digits_target and digits_target == digits_seed) or target_val == str(seed_value).strip():
                        matching_recs_list.append(r)
                else:
                    if target_val.lower() == clean_seed or str(seed_value).strip().lower() in target_val.lower():
                        matching_recs_list.append(r)

        if not matching_recs_list:
            return {
                "found": False,
                "message": f"No entity found matching {seed_identifier_type} = '{seed_value}'.",
                "steps": [],
                "linked_records": [],
            }

        first_rec = matching_recs_list[0]
        first_rec_uuid = first_rec["record_uuid"]
        root_entity_id = self.uf.find(first_rec_uuid)
        all_cluster_uuids = set(self.uf.get_clusters().get(root_entity_id, [first_rec_uuid]))
        matching_recs = pd.DataFrame(matching_recs_list)

        # Filter relevant edges in chronological order
        relevant_edges = []
        if not edges_df.empty:
            filtered = edges_df[
                edges_df["record_uuid_a"].isin(all_cluster_uuids) |
                edges_df["record_uuid_b"].isin(all_cluster_uuids)
            ].sort_values(by="step_order")
            relevant_edges = filtered.to_dict(orient="records")

        # Construct replay steps
        steps = []
        discovered_sources = set()
        discovered_identifiers: Dict[str, Set[str]] = {
            "email": set(), "phone": set(), "username": set(), "member_id": set(), "company": set(), "address": set()
        }

        # Step 0: Initial Seed Match
        initial_row = matching_recs.iloc[0]
        discovered_sources.add(initial_row["source_id"])
        for k in ["email_norm", "phone_norm", "username_norm", "member_id_norm", "company_norm", "address_norm"]:
            if k in initial_row and not is_blank_or_null(initial_row[k]):
                field_key = k.replace("_norm", "")
                discovered_identifiers[field_key].add(str(initial_row[k]))

        steps.append({
            "step_number": 1,
            "action": f"Seed Search Query ({seed_identifier_type} = '{seed_value}')",
            "source_dataset": initial_row["source_id"],
            "record_uuid": initial_row["record_uuid"],
            "details": f"Found initial match in {initial_row['source_id']} with record ID '{initial_row.get('source_record_id', '')}'",
            "discovered_fields": {k: list(v) for k, v in discovered_identifiers.items() if v},
        })

        # Step 1..N: Replay edges
        step_idx = 2
        for edge in relevant_edges:
            edge_type = edge["identifier_type"]
            edge_val = edge["identifier_value"]
            src_a = edge["source_id_a"]
            src_b = edge["source_id_b"]

            target_src = src_b if src_a in discovered_sources else src_a
            target_uuid = edge["record_uuid_b"] if src_a in discovered_sources else edge["record_uuid_a"]
            discovered_sources.add(src_a)
            discovered_sources.add(src_b)

            steps.append({
                "step_number": step_idx,
                "action": f"Multi-Hop Graph Traversal via [{edge_type.upper()}: {edge_val}]",
                "source_dataset": target_src,
                "record_uuid": target_uuid,
                "details": f"Discovered connection from {src_a} to {src_b} sharing {edge_type} '{edge_val}' (Confidence: {edge['match_confidence'] * 100:.0f}%)",
                "edge": edge,
            })
            step_idx += 1

        return {
            "found": True,
            "root_entity_id": root_entity_id,
            "total_cluster_size": len(all_cluster_uuids),
            "total_sources_linked": len(discovered_sources),
            "sources": list(discovered_sources),
            "steps": steps,
            "cluster_record_uuids": list(all_cluster_uuids),
        }
