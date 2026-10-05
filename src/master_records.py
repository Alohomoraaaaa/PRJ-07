"""
Master (Golden) Record Synthesis & Fine-Grained Field Provenance (PRJ-07).
Generates consolidated golden entity profiles via deterministic 5-tier conflict resolution,
populates the normalized entity_members junction table, records detailed field-level provenance,
and provides a unified master view surfacing ALL original raw & normalized columns.
"""

import re
import uuid
import json
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from src.config import CANONICAL_FIELDS


def resolve_canonical_field(
    field_name: str,
    field_entries: List[Dict[str, Any]],
    priority_source: str = "src_hr_db",
) -> Tuple[str, str, Dict[str, Any], Optional[str]]:
    """
    Deterministically resolve conflicting field values across records in an entity cluster.

    Resolution Strategy:
      1. Non-Null Rule: Filter out empty strings and nulls.
      2. Unanimous Consensus: If all values agree, select without ambiguity.
      3. Majority Rule: Select the most frequent value across contributing records.
      4. Source Authority Priority: If tied, prefer value from the primary trusted source.
      5. Completeness / Length Heuristic: Prefer more descriptive / standard valid strings.
      6. Deterministic Fallback: Lexicographical sort.
    """
    valid_entries = [
        e for e in field_entries
        if e.get("norm_value") and str(e["norm_value"]).strip() and str(e["norm_value"]).lower() not in ("nan", "none", "null", "undefined", "n/a", "na", "-", "")
    ]

    if not valid_entries:
        return "", "ALL_NULL", {}, None

    if len(valid_entries) == 1:
        win = valid_entries[0]
        return win["norm_value"], "SINGLE_NON_NULL", win, None

    # 2. Check for Unanimous Consensus
    distinct_values = list(set(e["norm_value"] for e in valid_entries))
    if len(distinct_values) == 1:
        return distinct_values[0], "UNANIMOUS_CONSENSUS", valid_entries[0], None

    # Distinct values exist: build conflict log
    formatted_entries = [f"{e['source_id']}:{e['norm_value']}" for e in valid_entries]
    conflict_summary = "Conflicting values: " + ", ".join(formatted_entries)

    # 3. Frequency Analysis (Majority Voting)
    val_counts = Counter(e["norm_value"] for e in valid_entries)
    max_freq = max(val_counts.values())
    top_values = [v for v, c in val_counts.items() if c == max_freq]

    if len(top_values) == 1:
        win_val = top_values[0]
        win_entry = next(e for e in valid_entries if e["norm_value"] == win_val)
        return win_val, "MAJORITY_VOTE", win_entry, conflict_summary

    # 4. Source Authority Priority
    for entry in valid_entries:
        if entry["norm_value"] in top_values and entry["source_id"] == priority_source:
            return entry["norm_value"], "SOURCE_AUTHORITY_PRIORITY", entry, conflict_summary

    # 5. Completeness / Length Heuristic for Text & Names
    candidate_entries = [e for e in valid_entries if e["norm_value"] in top_values]
    if field_name in ("name", "address", "company", "notes", "business_name", "category"):
        candidate_entries.sort(key=lambda x: (-len(str(x["norm_value"])), str(x["norm_value"])))
        win_entry = candidate_entries[0]
        return win_entry["norm_value"], "COMPLETENESS_LENGTH_TIE_BREAK", win_entry, conflict_summary

    # 6. Lexicographical Fallback
    candidate_entries.sort(key=lambda x: str(x["norm_value"]))
    win_entry = candidate_entries[0]
    return win_entry["norm_value"], "DETERMINISTIC_LEXICOGRAPHICAL", win_entry, conflict_summary


def synthesize_master_entities(
    clusters: Dict[str, List[str]],
    canonical_records_df: pd.DataFrame,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Synthesize master entities, entity membership junction entries, and field provenance records.
    Dynamically discovers and resolves ALL mapped canonical attributes and semantic fields
    across member records so that semantically equivalent fields are merged into ONE final golden attribute.

    Parameters:
        clusters: Dict mapping root_id -> list of record_uuids
        canonical_records_df: DataFrame containing all canonical_records

    Returns:
        (master_entities_list, entity_members_list, field_provenance_list)
    """
    master_entities = []
    entity_members = []
    field_provenance = []

    # Map record_uuid -> dict
    recs_by_uuid = {}
    for _, row in canonical_records_df.iterrows():
        r_uuid = row["record_uuid"]
        try:
            raw_dict = json.loads(row["raw_fields_json"]) if isinstance(row["raw_fields_json"], str) else (row["raw_fields_json"] or {})
        except Exception:
            raw_dict = {}
        try:
            norm_dict = json.loads(row["norm_fields_json"]) if isinstance(row["norm_fields_json"], str) else (row["norm_fields_json"] or {})
        except Exception:
            norm_dict = {}

        recs_by_uuid[r_uuid] = {
            "record_uuid": r_uuid,
            "source_id": row["source_id"],
            "source_record_id": row.get("source_record_id", ""),
            "raw_fields": raw_dict,
            "norm_fields": norm_dict,
        }

    prov_counter = 1
    sorted_roots = sorted(clusters.keys())

    for ent_idx, root_id in enumerate(sorted_roots, start=1):
        entity_id = f"ENT_{ent_idx:05d}"
        member_uuids = sorted(clusters[root_id])
        cluster_size = len(member_uuids)

        # Collect field candidates across all members: {canonical_attribute: [entry_dicts]}
        field_entries_map: Dict[str, List[Dict[str, Any]]] = {}

        # 1. Populate entity_members junction table and collect mapped & unmapped attributes
        for r_uuid in member_uuids:
            rec = recs_by_uuid.get(r_uuid)
            if not rec:
                continue
            entity_members.append({
                "entity_id": entity_id,
                "record_uuid": r_uuid,
                "source_id": rec["source_id"],
            })

            norm_fields = rec["norm_fields"]
            raw_fields = rec["raw_fields"]

            # Collect mapped canonical fields (each mapped concept consolidates all source variations)
            for c_field, norm_val in norm_fields.items():
                if not c_field or c_field == "unmapped":
                    continue
                if norm_val is not None and str(norm_val).strip() and str(norm_val).strip().lower() not in ("nan", "none", "null", "undefined", ""):
                    if c_field not in field_entries_map:
                        field_entries_map[c_field] = []
                    # Locate source raw column name for explainability and provenance
                    src_col = next((rk for rk, rv in raw_fields.items() if str(rv).strip() == str(norm_val).strip()), c_field)
                    field_entries_map[c_field].append({
                        "raw_value": raw_fields.get(src_col, norm_val),
                        "norm_value": str(norm_val).strip(),
                        "source_id": rec["source_id"],
                        "record_uuid": r_uuid,
                        "source_column": src_col,
                    })

        # 2. Resolve golden values & log field-level provenance dynamically for ALL attributes
        golden_attributes: Dict[str, str] = {}

        # Resolve all discovered attributes
        for c_field in sorted(field_entries_map.keys()):
            entries = field_entries_map[c_field]
            if not entries:
                continue
            res_val, res_rule, winning_entry, conflict_info = resolve_canonical_field(c_field, entries)
            if res_val:
                golden_attributes[c_field] = res_val
                prov_id = f"PROV_{prov_counter:07d}"
                prov_counter += 1
                field_provenance.append({
                    "provenance_id": prov_id,
                    "entity_id": entity_id,
                    "canonical_field": c_field,
                    "resolved_value": res_val,
                    "source_id": winning_entry.get("source_id", "unknown"),
                    "record_uuid": winning_entry.get("record_uuid", member_uuids[0]),
                    "source_column": winning_entry.get("source_column", c_field),
                    "raw_value": winning_entry.get("raw_value", res_val),
                    "normalized_value": res_val,
                    "resolution_rule": res_rule,
                    "conflict_details": conflict_info,
                })

        # Dynamically calculate quality score based on populated attributes
        total_attrs = len(field_entries_map)
        populated_attrs = len(golden_attributes)
        quality_score = round(populated_attrs / total_attrs, 2) if total_attrs > 0 else 1.0

        master_entities.append({
            "entity_id": entity_id,
            "cluster_size": cluster_size,
            "golden_name": golden_attributes.get("name") or golden_attributes.get("full_name") or golden_attributes.get("business_name") or golden_attributes.get("username", ""),
            "golden_email": golden_attributes.get("email") or golden_attributes.get("email_id", ""),
            "golden_phone": golden_attributes.get("phone") or golden_attributes.get("phone_number", ""),
            "golden_username": golden_attributes.get("username") or golden_attributes.get("handle", ""),
            "golden_member_id": golden_attributes.get("member_id") or golden_attributes.get("aadhaar") or golden_attributes.get("employee_id", ""),
            "golden_address": golden_attributes.get("address") or golden_attributes.get("city", ""),
            "golden_company": golden_attributes.get("company") or golden_attributes.get("organization", ""),
            "raw_attributes_json": json.dumps(golden_attributes),
            "entity_quality_score": quality_score,
        })

    return master_entities, entity_members, field_provenance


def get_unified_master_dataframe(
    db_manager: Any,
    where_clause: str = "1=1",
    params: Optional[List[Any]] = None,
) -> pd.DataFrame:
    """
    Retrieve unified master repository records with clean, consolidated semantic column headers.
    Guarantees:
      - ONE ROW = ONE RESOLVED ENTITY
      - ONE COLUMN = ONE UNIQUE SEMANTIC ATTRIBUTE (consolidating all mapped source variations)
      - Stable entity identifier, cluster size, and contributing source lineage
    """
    sql = f"""
        SELECT 
            entity_id,
            cluster_size,
            golden_name,
            golden_email,
            golden_phone,
            golden_username,
            golden_member_id,
            golden_address,
            golden_company,
            raw_attributes_json,
            entity_quality_score
        FROM master_entities
        WHERE {where_clause}
        ORDER BY cluster_size DESC, entity_id ASC
    """
    df_master = db_manager.execute_query(sql, params)
    if df_master.empty:
        return pd.DataFrame()

    # Retrieve source list for each entity
    entity_sources = {}
    sources_query = """
        SELECT em.entity_id, ds.source_name
        FROM entity_members em
        JOIN canonical_records cr ON em.record_uuid = cr.record_uuid
        JOIN data_sources ds ON cr.source_id = ds.source_id
    """
    try:
        sources_df = db_manager.execute_query(sources_query)
        for _, row in sources_df.iterrows():
            eid = row["entity_id"]
            sname = row["source_name"]
            if eid not in entity_sources:
                entity_sources[eid] = set()
            entity_sources[eid].add(sname)
    except Exception:
        pass

    # Parse golden attributes for every entity
    parsed_golden_list = []
    all_discovered_attributes = set()

    for raw_json in df_master["raw_attributes_json"]:
        try:
            d = json.loads(raw_json) if isinstance(raw_json, str) and raw_json else {}
        except Exception:
            d = {}
        parsed_golden_list.append(d)
        all_discovered_attributes.update(d.keys())

    # Establish intuitive semantic column ordering
    priority_order = [
        "name", "username", "email", "phone", "aadhaar", "city", "address",
        "company", "department", "role", "salary", "bio", "followers",
        "following", "platform", "rating", "review_count", "category",
        "latitude", "longitude", "notes"
    ]
    ordered_dynamic_cols = [c for c in priority_order if c in all_discovered_attributes]
    remaining_dynamic_cols = sorted([c for c in all_discovered_attributes if c not in priority_order])
    final_attr_cols = ordered_dynamic_cols + remaining_dynamic_cols

    # Build clean output rows: ONE ROW = ONE ENTITY, ONE COLUMN = ONE SEMANTIC ATTRIBUTE
    clean_rows = []
    for idx, row in df_master.iterrows():
        g_dict = parsed_golden_list[idx] if idx < len(parsed_golden_list) else {}
        eid = row["entity_id"]
        srcs = sorted(list(entity_sources.get(eid, [])))
        sources_str = ", ".join(srcs) if srcs else "1 Source"

        out_item = {
            "entity_id": eid,
            "cluster_size": row["cluster_size"],
            "sources": sources_str,
        }

        # Populate each discovered semantic attribute with its resolved golden value
        for attr in final_attr_cols:
            out_item[attr] = g_dict.get(attr, "")

        clean_rows.append(out_item)

    df_out = pd.DataFrame(clean_rows)

    # Filter out columns that are completely blank across all records (keep core cols)
    cols_to_keep = [
        c for c in df_out.columns
        if c in ["entity_id", "cluster_size", "sources"] or df_out[c].astype(str).str.strip().ne("").any()
    ]
    return df_out[cols_to_keep]
