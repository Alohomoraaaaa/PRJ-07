"""
End-to-End Entity Resolution & Ingestion Pipeline Orchestrator (PRJ-07).
Coordinates dynamic schema inspection, field normalization, inverted index matching,
authoritative Union-Find graph clustering, and master entity synthesis for arbitrary user datasets.
"""

import os
import sys
import json
import uuid
import time
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import DEFAULT_DB_PATH, DEMO_DATA_DIR, JOB_STATUS_COMPLETED
from src.demo_datasets import generate_demonstration_suite
from src.mapping import SchemaMappingEngine
from src.normalizer import DataNormalizer
from src.clustering import UnionFind
from src.matcher import HierarchicalMatcher, is_blank_or_null
from src.master_records import synthesize_master_entities, get_unified_master_dataframe
from src.database import get_db_manager, DatabaseManager


def run_dynamic_user_pipeline(
    datasets_spec: List[Dict[str, Any]],
    db_path: str = DEFAULT_DB_PATH,
    reset_db: bool = True,
) -> Dict[str, Any]:
    """
    Execute generalized entity resolution over arbitrary user-provided datasets.

    Parameters:
        datasets_spec: List of dataset dictionaries containing:
            - 'source_id': unique identifier (e.g. 'src_salesforce')
            - 'source_name': user-facing display name (e.g. 'Customers_Export.csv')
            - 'dataframe': pd.DataFrame with raw tabular rows
            - 'mapping': Dict[str, str] mapping raw column names to canonical target fields
            - 'is_multipart': bool (optional, default False)
            - 'part_number': int (optional, default 1)
        db_path: Path to target DuckDB database file
        reset_db: Whether to reset repository tables before ingesting (default True)

    Returns:
        Summary statistics dictionary with counts and performance metrics.
    """
    start_time = time.time()
    db = get_db_manager(db_path)

    if reset_db:
        db.reset_database()

    normalizer = DataNormalizer()
    uf = UnionFind()
    matcher = HierarchicalMatcher(union_find=uf)

    # If incremental (reset_db is False), recover existing UnionFind state
    if not reset_db:
        existing_recs = db.execute_query("SELECT record_uuid FROM canonical_records")
        existing_uuids = existing_recs["record_uuid"].tolist() if not existing_recs.empty else []
        existing_edges = db.execute_query("SELECT record_uuid_a, record_uuid_b FROM enrichment_edges ORDER BY step_order ASC")
        edge_tuples = [(r["record_uuid_a"], r["record_uuid_b"]) for _, r in existing_edges.iterrows()] if not existing_edges.empty else []
        uf.rebuild_from_persisted_edges(existing_uuids, edge_tuples)

    all_canonical_records: List[Dict[str, Any]] = []
    all_enrichment_edges: List[Dict[str, Any]] = []
    all_index_entries: List[Dict[str, Any]] = []

    # Get current max sequence counters from DB
    existing_rec_count = db.execute_scalar("SELECT COUNT(*) FROM canonical_records") or 0
    existing_idx_count = db.execute_scalar("SELECT COUNT(*) FROM identifier_index") or 0
    existing_edge_count = db.execute_scalar("SELECT COUNT(*) FROM enrichment_edges") or 0

    rec_counter = existing_rec_count + 1
    idx_counter = existing_idx_count + 1
    matcher.step_counter = existing_edge_count + 1
    batch_id = f"BATCH_{int(time.time())}"

    # Ingest and process each user dataset
    for src in datasets_spec:
        source_id = src["source_id"]
        source_name = src["source_name"]
        df_data = src["dataframe"]
        mapping = src.get("mapping", {})
        is_multipart = src.get("is_multipart", False)
        part_number = src.get("part_number", 1)
        total_parts = src.get("total_parts", 1)
        part_id = f"part_{source_id}_{part_number:02d}"

        # Insert or update data_sources catalog
        db.execute_write(
            """
            INSERT OR REPLACE INTO data_sources (
                source_id, source_name, file_format, is_multipart, total_parts, total_raw_records
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [source_id, source_name, "CSV", is_multipart, total_parts, len(df_data)]
        )

        # Insert dataset_parts catalog
        db.execute_write(
            """
            INSERT OR REPLACE INTO dataset_parts (
                part_id, source_id, part_number, file_path, record_count, status
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [part_id, source_id, part_number, source_name, len(df_data), "INGESTED"]
        )

        # Process and normalize rows
        for row_idx, raw_row in df_data.iterrows():
            raw_dict = raw_row.to_dict()
            rec_uuid = f"REC_{rec_counter:08d}"
            rec_counter += 1

            raw_fields, norm_fields, validity = normalizer.process_raw_record(raw_dict, mapping)

            # Match against existing inverted index and perform Union-Find
            edges = matcher.match_and_link_record(
                record_uuid=rec_uuid,
                source_id=source_id,
                norm_fields=norm_fields,
                import_batch_id=batch_id,
            )
            all_enrichment_edges.extend(edges)

            # Extract source row identifier if present, else synthesize
            src_rec_id = str(
                raw_dict.get("id") or
                raw_dict.get("record_id") or
                raw_dict.get("cust_id") or
                raw_dict.get("employee_id") or
                raw_dict.get("member_id") or
                f"ROW_{row_idx+1}"
            )

            # Canonical record entry
            all_canonical_records.append({
                "record_uuid": rec_uuid,
                "source_id": source_id,
                "part_id": part_id,
                "source_record_id": src_rec_id,
                "import_batch_id": batch_id,
                "raw_fields_json": json.dumps(raw_fields),
                "norm_fields_json": json.dumps(norm_fields),
                "email_norm": norm_fields.get("email", ""),
                "phone_norm": norm_fields.get("phone", ""),
                "username_norm": norm_fields.get("username", ""),
                "member_id_norm": norm_fields.get("member_id", ""),
                "name_norm": norm_fields.get("name", ""),
                "company_norm": norm_fields.get("company", ""),
                "address_norm": norm_fields.get("address", ""),
                "is_unmatched": True,
            })

            # Inverted index entries for all populated identifiers (skips blank/null values)
            for ident_col in ["email", "phone", "username", "member_id"]:
                val = norm_fields.get(ident_col)
                if not is_blank_or_null(val):
                    all_index_entries.append({
                        "index_id": f"IDX_{idx_counter:08d}",
                        "identifier_type": ident_col,
                        "identifier_value": str(val).strip(),
                        "record_uuid": rec_uuid,
                        "source_id": source_id,
                    })
                    idx_counter += 1

    # Convert to DataFrames for batch DuckDB registration
    df_canon = pd.DataFrame(all_canonical_records)
    df_edges = pd.DataFrame(all_enrichment_edges)
    df_idx = pd.DataFrame(all_index_entries)

    # Persist Canonical Records
    if not df_canon.empty:
        db.conn.register("tmp_canon", df_canon)
        cols_canon = ", ".join(df_canon.columns.tolist())
        db.conn.execute(f"INSERT INTO canonical_records ({cols_canon}) SELECT * FROM tmp_canon")
        db.conn.unregister("tmp_canon")

    # Persist Inverted Index
    if not df_idx.empty:
        db.conn.register("tmp_idx", df_idx)
        cols_idx = ", ".join(df_idx.columns.tolist())
        db.conn.execute(f"INSERT INTO identifier_index ({cols_idx}) SELECT * FROM tmp_idx")
        db.conn.unregister("tmp_idx")

    # Persist Enrichment Edges
    if not df_edges.empty:
        db.conn.register("tmp_edges", df_edges)
        cols_edges = ", ".join(df_edges.columns.tolist())
        db.conn.execute(f"INSERT INTO enrichment_edges ({cols_edges}) SELECT * FROM tmp_edges")
        db.conn.unregister("tmp_edges")

    # Fetch ALL canonical records from database to synthesize complete master entities
    all_canon_df = db.execute_query("SELECT * FROM canonical_records")

    # Graph Clusters -> Master Records + entity_members + field_provenance
    clusters = uf.get_clusters()
    master_entities, entity_members, field_provenance = synthesize_master_entities(clusters, all_canon_df)

    df_master = pd.DataFrame(master_entities)
    df_members = pd.DataFrame(entity_members)
    df_prov = pd.DataFrame(field_provenance)

    # Replace master tables
    db.execute_write("DELETE FROM master_entities")
    db.execute_write("DELETE FROM entity_members")
    db.execute_write("DELETE FROM field_provenance")

    if not df_master.empty:
        db.conn.register("tmp_master", df_master)
        cols_master = ", ".join(df_master.columns.tolist())
        db.conn.execute(f"INSERT INTO master_entities ({cols_master}) SELECT * FROM tmp_master")
        db.conn.unregister("tmp_master")

    if not df_members.empty:
        db.conn.register("tmp_members", df_members)
        cols_members = ", ".join(df_members.columns.tolist())
        db.conn.execute(f"INSERT INTO entity_members ({cols_members}) SELECT * FROM tmp_members")
        db.conn.unregister("tmp_members")

    if not df_prov.empty:
        db.conn.register("tmp_prov", df_prov)
        cols_prov = ", ".join(df_prov.columns.tolist())
        db.conn.execute(f"INSERT INTO field_provenance ({cols_prov}) SELECT * FROM tmp_prov")
        db.conn.unregister("tmp_prov")

    # Update is_unmatched flag in canonical_records for linked records
    db.execute_write("""
    UPDATE canonical_records
    SET is_unmatched = FALSE
    WHERE record_uuid IN (
        SELECT record_uuid FROM entity_members
        WHERE entity_id IN (SELECT entity_id FROM master_entities WHERE cluster_size > 1)
    )
    """)

    elapsed = round(time.time() - start_time, 2)
    stats = db.get_system_statistics()
    stats["elapsed_seconds"] = elapsed

    # Add duplicate resolution metric
    total_raw = stats["total_records"]
    total_master = stats["total_entities"]
    duplicates_resolved = max(0, total_raw - total_master)
    reduction_pct = round((duplicates_resolved / total_raw * 100.0), 1) if total_raw > 0 else 0.0
    stats["duplicates_resolved"] = duplicates_resolved
    stats["reduction_percentage"] = reduction_pct

    return stats


def run_full_demonstration_pipeline(
    db_path: str = DEFAULT_DB_PATH,
    data_dir: str = DEMO_DATA_DIR,
    include_incremental_d: bool = False,
) -> Dict[str, Any]:
    """
    Execute demonstration pipeline using customer_a.csv and customer_b.csv.
    Delegates to the generalized run_dynamic_user_pipeline.
    """
    demo_files = generate_demonstration_suite(data_dir)
    mapping_engine = SchemaMappingEngine()

    df_a = pd.read_csv(demo_files["customer_a"])
    df_b = pd.read_csv(demo_files["customer_b"])

    map_a = {c: res["suggested_canonical"] for c, res in mapping_engine.suggest_mappings_for_table(df_a.columns.tolist()).items()}
    map_b = {c: res["suggested_canonical"] for c, res in mapping_engine.suggest_mappings_for_table(df_b.columns.tolist()).items()}

    spec = [
        {
            "source_id": "src_cust_a",
            "source_name": "customer_a.csv",
            "dataframe": df_a,
            "mapping": map_a,
            "is_multipart": False,
            "part_number": 1,
            "total_parts": 1,
        },
        {
            "source_id": "src_cust_b",
            "source_name": "customer_b.csv",
            "dataframe": df_b,
            "mapping": map_b,
            "is_multipart": False,
            "part_number": 1,
            "total_parts": 1,
        },
    ]

    return run_dynamic_user_pipeline(datasets_spec=spec, db_path=db_path, reset_db=True)


if __name__ == "__main__":
    res = run_full_demonstration_pipeline()
    print("=" * 60)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    for k, v in res.items():
        print(f"  • {k}: {v}")
