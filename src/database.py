"""
DuckDB Database Manager & Relational Schema (Single-Process Thread-Safe).
Manages connection pooling, atomic chunk transactions, and query execution for
multi-database entity resolution, provenance logging, and master repository tables.
"""

import os
import json
import threading
import duckdb
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from src.config import DEFAULT_DB_PATH, VALID_JOB_STATUSES


class DatabaseManager:
    """
    Singleton thread-safe DuckDB connection manager for single-process Streamlit deployments.
    Guards write transactions with a re-entrant lock to prevent file lock contention and deadlocks.
    """

    _instance: Optional["DatabaseManager"] = None
    _lock = threading.RLock()

    def __new__(cls, db_path: str = DEFAULT_DB_PATH):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DatabaseManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        if self._initialized:
            return
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.write_lock = threading.RLock()
        try:
            self.conn = duckdb.connect(self.db_path)
        except Exception as e:
            if "already open" in str(e).lower():
                raise RuntimeError(
                    f"⚠️ DuckDB Database '{self.db_path}' is currently open in another process (e.g. Streamlit app). "
                    "Please stop the running Streamlit server (Ctrl+C in its terminal) or use the Streamlit web dashboard directly."
                ) from e
            raise e
        self.init_schema()
        self._initialized = True

    def init_schema(self):
        """Initialize relational database schema with indices and junction tables."""
        with self.write_lock:
            # 1. Data Sources Catalog
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS data_sources (
                source_id VARCHAR PRIMARY KEY,
                source_name VARCHAR NOT NULL,
                file_format VARCHAR NOT NULL,
                is_multipart BOOLEAN DEFAULT FALSE,
                total_parts INTEGER DEFAULT 1,
                total_raw_records INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 2. Dataset Parts Catalog
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS dataset_parts (
                part_id VARCHAR PRIMARY KEY,
                source_id VARCHAR NOT NULL,
                part_number INTEGER NOT NULL,
                file_path VARCHAR NOT NULL,
                record_count INTEGER DEFAULT 0,
                status VARCHAR DEFAULT 'PENDING'
            );
            """)

            # 3. Canonical Records (Dual Storage: Raw + Normalized)
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS canonical_records (
                record_uuid VARCHAR PRIMARY KEY,
                source_id VARCHAR NOT NULL,
                part_id VARCHAR NOT NULL,
                source_record_id VARCHAR,
                import_batch_id VARCHAR NOT NULL,
                raw_fields_json VARCHAR NOT NULL,
                norm_fields_json VARCHAR NOT NULL,
                email_norm VARCHAR,
                phone_norm VARCHAR,
                username_norm VARCHAR,
                member_id_norm VARCHAR,
                name_norm VARCHAR,
                company_norm VARCHAR,
                address_norm VARCHAR,
                is_unmatched BOOLEAN DEFAULT TRUE,
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 4. Inverted Identifier Index
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS identifier_index (
                index_id VARCHAR PRIMARY KEY,
                identifier_type VARCHAR NOT NULL,
                identifier_value VARCHAR NOT NULL,
                record_uuid VARCHAR NOT NULL,
                source_id VARCHAR NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ident_lookup ON identifier_index(identifier_type, identifier_value);")

            # 5. Enrichment Edges (Authoritative Union-Find Audit Trail)
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS enrichment_edges (
                edge_id VARCHAR PRIMARY KEY,
                record_uuid_a VARCHAR NOT NULL,
                record_uuid_b VARCHAR NOT NULL,
                source_id_a VARCHAR NOT NULL,
                source_id_b VARCHAR NOT NULL,
                identifier_type VARCHAR NOT NULL,
                identifier_value VARCHAR NOT NULL,
                match_confidence DOUBLE DEFAULT 1.0,
                import_batch_id VARCHAR NOT NULL,
                step_order INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 6. Master Entities (Golden Repository)
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS master_entities (
                entity_id VARCHAR PRIMARY KEY,
                cluster_size INTEGER DEFAULT 1,
                golden_name VARCHAR,
                golden_email VARCHAR,
                golden_phone VARCHAR,
                golden_username VARCHAR,
                golden_member_id VARCHAR,
                golden_address VARCHAR,
                golden_company VARCHAR,
                raw_attributes_json VARCHAR,
                entity_quality_score DOUBLE DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 7. Entity Members (Normalized Junction Table)
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS entity_members (
                entity_id VARCHAR NOT NULL,
                record_uuid VARCHAR NOT NULL,
                source_id VARCHAR NOT NULL,
                PRIMARY KEY (entity_id, record_uuid)
            );
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_mem ON entity_members(entity_id);")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_rec_mem ON entity_members(record_uuid);")

            # 8. Field Provenance (Fine-Grained Attribute Lineage)
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS field_provenance (
                provenance_id VARCHAR PRIMARY KEY,
                entity_id VARCHAR NOT NULL,
                canonical_field VARCHAR NOT NULL,
                resolved_value VARCHAR NOT NULL,
                source_id VARCHAR NOT NULL,
                record_uuid VARCHAR NOT NULL,
                source_column VARCHAR NOT NULL,
                raw_value VARCHAR NOT NULL,
                normalized_value VARCHAR NOT NULL,
                resolution_rule VARCHAR NOT NULL,
                conflict_details VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_field_prov ON field_provenance(entity_id);")

            # 9. Job Runs & Resumability Checkpoints
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS job_runs (
                job_id VARCHAR PRIMARY KEY,
                source_id VARCHAR,
                job_type VARCHAR NOT NULL,
                current_stage VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                total_records INTEGER DEFAULT 0,
                processed_records INTEGER DEFAULT 0,
                last_checkpoint_chunk INTEGER DEFAULT 0,
                error_message VARCHAR,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP
            );
            """)

    def execute_query(self, sql: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
        """Execute read query and return pandas DataFrame."""
        try:
            if params:
                return self.conn.execute(sql, params).df()
            return self.conn.execute(sql).df()
        except Exception as e:
            print(f"[DB ERROR] Query failed: {sql} | Error: {e}")
            return pd.DataFrame()

    def execute_scalar(self, sql: str, params: Optional[List[Any]] = None) -> Any:
        """Execute query returning a single scalar value."""
        try:
            if params:
                res = self.conn.execute(sql, params).fetchone()
            else:
                res = self.conn.execute(sql).fetchone()
            return res[0] if res else None
        except Exception as e:
            print(f"[DB ERROR] Scalar query failed: {sql} | Error: {e}")
            return None

    def execute_write(self, sql: str, params: Optional[List[Any]] = None):
        """Execute a single write query guarded by write lock."""
        with self.write_lock:
            if params:
                self.conn.execute(sql, params)
            else:
                self.conn.execute(sql)

    def reset_database(self):
        """Drop all tables and recreate clean schema (for fresh demos)."""
        with self.write_lock:
            tables = [
                "field_provenance",
                "entity_members",
                "master_entities",
                "enrichment_edges",
                "identifier_index",
                "canonical_records",
                "dataset_parts",
                "data_sources",
                "job_runs",
            ]
            for tbl in tables:
                self.conn.execute(f"DROP TABLE IF EXISTS {tbl}")
            self.init_schema()

    def get_system_statistics(self) -> Dict[str, Any]:
        """Fetch real-time analytical KPI metrics across the unified repository."""
        total_sources = self.execute_scalar("SELECT COUNT(*) FROM data_sources") or 0
        total_records = self.execute_scalar("SELECT COUNT(*) FROM canonical_records") or 0
        total_entities = self.execute_scalar("SELECT COUNT(*) FROM master_entities") or 0
        total_edges = self.execute_scalar("SELECT COUNT(*) FROM enrichment_edges") or 0
        total_singletons = self.execute_scalar("SELECT COUNT(*) FROM master_entities WHERE cluster_size = 1") or 0
        total_linked = self.execute_scalar("SELECT COUNT(*) FROM master_entities WHERE cluster_size > 1") or 0

        avg_cluster = self.execute_scalar("SELECT AVG(cluster_size) FROM master_entities") or 1.0

        return {
            "total_sources": total_sources,
            "total_records": total_records,
            "total_entities": total_entities,
            "total_edges": total_edges,
            "total_enrichment_edges": total_edges,
            "total_singletons": total_singletons,
            "total_linked_entities": total_linked,
            "average_cluster_size": round(float(avg_cluster), 2),
        }


def get_db_manager(db_path: str = DEFAULT_DB_PATH) -> DatabaseManager:
    """Helper to access singleton DatabaseManager."""
    return DatabaseManager(db_path)
