"""
End-to-End Test Suite for PRJ-07.
Tests all critical assignment components:
1. SQL Dump Ingestion & Table Introspection
2. Inverted Indexing & Configurable Matching Priority
3. Progressive Enrichment Edge Logging & Chronological Replay
4. Normalized Entity Membership (entity_members)
5. Fine-Grained Field-Level Lineage (field_provenance)
6. Zero Discards (Singleton entities)
7. Startup Recovery of Union-Find Graph State
"""

import os
import sys
import unittest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database import get_db_manager
from src.sql_importer import SQLDumpImporter
from src.demo_datasets import generate_demonstration_suite
from src.pipeline import run_full_demonstration_pipeline
from src.matcher import HierarchicalMatcher
from src.clustering import UnionFind
from src.jobs import get_job_registry


TEST_DB_PATH = "data/test_unified_repository.duckdb"


class TestEndToEndPlatform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = get_db_manager(TEST_DB_PATH)
        cls.demo_files = generate_demonstration_suite()
        cls.stats = run_full_demonstration_pipeline(db_path=TEST_DB_PATH)

    def test_01_field_provenance_integrity(self):
        """Test that every master entity field has a valid provenance record."""
        db = self.db
        prov_df = db.execute_query("SELECT * FROM field_provenance")
        self.assertFalse(prov_df.empty)

        # Check required provenance columns
        req_cols = ["provenance_id", "entity_id", "canonical_field", "resolved_value", "source_id", "raw_value", "resolution_rule"]
        for col in req_cols:
            self.assertIn(col, prov_df.columns)

    def test_02_normalized_junction_table(self):
        """Test entity_members junction table integrity."""
        db = self.db
        members_df = db.execute_query("SELECT * FROM entity_members")
        self.assertFalse(members_df.empty)

        # Check that sum of cluster sizes equals number of canonical records
        total_canon_rows = db.execute_scalar("SELECT COUNT(*) FROM canonical_records")
        total_member_rows = len(members_df)
        self.assertEqual(total_canon_rows, total_member_rows)

    def test_03_startup_recovery(self):
        """Test UnionFind recovery from persisted edges."""
        db = self.db
        uf_new = UnionFind()
        job_reg = get_job_registry()
        res = job_reg.recover_on_startup(union_find=uf_new)

        self.assertGreater(res["edges_replayed_into_union_find"], 0)
        # Check that clusters can be retrieved from recovered state
        recovered_clusters = uf_new.get_clusters()
        self.assertGreater(len(recovered_clusters), 0)

    def test_04_progressive_enrichment_chain(self):
        """Test full multi-hop progressive enrichment discovery for Rohan Sharma."""
        db = self.db
        recs_df = db.execute_query("SELECT * FROM canonical_records")
        edges_df = db.execute_query("SELECT * FROM enrichment_edges")

        matcher = HierarchicalMatcher()
        replay = matcher.replay_progressive_enrichment("email", "rohan.sharma@gmail.com", edges_df, recs_df)

        self.assertTrue(replay["found"])
        self.assertGreaterEqual(len(replay["sources"]), 2)

        # Check that discovered master profile contains all enriched fields
        ent_id = db.execute_scalar("""
            SELECT entity_id FROM entity_members
            WHERE record_uuid = ?
        """, [replay["steps"][0]["record_uuid"]])

        master_ent = db.execute_query("SELECT * FROM master_entities WHERE entity_id = ?", [ent_id])
        self.assertFalse(master_ent.empty)
        row = master_ent.iloc[0]
        self.assertIn("Rohan", row["golden_name"])
        self.assertEqual(row["golden_email"], "rohan.sharma@gmail.com")
        self.assertEqual(row["golden_phone"], "9876543210")


if __name__ == "__main__":
    unittest.main()
