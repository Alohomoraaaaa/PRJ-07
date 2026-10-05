"""
Unit and Integration Tests for Phase A (Core Happy Path).
Verifies data normalization, deterministic schema mapping, inverted indexing,
Union-Find clustering with enrichment edge logging, master synthesis,
entity_members junction table, field provenance, and progressive enrichment replay.
"""

import os
import sys
import unittest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import DEFAULT_DB_PATH
from src.normalizer import DataNormalizer
from src.mapping import SchemaMappingEngine
from src.clustering import UnionFind
from src.matcher import HierarchicalMatcher
from src.database import get_db_manager
from src.pipeline import run_full_demonstration_pipeline


TEST_DB_PATH = "data/test_unified_repository.duckdb"


class TestPhaseACorePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = get_db_manager(TEST_DB_PATH)
        cls.stats = run_full_demonstration_pipeline(db_path=TEST_DB_PATH)

    def test_01_normalization(self):
        """Test dual normalization for email, phone, name, username."""
        norm = DataNormalizer()
        em, ok_em = norm.normalize_email("  John.Doe@TechCorp.COM  ")
        self.assertEqual(em, "john.doe@techcorp.com")
        self.assertTrue(ok_em)

        ph, ok_ph = norm.normalize_phone("+91 98765-43210")
        self.assertEqual(ph, "9876543210")
        self.assertTrue(ok_ph)

        nm, ok_nm = norm.normalize_name("  johnathan   doe  ")
        self.assertEqual(nm, "Johnathan Doe")
        self.assertTrue(ok_nm)

    def test_02_schema_mapping(self):
        """Test deterministic RapidFuzz / synonym schema mapping."""
        engine = SchemaMappingEngine()
        cols = ["email_id", "contact_no", "full_name", "residence_address", "emp_code"]
        proposals = engine.suggest_mappings_for_table(cols)

        self.assertEqual(proposals["email_id"]["suggested_canonical"], "email")
        self.assertEqual(proposals["contact_no"]["suggested_canonical"], "phone")
        self.assertEqual(proposals["full_name"]["suggested_canonical"], "name")
        self.assertEqual(proposals["emp_code"]["suggested_canonical"], "member_id")

    def test_03_union_find_clustering(self):
        """Test UnionFind graph connectivity and transitivity."""
        uf = UnionFind(["A", "B", "C", "D"])
        uf.union("A", "B")
        uf.union("B", "C")
        self.assertEqual(uf.find("A"), uf.find("C"))
        self.assertNotEqual(uf.find("A"), uf.find("D"))

    def test_04_database_tables_populated(self):
        """Verify all DuckDB tables and junction entities are populated."""
        db = self.db
        num_sources = db.execute_scalar("SELECT COUNT(*) FROM data_sources")
        num_recs = db.execute_scalar("SELECT COUNT(*) FROM canonical_records")
        num_entities = db.execute_scalar("SELECT COUNT(*) FROM master_entities")
        num_edges = db.execute_scalar("SELECT COUNT(*) FROM enrichment_edges")
        num_members = db.execute_scalar("SELECT COUNT(*) FROM entity_members")
        num_prov = db.execute_scalar("SELECT COUNT(*) FROM field_provenance")

        self.assertGreaterEqual(num_sources, 2)
        self.assertEqual(num_recs, 6)
        self.assertEqual(num_entities, 4)
        self.assertGreaterEqual(num_edges, 2)
        self.assertEqual(num_members, num_recs)  # Every record has exactly one membership row
        self.assertGreater(num_prov, 5)

    def test_05_unmatched_singletons_preserved(self):
        """Verify that unmatched records exist as singletons (no record loss)."""
        db = self.db
        singletons = db.execute_scalar("SELECT COUNT(*) FROM master_entities WHERE cluster_size = 1")
        self.assertGreater(singletons, 0)

        # Check that singletons have full field provenance
        singleton_entity_id = db.execute_scalar("SELECT entity_id FROM master_entities WHERE cluster_size = 1 LIMIT 1")
        prov_rows = db.execute_query("SELECT * FROM field_provenance WHERE entity_id = ?", [singleton_entity_id])
        self.assertFalse(prov_rows.empty)

    def test_06_progressive_enrichment_replay(self):
        """Verify progressive entity enrichment replay starting from email."""
        db = self.db
        recs_df = db.execute_query("SELECT * FROM canonical_records")
        edges_df = db.execute_query("SELECT * FROM enrichment_edges")

        matcher = HierarchicalMatcher()
        # Seed on Rohan Sharma
        replay = matcher.replay_progressive_enrichment("email", "rohan.sharma@gmail.com", edges_df, recs_df)
        self.assertTrue(replay["found"])
        self.assertGreaterEqual(len(replay["sources"]), 2)
        self.assertGreaterEqual(len(replay["steps"]), 1)

    def test_07_no_merge_on_blank_identifiers(self):
        """Unit test: Two records with no shared non-blank identifier must NOT merge."""
        uf = UnionFind(["rec_1", "rec_2", "rec_3"])
        matcher = HierarchicalMatcher(union_find=uf)

        # Record 1 with blanks across all identifier fields
        rec1 = {
            "email": None,
            "phone": "",
            "username": "   ",
            "member_id": None,
            "name": None,
            "address": "",
        }
        # Record 2 also with blanks across all identifier fields
        rec2 = {
            "email": "",
            "phone": "  ",
            "username": None,
            "member_id": "",
            "name": "",
            "address": None,
        }
        # Record 3 with valid email
        rec3 = {
            "email": "user@example.com",
            "phone": "",
            "username": None,
            "member_id": None,
            "name": "User Three",
            "address": "123 Main St",
        }

        # Register record 1
        matcher.register_record_identifiers("rec_1", "src_test_a", rec1)
        # Attempt to match record 2
        edges2 = matcher.match_and_link_record("rec_2", "src_test_b", rec2, import_batch_id="batch_1")
        self.assertEqual(len(edges2), 0)

        # Register record 2 and attempt to match record 3
        matcher.register_record_identifiers("rec_2", "src_test_b", rec2)
        edges3 = matcher.match_and_link_record("rec_3", "src_test_c", rec3, import_batch_id="batch_1")
        self.assertEqual(len(edges3), 0)

        # Verify Union-Find has NOT linked rec_1, rec_2, or rec_3
        clusters = uf.get_clusters()
        self.assertEqual(len(clusters), 3)
        self.assertEqual(clusters["rec_1"], ["rec_1"])
        self.assertEqual(clusters["rec_2"], ["rec_2"])
        self.assertEqual(clusters["rec_3"], ["rec_3"])

        # Verify inverted index has no blank/None keys
        for (ident_type, val), entries in matcher.inverted_index.items():
            self.assertNotEqual(val, "")
            self.assertNotEqual(val, "None")
            self.assertNotEqual(val, "nan")

    def test_08_google_maps_dataset_simulation_and_field_preservation(self):
        """Verify 2 Google Maps datasets with identical schemas and no ID fields produce distinct singletons and retain all raw columns."""
        from src.pipeline import run_dynamic_user_pipeline
        from src.master_records import get_unified_master_dataframe

        df_mumbai = pd.DataFrame([
            {"business_name": "OffDuty India Bandra", "rating": 4.6, "category": "Clothing Store", "review_count": 128, "latitude": 19.0596, "longitude": 72.8295, "city": "Mumbai"},
            {"business_name": "Cafe Mocha Bandra", "rating": 4.2, "category": "Cafe", "review_count": 540, "latitude": 19.0601, "longitude": 72.8310, "city": "Mumbai"},
            {"business_name": "Crossword Bookstores", "rating": 4.5, "category": "Book Store", "review_count": 310, "latitude": 19.0650, "longitude": 72.8340, "city": "Mumbai"},
        ])

        df_delhi = pd.DataFrame([
            {"business_name": "OffDuty India Connaught Place", "rating": 4.7, "category": "Clothing Store", "review_count": 215, "latitude": 28.6315, "longitude": 77.2167, "city": "Delhi"},
            {"business_name": "Cafe Coffee Day CP", "rating": 4.0, "category": "Cafe", "review_count": 420, "latitude": 28.6320, "longitude": 77.2180, "city": "Delhi"},
            {"business_name": "Oxford Book Store", "rating": 4.4, "category": "Book Store", "review_count": 195, "latitude": 28.6290, "longitude": 77.2150, "city": "Delhi"},
        ])

        mapping_engine = SchemaMappingEngine()
        specs = [
            {
                "source_id": "src_gmaps_mumbai",
                "source_name": "gmaps_mumbai.csv",
                "dataframe": df_mumbai,
                "mapping": {c: p["suggested_canonical"] for c, p in mapping_engine.suggest_mappings_for_table(df_mumbai.columns.tolist()).items()},
            },
            {
                "source_id": "src_gmaps_delhi",
                "source_name": "gmaps_delhi.csv",
                "dataframe": df_delhi,
                "mapping": {c: p["suggested_canonical"] for c, p in mapping_engine.suggest_mappings_for_table(df_delhi.columns.tolist()).items()},
            },
        ]

        gmaps_test_db = "data/test_gmaps.duckdb"
        stats = run_dynamic_user_pipeline(datasets_spec=specs, db_path=gmaps_test_db, reset_db=True)

        # 3 in Mumbai + 3 in Delhi = 6 records total
        self.assertEqual(stats["total_records"], 6)
        # Should NOT collapse into 1 entity — must produce 6 distinct singletons
        self.assertEqual(stats["total_entities"], 6)
        self.assertEqual(stats["total_singletons"], 6)
        self.assertEqual(stats["duplicates_resolved"], 0)

        # Verify all raw descriptive fields are present in raw_fields_json in database
        gmaps_db = get_db_manager(gmaps_test_db)
        raw_recs = gmaps_db.execute_query("SELECT raw_fields_json FROM canonical_records")
        self.assertEqual(len(raw_recs), 6)
        import json
        for _, row in raw_recs.iterrows():
            parsed_raw = json.loads(row["raw_fields_json"])
            self.assertIn("rating", parsed_raw)
            self.assertIn("category", parsed_raw)
            self.assertIn("review_count", parsed_raw)
            self.assertIn("latitude", parsed_raw)
            self.assertIn("longitude", parsed_raw)
            self.assertIn("city", parsed_raw)

        # Verify get_unified_master_dataframe returns all descriptive columns
        unified_master_df = get_unified_master_dataframe(gmaps_db)
        self.assertEqual(len(unified_master_df), 6)
        self.assertIn("rating", unified_master_df.columns)
        self.assertIn("category", unified_master_df.columns)
        self.assertIn("review_count", unified_master_df.columns)
        self.assertIn("latitude", unified_master_df.columns)
        self.assertIn("longitude", unified_master_df.columns)
        self.assertIn("city", unified_master_df.columns)


if __name__ == "__main__":
    unittest.main()

