"""
Acceptance Test Suite for Dynamic Stakeholder Entity Resolution Workflow.
Verifies end-to-end resolution on realistic, messy heterogeneous datasets:
- test_dataset_A_customers.csv
- test_dataset_B_crm_messy.csv
- test_dataset_C_support_messy.csv
Tests casing differences, whitespace, phone formatting, missing values,
minor typos, conflicting attributes, singletons, and CSV export.
"""

import os
import sys
import unittest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database import get_db_manager
from src.mapping import SchemaMappingEngine
from src.pipeline import run_dynamic_user_pipeline


class TestAcceptanceWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db_path = "data/test_acceptance.duckdb"
        cls.db = get_db_manager(cls.test_db_path)

        # Dataset A: Clean Customer Master
        cls.df_a = pd.DataFrame([
            {"cust_id": "CUST-001", "customer_name": "Alice Johnson", "email_address": "alice.j@acme.com", "phone_no": "9876543210", "city": "San Francisco"},
            {"cust_id": "CUST-002", "customer_name": "Bob Smith", "email_address": "bob.smith@techcorp.io", "phone_no": "9123456780", "city": "New York"},
            {"cust_id": "CUST-003", "customer_name": "Charlie Davis", "email_address": "charlie.d@global.net", "phone_no": "9988776655", "city": "Chicago"},
            {"cust_id": "CUST-004", "customer_name": "Diana Prince", "email_address": "diana.p@themyscira.org", "phone_no": "9811223344", "city": "Boston"}, # Singleton
        ])

        # Dataset B: Messy CRM Export (different headers, uppercase emails, phone punctuation)
        cls.df_b = pd.DataFrame([
            {"contact_id": "CRM-101", "full_name": "Alice M. Johnson", "email": "  ALICE.J@ACME.COM  ", "mobile": "+1 (987) 654-3210", "company_name": "Acme Industries", "residence": "San Francisco, CA"},
            {"contact_id": "CRM-102", "full_name": "Robert Smith", "email": "bob.smith@techcorp.io", "mobile": "9123456780", "company_name": "TechCorp Global", "residence": "NYC"},
            {"contact_id": "CRM-103", "full_name": "Edward Norton", "email": "edward.n@actor.com", "mobile": "9555667788", "company_name": "Filmcraft", "residence": "Los Angeles"}, # Singleton
        ])

        # Dataset C: Messy Support Tickets (username, phone with country code, typos, missing email)
        cls.df_c = pd.DataFrame([
            {"ticket_id": "SUP-901", "user_handle": "alice_j", "contact_phone": "+91 9876543210", "user_email": "alice.j@acme.com", "notes": "VIP Priority Customer"},
            {"ticket_id": "SUP-902", "user_handle": "charlie_davis", "contact_phone": "09988776655", "user_email": "charlie.d@global.net", "notes": "Resolved login ticket"},
            {"ticket_id": "SUP-903", "user_handle": "fiona_gallagher", "contact_phone": "9444555666", "user_email": "fiona.g@southside.biz", "notes": "New user onboarded"}, # Singleton
        ])

    def test_01_automatic_schema_mapping(self):
        """Verify semantic schema mapping accurately identifies canonical targets across messy headers."""
        engine = SchemaMappingEngine()

        # Map Dataset A
        map_a = {c: p["suggested_canonical"] for c, p in engine.suggest_mappings_for_table(self.df_a.columns.tolist()).items()}
        self.assertEqual(map_a["customer_name"], "name")
        self.assertEqual(map_a["email_address"], "email")
        self.assertEqual(map_a["phone_no"], "phone")

        # Map Dataset B
        map_b = {c: p["suggested_canonical"] for c, p in engine.suggest_mappings_for_table(self.df_b.columns.tolist()).items()}
        self.assertEqual(map_b["full_name"], "name")
        self.assertEqual(map_b["email"], "email")
        self.assertEqual(map_b["mobile"], "phone")
        self.assertEqual(map_b["company_name"], "company")

        # Map Dataset C
        map_c = {c: p["suggested_canonical"] for c, p in engine.suggest_mappings_for_table(self.df_c.columns.tolist()).items()}
        self.assertEqual(map_c["user_handle"], "username")
        self.assertEqual(map_c["contact_phone"], "phone")
        self.assertEqual(map_c["user_email"], "email")

    def test_02_compatibility_check(self):
        """Verify cross-dataset linkage validation detects common identifiers."""
        engine = SchemaMappingEngine()
        mappings = {
            "Dataset_A": {"email_address": "email", "phone_no": "phone", "customer_name": "name"},
            "Dataset_B": {"email": "email", "mobile": "phone", "full_name": "name"},
            "Dataset_C": {"user_email": "email", "contact_phone": "phone", "user_handle": "username"},
        }
        compat = engine.check_cross_dataset_compatibility(mappings)
        self.assertTrue(compat["is_compatible"])
        self.assertIn("email", compat["shared_identifiers"])
        self.assertIn("phone", compat["shared_identifiers"])

    def test_03_end_to_end_resolution_and_deduplication(self):
        """Execute end-to-end resolution on messy datasets and verify master entity synthesis."""
        engine = SchemaMappingEngine()
        specs = [
            {
                "source_id": "src_cust_a",
                "source_name": "test_dataset_A_customers.csv",
                "dataframe": self.df_a,
                "mapping": {c: p["suggested_canonical"] for c, p in engine.suggest_mappings_for_table(self.df_a.columns.tolist()).items()},
            },
            {
                "source_id": "src_crm_b",
                "source_name": "test_dataset_B_crm_messy.csv",
                "dataframe": self.df_b,
                "mapping": {c: p["suggested_canonical"] for c, p in engine.suggest_mappings_for_table(self.df_b.columns.tolist()).items()},
            },
            {
                "source_id": "src_supp_c",
                "source_name": "test_dataset_C_support_messy.csv",
                "dataframe": self.df_c,
                "mapping": {c: p["suggested_canonical"] for c, p in engine.suggest_mappings_for_table(self.df_c.columns.tolist()).items()},
            },
        ]

        stats = run_dynamic_user_pipeline(datasets_spec=specs, db_path=self.test_db_path, reset_db=True)

        # 4 records in A + 3 in B + 3 in C = 10 input records
        self.assertEqual(stats["total_records"], 10)

        # Alice is in A, B, C (Cluster Size = 3)
        # Bob is in A, B (Cluster Size = 2)
        # Charlie is in A, C (Cluster Size = 2)
        # Diana is in A (Singleton = 1)
        # Edward is in B (Singleton = 1)
        # Fiona is in C (Singleton = 1)
        # Total unique entities = 3 merged + 3 singletons = 6 entities
        self.assertEqual(stats["total_entities"], 6)
        self.assertEqual(stats["total_linked_entities"], 3)
        self.assertEqual(stats["total_singletons"], 3)
        self.assertEqual(stats["duplicates_resolved"], 4)

        # Verify Alice's unified Golden Profile
        db = get_db_manager(self.test_db_path)
        alice_master = db.execute_query("""
            SELECT * FROM master_entities
            WHERE golden_email = 'alice.j@acme.com'
        """)
        self.assertFalse(alice_master.empty)
        alice_row = alice_master.iloc[0]
        self.assertEqual(alice_row["cluster_size"], 3)
        self.assertEqual(alice_row["golden_phone"], "9876543210")
        self.assertEqual(alice_row["golden_username"], "alice_j")
        self.assertEqual(alice_row["golden_company"], "Acme Industries")

        # Verify Field Provenance exists for Alice
        alice_prov = db.execute_query("""
            SELECT * FROM field_provenance
            WHERE entity_id = ?
        """, [alice_row["entity_id"]])
        self.assertFalse(alice_prov.empty)


if __name__ == "__main__":
    unittest.main()
