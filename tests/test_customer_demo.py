"""
Unit and Integration Test for Customer A and Customer B datasets.
Tests:
- first_name + last_name ↔ full_name
- email ↔ email_id
- phone_number ↔ mobile_number (+91 vs standard)
- aadhaar ↔ aadhar_number (spaces vs clean)
- city ↔ city_name
- address ↔ residential_address
- Exact outcome: 6 input rows -> 4 final master entities
"""

import os
import sys
import unittest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database import get_db_manager
from src.mapping import SchemaMappingEngine
from src.pipeline import run_dynamic_user_pipeline
from src.master_records import get_unified_master_dataframe


class TestCustomerDemoResolution(unittest.TestCase):
    def setUp(self):
        self.db_path = "data/test_customer_demo.duckdb"
        self.db = get_db_manager(self.db_path)

        self.df_a = pd.DataFrame([
            {"first_name": "Rohan", "last_name": "Sharma", "email": "rohan.sharma@gmail.com", "phone_number": "+91 9876543210", "aadhaar": "4521 6789 1234", "city": "Mumbai", "address": "12 MG Road, Andheri"},
            {"first_name": "Priya", "last_name": "Patil", "email": "priya.patil@gmail.com", "phone_number": "+91 9988776655", "aadhaar": "5234 1122 3344", "city": "Pune", "address": "45 FC Road, Shivajinagar"},
            {"first_name": "Amit", "last_name": "Deshmukh", "email": "amit.d@gmail.com", "phone_number": "+91 9123456789", "aadhaar": "6789 2233 4455", "city": "Nashik", "address": "18 College Road, Nashik"},
        ])

        self.df_b = pd.DataFrame([
            {"full_name": "Rohan Kumar Sharma", "email_id": "rohan.sharma@gmail.com", "mobile_number": "9876543210", "aadhar_number": "452167891234", "city_name": "Mumbai", "residential_address": "12, M.G. Road, Andheri East"},
            {"full_name": "Priya N Patil", "email_id": "priya.patil@gmail.com", "mobile_number": "9988776655", "aadhar_number": "523411223344", "city_name": "Pune", "residential_address": "45 FC Road, Shivaji Nagar"},
            {"full_name": "Sneha Kulkarni", "email_id": "sneha.k@gmail.com", "mobile_number": "8899001122", "aadhar_number": "789012345678", "city_name": "Thane", "residential_address": "22 Ghodbunder Road, Thane"},
        ])

    def test_01_mapping_engine(self):
        """Verify automatic schema mapping on customer datasets."""
        engine = SchemaMappingEngine()
        map_a = {c: p["suggested_canonical"] for c, p in engine.suggest_mappings_for_table(self.df_a.columns.tolist()).items()}
        self.assertEqual(map_a["first_name"], "first_name")
        self.assertEqual(map_a["last_name"], "last_name")
        self.assertEqual(map_a["email"], "email")
        self.assertEqual(map_a["phone_number"], "phone")
        self.assertEqual(map_a["aadhaar"], "aadhaar")
        self.assertEqual(map_a["city"], "city")
        self.assertEqual(map_a["address"], "address")

        map_b = {c: p["suggested_canonical"] for c, p in engine.suggest_mappings_for_table(self.df_b.columns.tolist()).items()}
        self.assertEqual(map_b["full_name"], "name")
        self.assertEqual(map_b["email_id"], "email")
        self.assertEqual(map_b["mobile_number"], "phone")
        self.assertEqual(map_b["aadhar_number"], "aadhaar")
        self.assertEqual(map_b["city_name"], "city")
        self.assertEqual(map_b["residential_address"], "address")

    def test_02_resolution_outcome(self):
        """Verify 6 input rows -> 4 final master entities and clean master table output."""
        engine = SchemaMappingEngine()
        specs = [
            {
                "source_id": "src_cust_a",
                "source_name": "customer_a.csv",
                "dataframe": self.df_a,
                "mapping": {c: p["suggested_canonical"] for c, p in engine.suggest_mappings_for_table(self.df_a.columns.tolist()).items()},
            },
            {
                "source_id": "src_cust_b",
                "source_name": "customer_b.csv",
                "dataframe": self.df_b,
                "mapping": {c: p["suggested_canonical"] for c, p in engine.suggest_mappings_for_table(self.df_b.columns.tolist()).items()},
            },
        ]

        stats = run_dynamic_user_pipeline(datasets_spec=specs, db_path=self.db_path, reset_db=True)

        self.assertEqual(stats["total_records"], 6)
        self.assertEqual(stats["total_entities"], 4)
        self.assertEqual(stats["total_linked_entities"], 2)
        self.assertEqual(stats["total_singletons"], 2)
        self.assertEqual(stats["duplicates_resolved"], 2)

        # Inspect unified master dataframe
        unified_df = get_unified_master_dataframe(self.db)
        self.assertEqual(len(unified_df), 4)

        # Verify columns exist
        for expected_col in ["entity_id", "cluster_size", "sources", "name", "email", "phone", "aadhaar", "city", "address"]:
            self.assertIn(expected_col, unified_df.columns)

        # Check Rohan's unified profile (merged size = 2)
        rohan = unified_df[unified_df["email"] == "rohan.sharma@gmail.com"].iloc[0]
        self.assertEqual(rohan["cluster_size"], 2)
        self.assertIn("customer_a.csv", rohan["sources"])
        self.assertIn("customer_b.csv", rohan["sources"])
        self.assertEqual(rohan["phone"], "9876543210")
        self.assertIn("452167891234", rohan["aadhaar"].replace(" ", ""))

        # Check Priya's unified profile (merged size = 2)
        priya = unified_df[unified_df["email"] == "priya.patil@gmail.com"].iloc[0]
        self.assertEqual(priya["cluster_size"], 2)

        # Check Amit (singleton size = 1 from A)
        amit = unified_df[unified_df["email"] == "amit.d@gmail.com"].iloc[0]
        self.assertEqual(amit["cluster_size"], 1)

        # Check Sneha (singleton size = 1 from B)
        sneha = unified_df[unified_df["email"] == "sneha.k@gmail.com"].iloc[0]
        self.assertEqual(sneha["cluster_size"], 1)


if __name__ == "__main__":
    unittest.main()
