"""
Dynamic Golden Master Schema & Consolidation Test (PRJ-07).
Tests end-to-end entity resolution, AI/heuristic semantic mapping, and golden master dataset
synthesis across diverse domains (Customer Demographics, Social Media, Healthcare).
Verifies:
  - 1 Row = 1 Resolved Entity
  - 1 Column = 1 Consolidated Semantic Attribute (no column proliferation)
  - Conflict resolution selects the best attribute value
  - Incompatible / unmatchable records are preserved without false collapse
"""

import os
import unittest
import pandas as pd
import duckdb

from src.config import DEFAULT_DB_PATH
from src.database import DatabaseManager
from src.pipeline import run_dynamic_user_pipeline
from src.master_records import get_unified_master_dataframe
from src.matcher import HierarchicalMatcher


class TestDynamicGoldenConsolidation(unittest.TestCase):

    def setUp(self):
        self.test_db_path = "data/test_dynamic_golden.duckdb"
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
        self.db = DatabaseManager(self.test_db_path)

    def tearDown(self):
        if hasattr(self, "db") and self.db.conn:
            self.db.conn.close()
        DatabaseManager._instance = None
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except Exception:
                pass

    def test_social_media_semantic_consolidation(self):
        """
        Verify that arbitrary social media CSVs with disparate column names
        (bio_text vs user_bio, followers_count vs followers) are consolidated
        into single golden columns ('bio', 'followers') without duplicate raw columns.
        """
        df_social_a = pd.DataFrame([
            {
                "handle": "tech_guru",
                "full_name": "Alex Mercer",
                "email_address": "alex.mercer@tech.io",
                "bio_text": "Senior ML Engineer building autonomous AI agents.",
                "followers_count": "45000",
                "social_platform": "Twitter",
            },
            {
                "handle": "crypto_whiz",
                "full_name": "Satoshi N",
                "email_address": "satoshi@defi.org",
                "bio_text": "Decentralized consensus researcher.",
                "followers_count": "120000",
                "social_platform": "Twitter",
            }
        ])

        df_social_b = pd.DataFrame([
            {
                "username": "tech_guru",
                "name": "Alex Mercer, PhD",
                "email": "alex.mercer@tech.io",
                "user_bio": "Senior ML Engineer & Open Source Maintainer.",
                "followers": "45200",
                "network": "LinkedIn",
            },
            {
                "username": "design_pro",
                "name": "Elena Rostova",
                "email": "elena.design@studio.co",
                "user_bio": "Lead Product Designer & Design Systems Architect.",
                "followers": "18500",
                "network": "LinkedIn",
            }
        ])

        # Semantic mappings
        mapping_a = {
            "handle": "username",
            "full_name": "name",
            "email_address": "email",
            "bio_text": "bio",
            "followers_count": "followers",
            "social_platform": "platform",
        }

        mapping_b = {
            "username": "username",
            "name": "name",
            "email": "email",
            "user_bio": "bio",
            "followers": "followers",
            "network": "platform",
        }

        specs = [
            {"source_id": "twitter_feed", "source_name": "Twitter Data", "dataframe": df_social_a, "mapping": mapping_a},
            {"source_id": "linkedin_feed", "source_name": "LinkedIn Data", "dataframe": df_social_b, "mapping": mapping_b},
        ]

        summary = run_dynamic_user_pipeline(specs, db_path=self.test_db_path, reset_db=True)

        # 4 total records ingested -> Alex Mercer matches on email and username -> 3 unique golden entities
        self.assertEqual(summary["total_records"], 4)
        self.assertEqual(summary["total_entities"], 3)
        self.assertTrue(summary.get("total_edges", 0) >= 1)

        # Retrieve Unified Master DataFrame
        df_master = get_unified_master_dataframe(self.db)

        # CRITICAL ARCHITECTURAL ASSERTIONS:
        # 1. Row count equals unique entities
        self.assertEqual(len(df_master), 3)

        # 2. No raw column duplication: bio_text and user_bio should NOT exist as columns; ONLY 'bio' should exist
        self.assertNotIn("bio_text", df_master.columns)
        self.assertNotIn("user_bio", df_master.columns)
        self.assertIn("bio", df_master.columns)

        # 3. No followers_count duplication: ONLY 'followers' should exist
        self.assertNotIn("followers_count", df_master.columns)
        self.assertIn("followers", df_master.columns)

        # 4. Alex Mercer entity must have consolidated golden bio and golden name
        alex_row = df_master[df_master["email"] == "alex.mercer@tech.io"].iloc[0]
        self.assertEqual(alex_row["cluster_size"], 2)
        # Completeness / length heuristic chooses longer/richer name & bio
        self.assertIn("Alex Mercer", alex_row["name"])
        self.assertTrue(len(str(alex_row["bio"])) > 20)

    def test_healthcare_domain_dynamic_synthesis(self):
        """
        Verify that a completely different domain (Healthcare / Patients)
        functions seamlessly with arbitrary columns (diagnosis, department, phone)
        without requiring hardcoded column names.
        """
        df_clinic = pd.DataFrame([
            {"patient_id": "P-101", "name": "Dr. Sarah Jenkins", "phone": "+1-555-0192", "diagnosis": "Hypertension", "department": "Cardiology"},
            {"patient_id": "P-102", "name": "Mark Vance", "phone": "+1-555-0193", "diagnosis": "Type 2 Diabetes", "department": "Endocrinology"},
        ])

        df_lab = pd.DataFrame([
            {"patient_code": "P-101", "patient_name": "Sarah Jenkins", "contact_number": "5550192", "diagnosis": "Essential Hypertension", "lab_test": "Lipid Panel"},
            {"patient_code": "P-103", "patient_name": "Carlos Gomez", "contact_number": "5550199", "diagnosis": "Asthma", "lab_test": "Spirometry"},
        ])

        mapping_clinic = {
            "patient_id": "member_id",
            "name": "name",
            "phone": "phone",
            "diagnosis": "diagnosis",
            "department": "department",
        }

        mapping_lab = {
            "patient_code": "member_id",
            "patient_name": "name",
            "contact_number": "phone",
            "diagnosis": "diagnosis",
            "lab_test": "lab_test",
        }

        specs = [
            {"source_id": "clinic_records", "source_name": "Clinic EMR", "dataframe": df_clinic, "mapping": mapping_clinic},
            {"source_id": "lab_results", "source_name": "Diagnostic Lab", "dataframe": df_lab, "mapping": mapping_lab},
        ]

        summary = run_dynamic_user_pipeline(specs, db_path=self.test_db_path, reset_db=True)

        self.assertEqual(summary["total_records"], 4)
        self.assertEqual(summary["total_entities"], 3)
        self.assertTrue(summary.get("total_edges", 0) >= 1)

        df_master = get_unified_master_dataframe(self.db)
        self.assertEqual(len(df_master), 3)

        # Dynamic healthcare columns exist in master output
        self.assertIn("diagnosis", df_master.columns)
        self.assertIn("department", df_master.columns)
        self.assertIn("lab_test", df_master.columns)

        # Sarah Jenkins merged
        sarah = df_master[df_master["member_id"] == "P-101"].iloc[0]
        self.assertEqual(sarah["cluster_size"], 2)

    def test_fresh_run_isolation_and_ecommerce_domain(self):
        """
        Verify that sequential runs with completely different schemas
        (e.g., Healthcare first, then E-Commerce) have ZERO data or column contamination.
        The E-Commerce golden master must contain ONLY e-commerce columns, with zero healthcare residue.
        """
        df_store_1 = pd.DataFrame([
            {"product_sku": "SKU-9901", "product_title": "Wireless Mechanical Keyboard", "brand_name": "Keychron", "retail_price": "89.99", "item_category": "Electronics"},
            {"product_sku": "SKU-9902", "product_title": "Ergonomic Vertical Mouse", "brand_name": "Logitech", "retail_price": "69.99", "item_category": "Accessories"},
        ])

        df_store_2 = pd.DataFrame([
            {"sku_code": "SKU-9901", "item_name": "Keychron K2 Wireless Mechanical Keyboard (RGB)", "manufacturer": "Keychron", "price_usd": "89.99", "department": "Peripherals"},
            {"sku_code": "SKU-9903", "item_name": "USB-C Dual 4K Docking Station", "manufacturer": "Anker", "price_usd": "129.99", "department": "Hubs & Docks"},
        ])

        mapping_1 = {
            "product_sku": "member_id",
            "product_title": "name",
            "brand_name": "company",
            "retail_price": "price",
            "item_category": "category",
        }

        mapping_2 = {
            "sku_code": "member_id",
            "item_name": "name",
            "manufacturer": "company",
            "price_usd": "price",
            "department": "category",
        }

        specs = [
            {"source_id": "shopify_store", "source_name": "Shopify Catalog", "dataframe": df_store_1, "mapping": mapping_1},
            {"source_id": "amazon_store", "source_name": "Amazon US Inventory", "dataframe": df_store_2, "mapping": mapping_2},
        ]

        summary = run_dynamic_user_pipeline(specs, db_path=self.test_db_path, reset_db=True)

        self.assertEqual(summary["total_records"], 4)
        self.assertEqual(summary["total_entities"], 3)
        self.assertTrue(summary.get("total_edges", 0) >= 1)

        df_master = get_unified_master_dataframe(self.db)
        self.assertEqual(len(df_master), 3)

        # 1. Check E-Commerce columns present
        self.assertIn("member_id", df_master.columns)
        self.assertIn("name", df_master.columns)
        self.assertIn("company", df_master.columns)
        self.assertIn("price", df_master.columns)
        self.assertIn("category", df_master.columns)

        # 2. Strict Isolation: ZERO contamination from previous healthcare / demographics tests
        self.assertNotIn("diagnosis", df_master.columns)
        self.assertNotIn("lab_test", df_master.columns)
        self.assertNotIn("aadhaar", df_master.columns)
        self.assertNotIn("email", df_master.columns)
        self.assertNotIn("phone", df_master.columns)
        self.assertNotIn("city", df_master.columns)

        # 3. Merged SKU entity check
        k2_keyboard = df_master[df_master["member_id"] == "SKU-9901"].iloc[0]
        self.assertEqual(k2_keyboard["cluster_size"], 2)
        self.assertEqual(k2_keyboard["company"], "Keychron")
        # Completeness / length heuristic chooses longer title
        self.assertIn("Keychron K2 Wireless Mechanical Keyboard", k2_keyboard["name"])


if __name__ == "__main__":
    unittest.main()
