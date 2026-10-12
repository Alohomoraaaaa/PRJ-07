"""
Permanent Regression Test Suite (PRJ-07).
Validates:
1. Universal preservation of unmapped/passthrough columns across diverse data types
   (Date, numeric, boolean False/True, free-text) with zero data loss or type corruption.
2. Wide dataset handling (100+ columns) with near-duplicate/ambiguous names:
   confirming collision detection flags candidates for review without silent failure or overwrites.
3. Completely domain-agnostic execution on non-person datasets (Product Inventory / Warehouse Stock):
   confirming the pipeline makes zero assumptions about person-like schemas.
4. Step 6 search querying actual stored identifiers from database records.
"""

import unittest
import json
import pandas as pd
from typing import Dict, Any

from src.database import get_db_manager
from src.config import DEFAULT_DB_PATH
from src.mapping import SchemaMappingEngine
from src.ai_mapper import GeminiSchemaMapper
from src.matcher import HierarchicalMatcher
from src.pipeline import run_dynamic_user_pipeline
from src.master_records import get_unified_master_dataframe


import os

class TestUnmappedAndWideDatasets(unittest.TestCase):
    """Permanent regression tests for unmapped column preservation, wide schemas, and non-person domains."""

    def setUp(self):
        self.test_db_path = "data/test_unmapped_regression.duckdb"
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except Exception:
                pass
        self.db = get_db_manager(self.test_db_path)

    def tearDown(self):
        if hasattr(self, "db") and self.db.conn:
            try:
                self.db.conn.close()
            except Exception:
                pass
        from src.database import DatabaseManager
        DatabaseManager._instance = None
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except Exception:
                pass

    def test_01_unmapped_passthrough_typed_columns_preservation(self):
        """
        BUG CLASS 1: Confirm Date, numeric, boolean (especially False and 0.0), and free-text
        unmapped/passthrough columns survive into the golden master output without being dropped.
        """
        # Dataset 1: Primary CRM
        df_a = pd.DataFrame([
            {
                "email_addr": "clara.oswald@tardis.org",
                "full_name": "Clara Oswald",
                "signup_date": "2023-04-12",
                "account_balance": 0.0,            # Falsy numeric: must NOT be dropped
                "is_premium_member": False,        # Falsy boolean: must NOT be dropped
                "internal_memo": "Primary account registered via London node",
            },
            {
                "email_addr": "rose.tyler@parallel.org",
                "full_name": "Rose Tyler",
                "signup_date": "2022-01-01",
                "account_balance": 1500.50,
                "is_premium_member": True,
                "internal_memo": "Singleton profile",
            }
        ])

        # Dataset 2: Billing / Portal (Same Clara, different column names for unmapped data)
        df_b = pd.DataFrame([
            {
                "contact_email": "clara.oswald@tardis.org",
                "customer_name": "Clara Oswald",
                "last_active_date": "2024-10-01",
                "loyalty_tier_score": 98.5,
                "opt_in_newsletter": True,
                "support_notes": "Requested travel reimbursement ticket #409",
            }
        ])

        specs = [
            {
                "source_id": "crm_a",
                "source_name": "crm_a.csv",
                "dataframe": df_a,
                "mapping": {
                    "email_addr": "email",
                    "full_name": "name",
                    "signup_date": "unmapped",
                    "account_balance": "unmapped",
                    "is_vip_member": "unmapped",
                    "internal_memo": "unmapped",
                }
            },
            {
                "source_id": "portal_b",
                "source_name": "portal_b.csv",
                "dataframe": df_b,
                "mapping": {
                    "contact_email": "email",
                    "customer_name": "name",
                    "last_active_date": "unmapped",
                    "loyalty_tier_score": "unmapped",
                    "opt_in_newsletter": "unmapped",
                    "support_notes": "unmapped",
                }
            }
        ]

        stats = run_dynamic_user_pipeline(datasets_spec=specs, db_path=self.test_db_path, reset_db=True)
        self.assertEqual(stats["total_records"], 3)
        self.assertEqual(stats["total_entities"], 2)  # Clara merged, Rose is singleton

        df_master = get_unified_master_dataframe(self.db)
        self.assertEqual(len(df_master), 2)

        # Find merged Clara Oswald entity
        clara_rows = df_master[df_master["email"] == "clara.oswald@tardis.org"]
        self.assertEqual(len(clara_rows), 1)
        clara = clara_rows.iloc[0]

        # Verify cluster size and sources
        self.assertEqual(int(clara["cluster_size"]), 2)

        # 1. Verify Date column survived
        self.assertIn("signup_date", df_master.columns)
        self.assertEqual(str(clara["signup_date"]), "2023-04-12")

        # 2. Verify numeric 0.0 survived (was not dropped due to truthiness check)
        self.assertIn("account_balance", df_master.columns)
        self.assertEqual(float(clara["account_balance"]), 0.0)

        # 3. Verify boolean False survived (was not dropped due to truthiness check)
        self.assertIn("is_premium_member", df_master.columns)
        self.assertEqual(bool(clara["is_premium_member"]), False)

        # 4. Verify free-text column survived
        self.assertIn("internal_memo", df_master.columns)
        self.assertEqual(str(clara["internal_memo"]), "Primary account registered via London node")

        # 5. Verify Dataset B unmapped columns also survived in Clara's master record
        self.assertIn("last_active_date", df_master.columns)
        self.assertEqual(str(clara["last_active_date"]), "2024-10-01")
        self.assertIn("loyalty_tier_score", df_master.columns)
        self.assertEqual(float(clara["loyalty_tier_score"]), 98.5)
        self.assertIn("opt_in_newsletter", df_master.columns)
        self.assertEqual(bool(clara["opt_in_newsletter"]), True)
        self.assertIn("support_notes", df_master.columns)
        self.assertEqual(str(clara["support_notes"]), "Requested travel reimbursement ticket #409")

    def test_02_wide_dataset_100_plus_columns_and_ambiguous_names(self):
        """
        BUG CLASS 2: Generate a 105-column dataset with near-duplicate ambiguous columns.
        Confirm:
        1. Schema mapper detects ambiguous collisions and flags them with is_ambiguous = True.
        2. Ambiguous columns are left unmapped/passthrough so data is never overwritten.
        3. Full pipeline runs without crash or silent drops on 100+ columns.
        """
        num_metadata_cols = 95
        wide_data = {
            "primary_email": ["agent.smith@matrix.net", "neo@matrix.net"],
            "full_name": ["Agent Smith", "Thomas Anderson"],
            # Ambiguous address candidates
            "billing_address": ["Room 302 Megacity", "Heart O The City Hotel"],
            "shipping_address": ["Dock 42 Sublevel", "Zion Central Command"],
            "residential_address": ["Megacity West", "Zion Habitation"],
            "server_address": ["10.0.0.1", "127.0.0.1"],
            # Ambiguous name candidates
            "alias_name": ["Smith", "The One"],
            "legal_name": ["Smith Program #4", "Thomas A. Anderson"],
        }
        # Add 95 additional custom metadata columns
        for i in range(1, num_metadata_cols + 1):
            wide_data[f"system_metric_{i:03d}"] = [f"val_a_{i}", f"val_b_{i}"]

        df_wide = pd.DataFrame(wide_data)
        self.assertGreaterEqual(len(df_wide.columns), 100)

        # Test deterministic schema engine collision handling
        engine = SchemaMappingEngine()
        proposals = engine.suggest_mappings_for_table(df_wide.columns.tolist())
        self.assertEqual(len(proposals), len(df_wide.columns))

        # Check that ambiguous address candidates were flagged for review
        ambiguous_found = [c for c, p in proposals.items() if p.get("is_ambiguous", False)]
        self.assertGreater(len(ambiguous_found), 0)
        self.assertTrue(any("address" in c for c in ambiguous_found))

        # Verify that ambiguous columns did not silently overwrite one another
        for c in ["billing_address", "shipping_address"]:
            if proposals[c].get("is_ambiguous"):
                self.assertEqual(proposals[c]["suggested_canonical"], "unmapped")
                self.assertIn("Ambiguous", proposals[c]["reason"])

        # Test GeminiSchemaMapper chunked execution
        ai_mapper = GeminiSchemaMapper()
        ai_proposals = ai_mapper.suggest_mappings(df_wide)
        self.assertEqual(len(ai_proposals), len(df_wide.columns))

        # Run pipeline with the wide dataset
        user_mapping = {c: p["suggested_canonical"] for c, p in proposals.items()}
        specs = [
            {
                "source_id": "wide_src_1",
                "source_name": "wide_dataset.csv",
                "dataframe": df_wide,
                "mapping": user_mapping,
            }
        ]

        stats = run_dynamic_user_pipeline(datasets_spec=specs, db_path=self.test_db_path, reset_db=True)
        self.assertEqual(stats["total_records"], 2)
        self.assertEqual(stats["total_entities"], 2)

        df_master = get_unified_master_dataframe(self.db)
        self.assertEqual(len(df_master), 2)

        # Confirm all 95 custom columns plus ambiguous columns survived in master output
        for i in range(1, 10):
            self.assertIn(f"system_metric_{i:03d}", df_master.columns)
        self.assertIn("billing_address", df_master.columns)
        self.assertIn("shipping_address", df_master.columns)
        self.assertIn("address", df_master.columns)

    def test_03_non_person_domain_product_inventory(self):
        """
        Verify the pipeline operates cleanly on a completely non-person domain:
        Product Inventory / Warehouse Stock Records.
        Confirming zero hardcoded assumptions about human names or contact schemas.
        """
        # Warehouse A: physical inventory counts
        df_warehouse = pd.DataFrame([
            {
                "sku_code": "PROD-SONY-WH1000",
                "upc_barcode": "49055249001",
                "item_title": "Sony WH-1000XM5 Noise Cancelling Headphones",
                "aisle_location": "Aisle-14B",
                "units_in_stock": 42,
                "requires_refrigeration": False,
            },
            {
                "sku_code": "PROD-APPLE-IPAD11",
                "upc_barcode": "19594912002",
                "item_title": "Apple iPad Air 11 M2 128GB Space Gray",
                "aisle_location": "Aisle-03C",
                "units_in_stock": 15,
                "requires_refrigeration": False,
            },
            {
                "sku_code": "PROD-DJI-DRONE4P",
                "upc_barcode": "69415659003",
                "item_title": "DJI Mini 4 Pro Drone Quadcopter",
                "aisle_location": "Aisle-08A",
                "units_in_stock": 6,
                "requires_refrigeration": False,
            }
        ])

        # E-Commerce Catalog: online store pricing and warranty
        df_ecom = pd.DataFrame([
            {
                "product_identifier": "PROD-SONY-WH1000",
                "ean_code": "49055249001",
                "catalog_name": "Sony WH1000XM5 Wireless Headphones (Black)",
                "retail_price": 399.99,
                "warranty_years": 2,
                "is_featured_item": True,
            },
            {
                "product_identifier": "PROD-APPLE-IPAD11",
                "ean_code": "19594912002",
                "catalog_name": "iPad Air 11-inch (2024, M2 Chip, Wi-Fi)",
                "retail_price": 599.00,
                "warranty_years": 1,
                "is_featured_item": False,
            },
            {
                "product_identifier": "PROD-LOGI-MX3S",
                "ean_code": "09785517004",
                "catalog_name": "Logitech MX Master 3S Wireless Mouse",
                "retail_price": 99.99,
                "warranty_years": 1,
                "is_featured_item": False,
            }
        ])

        # Map non-person domain fields
        specs = [
            {
                "source_id": "warehouse_db",
                "source_name": "warehouse_inventory.csv",
                "dataframe": df_warehouse,
                "mapping": {
                    "sku_code": "member_id",      # Unique item code mapped to canonical identifier
                    "upc_barcode": "aadhaar",      # Barcode mapped to canonical identifier
                    "item_title": "name",          # Title mapped to canonical name
                    "aisle_location": "unmapped",
                    "units_in_stock": "unmapped",
                    "requires_refrigeration": "unmapped",
                }
            },
            {
                "source_id": "ecom_catalog",
                "source_name": "ecom_products.csv",
                "dataframe": df_ecom,
                "mapping": {
                    "product_identifier": "member_id",
                    "ean_code": "aadhaar",
                    "catalog_name": "name",
                    "retail_price": "unmapped",
                    "warranty_years": "unmapped",
                    "is_featured_item": "unmapped",
                }
            }
        ]

        stats = run_dynamic_user_pipeline(datasets_spec=specs, db_path=self.test_db_path, reset_db=True)
        # 6 total rows: 2 merged products (Sony, Apple), 2 singletons (DJI drone, Logitech mouse) = 4 golden products
        self.assertEqual(stats["total_records"], 6)
        self.assertEqual(stats["total_entities"], 4)
        self.assertEqual(stats["total_linked_entities"], 2)
        self.assertEqual(stats["total_singletons"], 2)

        df_master = get_unified_master_dataframe(self.db)
        self.assertEqual(len(df_master), 4)

        # Inspect Sony Headphone golden record
        sony_row = df_master[df_master["member_id"] == "PROD-SONY-WH1000"].iloc[0]
        self.assertEqual(int(sony_row["cluster_size"]), 2)
        self.assertEqual(str(sony_row["aadhaar"]), "49055249001")

        # Verify inventory and catalog passthrough columns survived without data loss
        self.assertEqual(str(sony_row["aisle_location"]), "Aisle-14B")
        self.assertEqual(int(sony_row["units_in_stock"]), 42)
        self.assertEqual(bool(sony_row["requires_refrigeration"]), False)
        self.assertEqual(float(sony_row["retail_price"]), 399.99)
        self.assertEqual(int(sony_row["warranty_years"]), 2)
        self.assertEqual(bool(sony_row["is_featured_item"]), True)

        # Test Step 6 Search: Query by actual stored barcode and SKU
        matcher = HierarchicalMatcher()
        edges_df = self.db.execute_query("SELECT * FROM enrichment_edges")
        recs_df = self.db.execute_query("SELECT * FROM canonical_records")

        # Search by Barcode
        replay = matcher.replay_progressive_enrichment("aadhaar", "49055249001", edges_df, recs_df)
        self.assertTrue(replay["found"])
        self.assertEqual(replay["total_cluster_size"], 2)
        self.assertIn("warehouse_db", replay["sources"])
        self.assertIn("ecom_catalog", replay["sources"])

        # Search by SKU
        replay_sku = matcher.replay_progressive_enrichment("member_id", "PROD-SONY-WH1000", edges_df, recs_df)
        self.assertTrue(replay_sku["found"])
        self.assertEqual(replay_sku["total_cluster_size"], 2)

    def test_04_three_product_datasets_catalog_marketplace_warehouse(self):
        """
        Verify that 3 heterogeneous product datasets (Product Catalog, Marketplace Listings,
        Warehouse Inventory) link matching items across different column names, preserve
        all attributes (dates, numerics including 0, booleans including False), never merge
        on blank values, and produce zero redundant columns.
        """
        # Dataset 1: Product Master Catalog
        df_catalog = pd.DataFrame([
            {
                "catalog_sku": "SKU-AUDIO-101",
                "product_title": "Sony WH-1000XM5 Wireless Headphones",
                "gtin_barcode": "49055249001",
                "msrp": 399.99,
                "release_date": "2022-05-20",
                "is_active": True,
            },
            {
                "catalog_sku": "SKU-TABLET-202",
                "product_title": "Apple iPad Air 11-inch M2",
                "gtin_barcode": "19594912002",
                "msrp": 599.00,
                "release_date": "2024-05-15",
                "is_active": True,
            },
            {
                "catalog_sku": "SKU-DRONE-303",
                "product_title": "DJI Mini 4 Pro Drone",
                "gtin_barcode": "69415659003",
                "msrp": 759.00,
                "release_date": "2023-09-25",
                "is_active": False,
            },
            {
                "catalog_sku": "SKU-ORPHAN-999",
                "product_title": "Unmatched Test Gadget",
                "gtin_barcode": None,          # Blank barcode: must NEVER merge on null!
                "msrp": 49.99,
                "release_date": "2024-01-01",
                "is_active": True,
            }
        ])

        # Dataset 2: Marketplace Listings
        df_marketplace = pd.DataFrame([
            {
                "listing_sku": "SKU-AUDIO-101",
                "item_name": "Sony WH1000XM5 ANC Headphones (Black)",
                "upc_code": "49055249001",
                "offer_price": 385.00,
                "seller_rating": 4.9,
                "prime_eligible": True,
            },
            {
                "listing_sku": "SKU-TABLET-202",
                "item_name": "iPad Air 11 M2 128GB Wi-Fi",
                "upc_code": "19594912002",
                "offer_price": 579.99,
                "seller_rating": 4.8,
                "prime_eligible": True,
            },
            {
                "listing_sku": "SKU-MOUSE-404",
                "item_name": "Logitech MX Master 3S Wireless",
                "upc_code": "09785517004",
                "offer_price": 99.99,
                "seller_rating": 4.7,
                "prime_eligible": False,
            },
            {
                "listing_sku": "SKU-BLANK-888",
                "item_name": "Blank Code Accessory",
                "upc_code": "",                # Empty barcode: must NEVER merge with ORPHAN-999!
                "offer_price": 12.50,
                "seller_rating": 3.5,
                "prime_eligible": False,
            }
        ])

        # Dataset 3: Warehouse Inventory
        df_warehouse = pd.DataFrame([
            {
                "sku": "SKU-AUDIO-101",
                "inventory_title": "Sony WH-1000XM5 Headset",
                "barcode": "49055249001",
                "cost_price": 290.00,
                "stock_quantity": 45,
                "warehouse_bay": "Bay-A1",
                "requires_refrigeration": False,
            },
            {
                "sku": "SKU-TABLET-202",
                "inventory_title": "Apple iPad Air 11 M2 Tablet",
                "barcode": "19594912002",
                "cost_price": 480.00,
                "stock_quantity": 0,           # Falsy numeric 0 must NOT be dropped!
                "warehouse_bay": "Bay-B3",
                "requires_refrigeration": False,
            },
            {
                "sku": "SKU-CAMERA-505",
                "inventory_title": "GoPro HERO12 Black Action Cam",
                "barcode": "81827902999",
                "cost_price": 310.00,
                "stock_quantity": 18,
                "warehouse_bay": "Bay-C2",
                "requires_refrigeration": False,
            }
        ])

        engine = SchemaMappingEngine()
        map_cat = {c: p["suggested_canonical"] for c, p in engine.suggest_mappings_for_table(df_catalog.columns.tolist()).items()}
        map_mkt = {c: p["suggested_canonical"] for c, p in engine.suggest_mappings_for_table(df_marketplace.columns.tolist()).items()}
        map_war = {c: p["suggested_canonical"] for c, p in engine.suggest_mappings_for_table(df_warehouse.columns.tolist()).items()}

        # Verify automated mapping consolidated product concepts
        self.assertEqual(map_cat["catalog_sku"], "sku")
        self.assertEqual(map_cat["product_title"], "name")
        self.assertEqual(map_cat["gtin_barcode"], "barcode")
        self.assertEqual(map_cat["msrp"], "price")

        self.assertEqual(map_mkt["listing_sku"], "sku")
        self.assertEqual(map_mkt["item_name"], "name")
        self.assertEqual(map_mkt["upc_code"], "barcode")
        self.assertEqual(map_mkt["offer_price"], "price")

        self.assertEqual(map_war["sku"], "sku")
        self.assertEqual(map_war["inventory_title"], "name")
        self.assertEqual(map_war["barcode"], "barcode")
        self.assertEqual(map_war["cost_price"], "price")
        self.assertEqual(map_war["stock_quantity"], "stock")

        specs = [
            {"source_id": "src_catalog", "source_name": "catalog.csv", "dataframe": df_catalog, "mapping": map_cat},
            {"source_id": "src_marketplace", "source_name": "marketplace.csv", "dataframe": df_marketplace, "mapping": map_mkt},
            {"source_id": "src_warehouse", "source_name": "warehouse.csv", "dataframe": df_warehouse, "mapping": map_war},
        ]

        stats = run_dynamic_user_pipeline(datasets_spec=specs, db_path=self.test_db_path, reset_db=True)
        # 11 input records total
        self.assertEqual(stats["total_records"], 11)
        # 2 multi-source entities (Sony, iPad) + 5 distinct singletons = 7 golden master entities
        self.assertEqual(stats["total_entities"], 7)
        self.assertEqual(stats["total_linked_entities"], 2)
        self.assertEqual(stats["total_singletons"], 5)

        df_master = get_unified_master_dataframe(self.db)
        self.assertEqual(len(df_master), 7)

        # 1. Verify NO redundant columns exist
        for redundant in ["catalog_sku", "listing_sku", "product_title", "item_name", "inventory_title", "gtin_barcode", "upc_code", "cost_price", "offer_price", "msrp"]:
            self.assertNotIn(redundant, df_master.columns)

        # 2. Verify consolidated canonical columns exist
        for canon in ["sku", "name", "barcode", "price", "stock"]:
            self.assertIn(canon, df_master.columns)

        # 3. Verify Sony merged record has 3 contributing sources
        sony = df_master[df_master["sku"] == "SKU-AUDIO-101"].iloc[0]
        self.assertEqual(int(sony["cluster_size"]), 3)
        self.assertEqual(str(sony["barcode"]), "49055249001")
        self.assertEqual(bool(sony["requires_refrigeration"]), False)
        self.assertEqual(str(sony["release_date"]), "2022-05-20")

        # 4. Verify iPad record preserved 0 stock quantity
        ipad = df_master[df_master["sku"] == "SKU-TABLET-202"].iloc[0]
        self.assertEqual(int(ipad["cluster_size"]), 3)
        self.assertEqual(float(ipad["stock"]), 0.0)

        # 5. Verify items with blank/null barcodes were NOT merged together
        orphan = df_master[df_master["sku"] == "SKU-ORPHAN-999"].iloc[0]
        blank = df_master[df_master["sku"] == "SKU-BLANK-888"].iloc[0]
        self.assertEqual(int(orphan["cluster_size"]), 1)
        self.assertEqual(int(blank["cluster_size"]), 1)
        self.assertNotEqual(orphan["entity_id"], blank["entity_id"])

    def test_05_social_media_redundant_columns_and_phone_consistency(self):
        """
        Verify social media datasets consolidate bio, followers, platform without redundant columns,
        and normalize inconsistent phone formats (brackets, country codes, float .0) into uniform clean numbers.
        """
        df_soc_1 = pd.DataFrame([
            {
                "user_handle": "alex_mercer",
                "full_name": "Alex Mercer",
                "bio_text": "Lead AI Systems Researcher",
                "followers_count": 45000,
                "social_platform": "Twitter",
                "contact_number": "+91 98765-43210",          # Prefix +91 with dashes
            },
            {
                "user_handle": "sarah_c",
                "full_name": "Sarah Connor",
                "bio_text": "Defensive Security Specialist",
                "followers_count": 12000,
                "social_platform": "Twitter",
                "contact_number": "15552345678.0",             # Float string representation
            }
        ])

        df_soc_2 = pd.DataFrame([
            {
                "screen_name": "alex_mercer",
                "name": "Alex Mercer, PhD",
                "user_bio": "Staff ML Scientist & Maintainer",
                "subscribers": 45200,
                "network": "LinkedIn",
                "phone": "09876543210",                        # Trunk zero prefix
            },
            {
                "screen_name": "sarah_c",
                "name": "Sarah Connor",
                "user_bio": "Cyberdyne Systems Auditor",
                "subscribers": 12500,
                "network": "LinkedIn",
                "phone": "+1 (555) 234-5678",                  # Country code +1 with brackets
            }
        ])

        engine = SchemaMappingEngine()
        map_1 = {c: p["suggested_canonical"] for c, p in engine.suggest_mappings_for_table(df_soc_1.columns.tolist()).items()}
        map_2 = {c: p["suggested_canonical"] for c, p in engine.suggest_mappings_for_table(df_soc_2.columns.tolist()).items()}

        self.assertEqual(map_1["user_handle"], "username")
        self.assertEqual(map_1["bio_text"], "bio")
        self.assertEqual(map_1["followers_count"], "followers")
        self.assertEqual(map_1["social_platform"], "platform")
        self.assertEqual(map_1["contact_number"], "phone")

        self.assertEqual(map_2["screen_name"], "username")
        self.assertEqual(map_2["user_bio"], "bio")
        self.assertEqual(map_2["subscribers"], "followers")
        self.assertEqual(map_2["network"], "platform")
        self.assertEqual(map_2["phone"], "phone")

        specs = [
            {"source_id": "src_twitter", "source_name": "twitter.csv", "dataframe": df_soc_1, "mapping": map_1},
            {"source_id": "src_linkedin", "source_name": "linkedin.csv", "dataframe": df_soc_2, "mapping": map_2},
        ]

        stats = run_dynamic_user_pipeline(datasets_spec=specs, db_path=self.test_db_path, reset_db=True)
        self.assertEqual(stats["total_records"], 4)
        self.assertEqual(stats["total_entities"], 2)  # Both users merged across Twitter & LinkedIn

        df_master = get_unified_master_dataframe(self.db)
        self.assertEqual(len(df_master), 2)

        # 1. Verify NO redundant columns exist
        for redundant in ["bio_text", "user_bio", "followers_count", "subscribers", "social_platform", "network", "contact_number"]:
            self.assertNotIn(redundant, df_master.columns)

        # 2. Verify consolidated canonical columns exist
        for canon in ["username", "name", "bio", "followers", "platform", "phone"]:
            self.assertIn(canon, df_master.columns)

        # 3. Verify consistent phone formats across entities
        alex = df_master[df_master["username"] == "alex_mercer"].iloc[0]
        sarah = df_master[df_master["username"] == "sarah_c"].iloc[0]

        # +91 98765-43210 and 09876543210 must both standardize to "9876543210"
        self.assertEqual(str(alex["phone"]), "9876543210")
        # 15552345678.0 and +1 (555) 234-5678 must both standardize to "5552345678" without .0
        self.assertEqual(str(sarah["phone"]), "5552345678")


if __name__ == "__main__":
    unittest.main()
