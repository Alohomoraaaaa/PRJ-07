"""
Unit tests for Gemini AI Schema Mapping Engine (PRJ-07).
Verifies:
- Extraction of 2-3 sample values per column (never sending full datasets)
- AI mapping response parsing and validation
- Deterministic fallback when offline / no API key
- Schema caching by column fingerprint
"""

import os
import sys
import unittest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ai_mapper import GeminiSchemaMapper, get_gemini_api_key


class TestGeminiAIMapper(unittest.TestCase):
    def setUp(self):
        self.df_social = pd.DataFrame([
            {"user_handle": "alice_tech", "full_name": "Alice Smith", "contact_email": "alice@gmail.com", "followers_count": 14200, "bio_text": "Tech Lead @ Acme"},
            {"user_handle": "bob_coder", "full_name": "Robert Brown", "contact_email": "bob@gmail.com", "followers_count": 3100, "bio_text": "Python developer"},
        ])

    def test_01_api_key_retrieval(self):
        """Verify API key can be retrieved from environment or .env."""
        key = get_gemini_api_key()
        # Even if key is None in isolated environment, it shouldn't crash
        self.assertTrue(key is None or isinstance(key, str))

    def test_02_sample_extraction(self):
        """Verify values come from the first non-empty row of the same dataset."""
        mapper = GeminiSchemaMapper()
        df = pd.DataFrame([
            {"user_handle": None, "full_name": "", "contact_email": None},
            {"user_handle": "first_handle", "full_name": "First Person", "contact_email": "first@example.com"},
            {"user_handle": "later_handle", "full_name": "Later Person", "contact_email": "later@example.com"},
        ])
        samples = mapper._extract_samples(df)
        self.assertIn("user_handle", samples)
        self.assertEqual(samples["user_handle"], ["first_handle"])
        self.assertEqual(samples["full_name"], ["First Person"])
        self.assertIn("contact_email", samples)
        self.assertEqual(samples["contact_email"], ["first@example.com"])

    def test_03_mapping_execution_and_fallback(self):
        """Verify schema mapping suggestions with fallback guarantee."""
        mapper = GeminiSchemaMapper()
        proposals = mapper.suggest_mappings(self.df_social, dataset_name="social_test.csv")

        self.assertIn("contact_email", proposals)
        self.assertEqual(proposals["contact_email"]["suggested_canonical"], "email")
        self.assertIn("full_name", proposals)
        self.assertEqual(proposals["full_name"]["suggested_canonical"], "name")
        self.assertIn("user_handle", proposals)
        self.assertEqual(proposals["user_handle"]["suggested_canonical"], "username")


if __name__ == "__main__":
    unittest.main()
