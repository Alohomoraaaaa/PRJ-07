"""
Gemini AI-Assisted Schema Mapping Engine (PRJ-07).
Uses Gemini 2.5 Flash-Lite to intelligently suggest canonical field mappings
by analyzing column names and 2-3 sample values per column.
Includes deterministic fallback when offline or without API key.
"""

import os
import re
import json
import requests
import pandas as pd
from typing import Dict, List, Any, Optional

from src.config import CANONICAL_FIELDS
from src.mapping import SchemaMappingEngine


def get_gemini_api_key() -> Optional[str]:
    """Retrieve Gemini API Key from Streamlit secrets, environment, or .env file."""
    # 1. Check Streamlit secrets (Streamlit Cloud deployment)
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for var in ["GEMINI_API_KEY", "GEMINIY_API_KEY", "GOOGLE_API_KEY"]:
                if var in st.secrets and st.secrets[var]:
                    return str(st.secrets[var]).strip().strip("'\"")
    except Exception:
        pass

    # 2. Check OS environment variables
    for var in ["GEMINI_API_KEY", "GEMINIY_API_KEY", "GOOGLE_API_KEY"]:
        val = os.environ.get(var)
        if val and val.strip():
            return val.strip().strip("'\"")

    # 3. Check local .env file
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k in ("GEMINI_API_KEY", "GEMINIY_API_KEY", "GOOGLE_API_KEY") and v:
                            return v
        except Exception:
            pass
    return None


class GeminiSchemaMapper:
    """
    Intelligent schema mapper utilizing Gemini Flash-Lite.
    Never sends full datasets — only column names and 2-3 sample values.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_gemini_api_key()
        self.fallback_engine = SchemaMappingEngine()
        self.models_to_try = [
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",
            "gemini-2.0-flash",
        ]

    def _extract_samples(self, df: pd.DataFrame, max_samples: int = 3) -> Dict[str, List[str]]:
        """Extract up to 3 non-null string samples for each column."""
        samples = {}
        for col in df.columns:
            non_nulls = df[col].dropna().astype(str).str.strip()
            valid_samples = [s for s in non_nulls if s and s.lower() not in ("nan", "none", "null", "")]
            samples[str(col)] = valid_samples[:max_samples]
        return samples

    def suggest_mappings(self, df: pd.DataFrame, dataset_name: str = "") -> Dict[str, Dict[str, Any]]:
        """
        Suggest canonical field mappings for a DataFrame.
        Tries Gemini Flash-Lite first; falls back to deterministic engine on failure.
        """
        columns = [str(c) for c in df.columns]
        samples = self._extract_samples(df)

        if not self.api_key:
            # Deterministic fallback
            return self.fallback_engine.suggest_mappings_for_table(columns)

        canonical_list = list(CANONICAL_FIELDS.keys()) + [
            "bio", "followers", "following", "platform", "department", "salary"
        ]
        canonical_desc = ", ".join(canonical_list)

        prompt = f"""You are an expert Data Engineer specializing in Schema Matching and Entity Resolution.
Given the dataset columns and representative sample values below, map each column to the most appropriate canonical concept from the provided allowed list, or suggest a clean canonical snake_case attribute name if it represents a distinct semantic concept:

Allowed Canonical Concepts:
{canonical_desc}

Dataset Columns and Sample Values:
{json.dumps(samples, indent=2)}

Mapping Guidelines:
- "email", "email_id", "email_address", "contact_email" -> "email"
- "phone", "mobile_number", "contact_no", "phone_number", "mobile" -> "phone"
- "full_name", "name", "customer_name", "user_name" -> "name"
- "first_name", "fname", "given_name" -> "first_name"
- "last_name", "lname", "surname" -> "last_name"
- "aadhaar", "aadhar_number", "aadhaar_no", "national_id" -> "aadhaar"
- "username", "handle", "account_name" -> "username"
- "city", "city_name", "town", "locality" -> "city"
- "address", "residential_address", "street", "residence" -> "address"
- "company", "organization", "workplace", "employer" -> "company"
- "bio", "bio_text", "biography", "user_bio", "about_me" -> "bio"
- "followers", "followers_count", "follower_count", "subscribers" -> "followers"
- "following", "following_count" -> "following"
- "platform", "social_platform", "network" -> "platform"
- "department", "dept", "division" -> "department"
- "salary", "compensation", "wage", "ctc" -> "salary"
- "rating", "score", "stars" -> "rating"
- "review_count", "reviews" -> "review_count"
- "category", "industry", "type" -> "category"
- "latitude", "lat" -> "latitude"
- "longitude", "lng", "lon" -> "longitude"
- "notes", "skills", "details", "remarks" -> "notes"
- If a column is a metadata/custom column, keep its clean normalized attribute name or map to "unmapped".

Return ONLY a valid JSON object mapping each original column name to an object with:
"suggested_canonical": (one of the allowed concepts or clean snake_case semantic name),
"confidence": (float between 0.0 and 1.0),
"reason": (concise 1-sentence explanation).

Example JSON format:
{{
  "email_id": {{"suggested_canonical": "email", "confidence": 1.0, "reason": "Standard email address header and format"}},
  "full_name": {{"suggested_canonical": "name", "confidence": 0.98, "reason": "Full person name with title formatting"}}
}}
"""

        for model_name in self.models_to_try:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
                headers = {"Content-Type": "application/json"}
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.1,
                        "responseMimeType": "application/json",
                    }
                }

                resp = requests.post(url, headers=headers, json=payload, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
                    clean_text = re.sub(r"^```json\s*", "", raw_text.strip())
                    clean_text = re.sub(r"\s*```$", "", clean_text.strip())
                    parsed = json.loads(clean_text)

                    # Validate output contains all columns
                    results = {}
                    for col in columns:
                        if col in parsed and isinstance(parsed[col], dict):
                            sug = parsed[col].get("suggested_canonical", "unmapped")
                            if sug not in canonical_list and sug != "unmapped":
                                sug = "unmapped"
                            results[col] = {
                                "raw_column": col,
                                "suggested_canonical": sug,
                                "confidence": float(parsed[col].get("confidence", 0.90)),
                                "reason": f"[AI] {parsed[col].get('reason', 'Gemini AI Suggested')}",
                                "is_approved": False,
                            }
                        else:
                            fallback_single = self.fallback_engine.suggest_single_column_mapping(col)
                            results[col] = fallback_single
                    return results
            except Exception as e:
                continue

        # Fallback if API calls fail
        return self.fallback_engine.suggest_mappings_for_table(columns)
