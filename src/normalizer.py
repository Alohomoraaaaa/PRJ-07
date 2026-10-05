"""
Data Cleaning & Normalization Engine (PRJ-07).
Normalizes diverse field values (email, phone, name, address, username, member_id,
rating, category, review_count, latitude, longitude, city, business_name)
while guaranteeing dual-storage traceability (complete raw row vs normalized values).
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional


class DataNormalizer:
    """
    Deterministic field normalizer supporting dual-storage traceability.
    """

    @staticmethod
    def is_null_val(val: Any) -> bool:
        """Check if a raw field value is null or empty."""
        if val is None:
            return True
        if isinstance(val, float) and (np.isnan(val) or pd.isna(val)):
            return True
        s = str(val).strip()
        if not s or s.lower() in ("nan", "none", "null", "undefined", "n/a", "na", "-", ""):
            return True
        return False

    @classmethod
    def normalize_email(cls, val: Any) -> Tuple[str, bool]:
        """Normalize email address (strip, lowercase, validate structure)."""
        if cls.is_null_val(val):
            return "", True
        s = str(val).strip().lower()
        if re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", s):
            return s, True
        return s, False

    @classmethod
    def normalize_phone(cls, val: Any) -> Tuple[str, bool]:
        """
        Normalize phone numbers by stripping non-digit characters and standardizing prefixes.
        Converts +91 98765-43210, 09876543210, (022) 88776655 -> clean numeric representation.
        """
        if cls.is_null_val(val):
            return "", True
        s = str(val).strip()
        digits = re.sub(r"\D", "", s)

        # Standard Indian / General mobile prefixes
        if len(digits) == 12 and digits.startswith("91"):
            digits = digits[2:]
        elif len(digits) == 11 and digits.startswith("0"):
            digits = digits[1:]
        elif len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]

        is_valid = len(digits) >= 7 and len(digits) <= 15
        return digits, is_valid

    @classmethod
    def normalize_name(cls, val: Any) -> Tuple[str, bool]:
        """Normalize person name or business title (trim, collapse whitespace, title case)."""
        if cls.is_null_val(val):
            return "", True
        s = str(val).strip()
        clean = re.sub(r"\s+", " ", s).title()
        return clean, True

    @classmethod
    def normalize_username(cls, val: Any) -> Tuple[str, bool]:
        """Normalize online handle / username (lowercase, trim)."""
        if cls.is_null_val(val):
            return "", True
        s = str(val).strip().lower()
        clean = re.sub(r"\s+", "_", s)
        return clean, True

    @classmethod
    def normalize_member_id(cls, val: Any) -> Tuple[str, bool]:
        """Normalize member/employee identifier (uppercase, trim)."""
        if cls.is_null_val(val):
            return "", True
        s = str(val).strip().upper()
        return s, True

    @classmethod
    def normalize_address(cls, val: Any) -> Tuple[str, bool]:
        """Normalize street address / city / state / location."""
        if cls.is_null_val(val):
            return "", True
        s = str(val).strip()
        clean = re.sub(r"\s+", " ", s).title()
        return clean, True

    @classmethod
    def normalize_company(cls, val: Any) -> Tuple[str, bool]:
        """Normalize company / organization name."""
        if cls.is_null_val(val):
            return "", True
        s = str(val).strip()
        clean = re.sub(r"\s+", " ", s).title()
        return clean, True

    @classmethod
    def normalize_aadhaar(cls, val: Any) -> Tuple[str, bool]:
        """Normalize Aadhaar / National ID (remove whitespace, dashes, non-digits)."""
        if cls.is_null_val(val):
            return "", True
        s = str(val).strip()
        digits = re.sub(r"\D", "", s)
        is_valid = len(digits) == 12 or len(digits) >= 6
        return digits, is_valid

    @classmethod
    def normalize_numeric_field(cls, val: Any) -> Tuple[str, bool]:
        """Normalize numeric coordinates, ratings, and counts (latitude, longitude, rating, review_count)."""
        if cls.is_null_val(val):
            return "", True
        s = str(val).strip()
        if s.endswith(".0") and not ("." in s[:-2]):
            s = s[:-2]
        return s, True

    @classmethod
    def normalize_value_by_type(cls, value: Any, field_type: str) -> Tuple[str, bool]:
        """Dispatch normalizer by canonical field type."""
        if field_type == "email":
            return cls.normalize_email(value)
        elif field_type == "phone":
            return cls.normalize_phone(value)
        elif field_type == "aadhaar":
            return cls.normalize_aadhaar(value)
        elif field_type in ("name", "business_name", "first_name", "last_name"):
            return cls.normalize_name(value)
        elif field_type == "username":
            return cls.normalize_username(value)
        elif field_type == "member_id":
            return cls.normalize_member_id(value)
        elif field_type in ("address", "city"):
            return cls.normalize_address(value)
        elif field_type == "company":
            return cls.normalize_company(value)
        elif field_type in ("rating", "review_count", "latitude", "longitude", "numeric"):
            return cls.normalize_numeric_field(value)
        else:
            if cls.is_null_val(value):
                return "", True
            s = str(value).strip()
            return s, True

    @classmethod
    def process_raw_record(
        cls,
        raw_row: Dict[str, Any],
        column_mapping: Dict[str, str],
    ) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, bool]]:
        """
        Process a single raw source row.
        Guarantees that ALL original raw columns are preserved in raw_fields.
        Returns:
            raw_fields: Dict of original raw string values {raw_col: raw_val}
            norm_fields: Dict of canonical normalized string values {canonical_field: norm_val}
            validity_flags: Dict {canonical_field: is_valid}
        """
        raw_fields: Dict[str, str] = {}
        norm_fields: Dict[str, str] = {}
        validity_flags: Dict[str, bool] = {}

        for raw_col, raw_val in raw_row.items():
            if cls.is_null_val(raw_val):
                str_val = ""
            else:
                str_val = str(raw_val).strip()

            raw_fields[raw_col] = str_val

            canonical = column_mapping.get(raw_col, "unmapped")
            if canonical != "unmapped":
                norm_val, is_valid = cls.normalize_value_by_type(str_val, canonical)
                norm_fields[canonical] = norm_val
                validity_flags[canonical] = is_valid

        # Automatic synthesis of full name if first_name / last_name present but name is absent
        if "name" not in norm_fields or not norm_fields["name"]:
            fname = norm_fields.get("first_name") or raw_fields.get("first_name", "")
            lname = norm_fields.get("last_name") or raw_fields.get("last_name", "")
            if fname or lname:
                comb_name, _ = cls.normalize_name(f"{fname} {lname}")
                if comb_name:
                    norm_fields["name"] = comb_name
                    validity_flags["name"] = True

        return raw_fields, norm_fields, validity_flags
