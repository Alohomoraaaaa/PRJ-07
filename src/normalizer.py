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
        """Check if a raw field value is null or empty across all data types."""
        if val is None:
            return True
        try:
            if pd.isna(val):
                return True
        except Exception:
            pass
        s = str(val).strip()
        if not s or s.lower() in ("nan", "none", "null", "undefined", "n/a", "na", "-", "<na>", "nat"):
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
        Converts +91 98765-43210, 09876543210, (555) 234-5678, 9876543210.0 -> clean standard representation.
        """
        if cls.is_null_val(val):
            return "", True
        if isinstance(val, (float, np.floating)):
            if np.isnan(val):
                return "", True
            s = f"{int(val)}" if val.is_integer() else str(val)
        else:
            s = str(val).strip()
            if s.endswith(".0"):
                s = s[:-2]

        digits = re.sub(r"\D", "", s)

        # Standard international and domestic trunk prefixes
        if digits.startswith("00"):
            digits = digits[2:]
        if len(digits) == 12 and digits.startswith("91"):
            digits = digits[2:]
        elif len(digits) == 11 and digits.startswith("0"):
            digits = digits[1:]
        elif len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]

        is_valid = 7 <= len(digits) <= 15
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
    def normalize_sku(cls, val: Any) -> Tuple[str, bool]:
        """Normalize product SKU / part number (strip, uppercase, standardize separators)."""
        if cls.is_null_val(val):
            return "", True
        s = str(val).strip().upper()
        if s.endswith(".0"):
            s = s[:-2]
        s = re.sub(r"[\s_]+", "-", s)
        is_valid = len(s) >= 2
        return s, is_valid

    @classmethod
    def normalize_barcode(cls, val: Any) -> Tuple[str, bool]:
        """Normalize barcode, UPC, EAN, or GTIN (clean alphanumeric digits, remove dots)."""
        if cls.is_null_val(val):
            return "", True
        s = str(val).strip()
        if s.endswith(".0"):
            s = s[:-2]
        clean = re.sub(r"[^A-Za-z0-9]", "", s).upper()
        is_valid = len(clean) >= 4
        return clean, is_valid

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
        elif field_type == "sku":
            return cls.normalize_sku(value)
        elif field_type == "barcode":
            return cls.normalize_barcode(value)
        elif field_type in ("name", "business_name", "first_name", "last_name"):
            return cls.normalize_name(value)
        elif field_type == "username":
            return cls.normalize_username(value)
        elif field_type in ("member_id", "uid"):
            return cls.normalize_member_id(value)
        elif field_type in ("address", "city"):
            return cls.normalize_address(value)
        elif field_type == "company":
            return cls.normalize_company(value)
        elif field_type in ("rating", "review_count", "latitude", "longitude", "numeric", "price", "stock", "followers", "following"):
            return cls.normalize_numeric_field(value)
        else:
            if cls.is_null_val(value):
                return "", True
            s = str(value).strip()
            return s, True

    @classmethod
    def preserve_typed_val(cls, val: Any) -> Any:
        """Preserve native typed values (bool, int, float, str) without forcing everything to string."""
        if cls.is_null_val(val):
            return None
        if isinstance(val, (bool, np.bool_)):
            return bool(val)
        if isinstance(val, (int, np.integer)):
            return int(val)
        if isinstance(val, (float, np.floating)):
            return float(val)
        if hasattr(val, "isoformat"):
            return val.isoformat()
        return str(val).strip()

    @classmethod
    def process_raw_record(
        cls,
        raw_row: Dict[str, Any],
        column_mapping: Dict[str, str],
    ) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, bool]]:
        """
        Process a single raw source row.
        Guarantees that ALL original raw columns are preserved in raw_fields with native types.
        Returns:
            raw_fields: Dict of original raw values {raw_col: typed_val}
            norm_fields: Dict of canonical normalized string values {canonical_field: norm_val}
            validity_flags: Dict {canonical_field: is_valid}
        """
        raw_fields: Dict[str, Any] = {}
        norm_fields: Dict[str, str] = {}
        validity_flags: Dict[str, bool] = {}

        for raw_col, raw_val in raw_row.items():
            typed_val = cls.preserve_typed_val(raw_val)
            raw_fields[raw_col] = typed_val
            str_val = "" if typed_val is None else str(typed_val).strip()

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
