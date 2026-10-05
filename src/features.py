"""
Similarity Feature Engineering Module (Phase 4).
Extracts fine-grained field-level similarity features across candidate record pairs
using string distances (Jaro-Winkler, Levenshtein, Token Jaccard), phonetic encodings,
and numeric/categorical exact comparisons.
"""

import os
import re
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np


def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1, max_l: int = 4) -> float:
    """
    Compute Jaro-Winkler similarity between two strings.
    Gives higher weight to common prefixes, ideal for person names.
    Returns float in [0.0, 1.0].
    """
    if not isinstance(s1, str) or not isinstance(s2, str):
        return 0.0
    s1, s2 = s1.strip().lower(), s2.strip().lower()
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    match_distance = max(len1, len2) // 2 - 1

    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    transpositions = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2.0) / matches) / 3.0

    # Common prefix adjustment
    prefix_len = 0
    for i in range(min(len1, len2, max_l)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break

    return float(jaro + prefix_len * p * (1.0 - jaro))


def levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Compute normalized Levenshtein similarity: 1.0 - (edit_distance / max_len).
    Returns float in [0.0, 1.0].
    """
    if not isinstance(s1, str) or not isinstance(s2, str):
        return 0.0
    s1, s2 = s1.strip().lower(), s2.strip().lower()
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    dist = dp[len1][len2]
    return float(1.0 - (dist / max(len1, len2)))


def token_jaccard_similarity(s1: str, s2: str) -> float:
    """
    Compute Jaccard token overlap between two strings (order-independent).
    Ideal for street addresses where word order can fluctuate.
    """
    if not isinstance(s1, str) or not isinstance(s2, str):
        return 0.0
    t1 = set(re.findall(r"\w+", s1.lower()))
    t2 = set(re.findall(r"\w+", s2.lower()))
    if not t1 or not t2:
        return 0.0
    intersection = t1.intersection(t2)
    union = t1.union(t2)
    return float(len(intersection) / len(union))


def exact_match(s1: Any, s2: Any) -> float:
    """Check binary equality between two normalized fields (ignoring empty strings)."""
    if s1 is None or s2 is None:
        return 0.0
    str1 = str(s1).strip().lower()
    str2 = str(s2).strip().lower()
    if not str1 or not str2 or str1 in ("nan", "none") or str2 in ("nan", "none"):
        return 0.0
    return 1.0 if str1 == str2 else 0.0


def extract_pairwise_features(
    candidates_df: pd.DataFrame,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build similarity feature matrix for all candidate pairs.

    Parameters:
        candidates_df: DataFrame with ['rec_id_a', 'rec_id_b', 'match_label', 'num_passes_triggered']
        df_a: Preprocessed Source A DataFrame (indexed or searchable by rec_id)
        df_b: Preprocessed Source B DataFrame (indexed or searchable by rec_id)

    Returns:
        features_df: DataFrame with rec_id keys, all computed similarity features, and target match_label
    """
    # Create fast dict lookups by rec_id
    dict_a = df_a.set_index("rec_id").to_dict(orient="index")
    dict_b = df_b.set_index("rec_id").to_dict(orient="index")

    feature_rows = []

    for _, row in candidates_df.iterrows():
        id_a = row["rec_id_a"]
        id_b = row["rec_id_b"]
        label = row.get("match_label", 0)
        passes_trig = row.get("num_passes_triggered", 1)

        rec_a = dict_a.get(id_a, {})
        rec_b = dict_b.get(id_b, {})

        # Field extractions
        gname_a = rec_a.get("given_name", "")
        gname_b = rec_b.get("given_name", "")

        sname_a = rec_a.get("surname", "")
        sname_b = rec_b.get("surname", "")

        addr_a = rec_a.get("address_1", "")
        addr_b = rec_b.get("address_1", "")

        street_no_a = rec_a.get("street_number", "")
        street_no_b = rec_b.get("street_number", "")

        suburb_a = rec_a.get("suburb", "")
        suburb_b = rec_b.get("suburb", "")

        postcode_a = rec_a.get("postcode", "")
        postcode_b = rec_b.get("postcode", "")

        state_a = rec_a.get("state", "")
        state_b = rec_b.get("state", "")

        dob_a = rec_a.get("date_of_birth", "")
        dob_b = rec_b.get("date_of_birth", "")

        ssn_a = rec_a.get("soc_sec_id", "")
        ssn_b = rec_b.get("soc_sec_id", "")

        sndx_sname_a = rec_a.get("soundex_surname", "")
        sndx_sname_b = rec_b.get("soundex_surname", "")

        sndx_gname_a = rec_a.get("soundex_given", "")
        sndx_gname_b = rec_b.get("soundex_given", "")

        # 1. Given Name similarities
        feat_gname_jw = jaro_winkler_similarity(gname_a, gname_b)
        feat_gname_lev = levenshtein_similarity(gname_a, gname_b)
        feat_gname_exact = exact_match(gname_a, gname_b)
        feat_gname_soundex = exact_match(sndx_gname_a, sndx_gname_b)

        # 2. Surname similarities
        feat_sname_jw = jaro_winkler_similarity(sname_a, sname_b)
        feat_sname_lev = levenshtein_similarity(sname_a, sname_b)
        feat_sname_exact = exact_match(sname_a, sname_b)
        feat_sname_soundex = exact_match(sndx_sname_a, sndx_sname_b)

        # 3. Address similarities
        feat_addr_jw = jaro_winkler_similarity(addr_a, addr_b)
        feat_addr_lev = levenshtein_similarity(addr_a, addr_b)
        feat_addr_jaccard = token_jaccard_similarity(addr_a, addr_b)
        feat_street_no_exact = exact_match(street_no_a, street_no_b)

        # 4. Suburb similarities
        feat_suburb_jw = jaro_winkler_similarity(suburb_a, suburb_b)
        feat_suburb_lev = levenshtein_similarity(suburb_a, suburb_b)
        feat_suburb_exact = exact_match(suburb_a, suburb_b)

        # 5. Postcode & State similarities
        feat_postcode_exact = exact_match(postcode_a, postcode_b)
        feat_postcode_lev = levenshtein_similarity(postcode_a, postcode_b)
        feat_state_exact = exact_match(state_a, state_b)

        # 6. Date of Birth similarities
        feat_dob_exact = exact_match(dob_a, dob_b)
        feat_dob_lev = levenshtein_similarity(dob_a, dob_b)
        feat_dob_year_match = 1.0 if (len(dob_a) >= 4 and len(dob_b) >= 4 and dob_a[:4] == dob_b[:4]) else 0.0

        # 7. Social Security ID similarities
        feat_ssn_exact = exact_match(ssn_a, ssn_b)
        feat_ssn_lev = levenshtein_similarity(ssn_a, ssn_b)

        # 8. Full Name compound similarity
        full_name_a = f"{gname_a} {sname_a}".strip()
        full_name_b = f"{gname_b} {sname_b}".strip()
        feat_full_name_jw = jaro_winkler_similarity(full_name_a, full_name_b)

        feature_rows.append({
            "rec_id_a": id_a,
            "rec_id_b": id_b,
            # Name features
            "given_name_jw": round(feat_gname_jw, 4),
            "given_name_lev": round(feat_gname_lev, 4),
            "given_name_exact": feat_gname_exact,
            "given_name_soundex": feat_gname_soundex,
            "surname_jw": round(feat_sname_jw, 4),
            "surname_lev": round(feat_sname_lev, 4),
            "surname_exact": feat_sname_exact,
            "surname_soundex": feat_sname_soundex,
            "full_name_jw": round(feat_full_name_jw, 4),
            # Address features
            "address_1_jw": round(feat_addr_jw, 4),
            "address_1_lev": round(feat_addr_lev, 4),
            "address_1_jaccard": round(feat_addr_jaccard, 4),
            "street_number_exact": feat_street_no_exact,
            # Suburb features
            "suburb_jw": round(feat_suburb_jw, 4),
            "suburb_lev": round(feat_suburb_lev, 4),
            "suburb_exact": feat_suburb_exact,
            # Postcode & State
            "postcode_exact": feat_postcode_exact,
            "postcode_lev": round(feat_postcode_lev, 4),
            "state_exact": feat_state_exact,
            # Date of Birth
            "dob_exact": feat_dob_exact,
            "dob_lev": round(feat_dob_lev, 4),
            "dob_year_match": feat_dob_year_match,
            # Social Security ID
            "ssn_exact": feat_ssn_exact,
            "ssn_lev": round(feat_ssn_lev, 4),
            # Meta blocking features
            "num_passes_triggered": passes_trig,
            # Ground Truth Target
            "match_label": label,
        })

    return pd.DataFrame(feature_rows)


def analyze_feature_dataset(features_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute statistical distributions, class balance, and feature summaries across match classes.
    """
    total_pairs = len(features_df)
    pos_mask = features_df["match_label"] == 1
    neg_mask = features_df["match_label"] == 0

    num_pos = int(pos_mask.sum())
    num_neg = int(neg_mask.sum())

    feature_cols = [c for c in features_df.columns if c not in ("rec_id_a", "rec_id_b", "match_label")]

    feature_stats = {}
    for col in feature_cols:
        mean_pos = float(features_df.loc[pos_mask, col].mean()) if num_pos > 0 else 0.0
        mean_neg = float(features_df.loc[neg_mask, col].mean()) if num_neg > 0 else 0.0
        std_pos = float(features_df.loc[pos_mask, col].std()) if num_pos > 0 else 0.0
        std_neg = float(features_df.loc[neg_mask, col].std()) if num_neg > 0 else 0.0

        feature_stats[col] = {
            "mean_match": round(mean_pos, 4),
            "mean_non_match": round(mean_neg, 4),
            "delta_mean": round(mean_pos - mean_neg, 4),
            "std_match": round(std_pos, 4),
            "std_non_match": round(std_neg, 4),
        }

    return {
        "total_candidate_pairs": total_pairs,
        "positive_matches": num_pos,
        "negative_matches": num_neg,
        "positive_class_ratio_pct": round(num_pos / total_pairs * 100.0, 2) if total_pairs > 0 else 0.0,
        "negative_class_ratio_pct": round(num_neg / total_pairs * 100.0, 2) if total_pairs > 0 else 0.0,
        "class_imbalance_ratio": f"1 : {round(num_neg / num_pos, 2)}" if num_pos > 0 else "N/A",
        "feature_statistics": feature_stats,
    }


def generate_all_feature_datasets(
    processed_dir: str = "data/processed",
) -> Dict[str, pd.DataFrame]:
    """
    Generate feature matrices for the full dataset and the disjoint train/test partitions.
    """
    try:
        from src.blocking import MultiPassBlocker
    except ImportError:
        from blocking import MultiPassBlocker

    # Load cleaned datasets
    df_4a = pd.read_csv(os.path.join(processed_dir, "cleaned_febrl4a.csv"), dtype=str).fillna("")
    df_4b = pd.read_csv(os.path.join(processed_dir, "cleaned_febrl4b.csv"), dtype=str).fillna("")

    train_a = pd.read_csv(os.path.join(processed_dir, "train_febrl4a.csv"), dtype=str).fillna("")
    train_b = pd.read_csv(os.path.join(processed_dir, "train_febrl4b.csv"), dtype=str).fillna("")
    test_a = pd.read_csv(os.path.join(processed_dir, "test_febrl4a.csv"), dtype=str).fillna("")
    test_b = pd.read_csv(os.path.join(processed_dir, "test_febrl4b.csv"), dtype=str).fillna("")

    blocker = MultiPassBlocker()

    # 1. Full Dataset
    cand_full, _ = blocker.generate_candidates(df_4a, df_4b)
    feats_full = extract_pairwise_features(cand_full, df_4a, df_4b)
    feats_full.to_csv(os.path.join(processed_dir, "candidate_features.csv"), index=False)

    # 2. Train Partition (Disjoint)
    cand_train, _ = blocker.generate_candidates(train_a, train_b)
    feats_train = extract_pairwise_features(cand_train, train_a, train_b)
    feats_train.to_csv(os.path.join(processed_dir, "train_features.csv"), index=False)

    # 3. Test Partition (Disjoint)
    cand_test, _ = blocker.generate_candidates(test_a, test_b)
    feats_test = extract_pairwise_features(cand_test, test_a, test_b)
    feats_test.to_csv(os.path.join(processed_dir, "test_features.csv"), index=False)

    return {
        "full": feats_full,
        "train": feats_train,
        "test": feats_test,
    }


if __name__ == "__main__":
    try:
        from src.preprocess import run_preprocessing_pipeline
        from src.blocking import MultiPassBlocker
    except ImportError:
        from preprocess import run_preprocessing_pipeline
        from blocking import MultiPassBlocker

    print("Executing Phase 4: Similarity Feature Engineering...")
    prep = run_preprocessing_pipeline()

    feature_sets = generate_all_feature_datasets()
    features_df = feature_sets["full"]
    analysis = analyze_feature_dataset(features_df)

    print("\n" + "=" * 80)
    print("PHASE 4: SIMILARITY FEATURE DATASET ANALYSIS")
    print("=" * 80)
    print(f"Total Candidate Pairs:      {analysis['total_candidate_pairs']:,}")
    print(f"Positive Matches (Class 1): {analysis['positive_matches']:,} ({analysis['positive_class_ratio_pct']}%)")
    print(f"Negative Matches (Class 0): {analysis['negative_matches']:,} ({analysis['negative_class_ratio_pct']}%)")
    print(f"Class Balance Ratio:        {analysis['class_imbalance_ratio']}")

    print("\n--- Feature Mean Comparison (Matches vs Non-Matches) ---")
    stats_df = pd.DataFrame(analysis["feature_statistics"]).T
    stats_df = stats_df.sort_values(by="delta_mean", ascending=False)
    print(stats_df[["mean_match", "mean_non_match", "delta_mean"]].to_string())
