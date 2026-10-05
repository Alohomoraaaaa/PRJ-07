"""
Main Entry Point for PRJ-07: Multi-Database Entity Resolution Platform.
Executes multi-database ingestion, schema mapping, normalization, inverted indexing,
Union-Find clustering, enrichment edge logging, and master repository synthesis.
"""

import sys
import argparse
from src.pipeline import run_full_demonstration_pipeline
from src.database import get_db_manager


def main():
    parser = argparse.ArgumentParser(description="Multi-Database Entity Resolution & Unified Data Repository")
    parser.add_argument("--demo", action="store_true", help="Run full demonstration suite across Datasets A, B, C, D")
    parser.add_argument("--stats", action="store_true", help="Print current repository statistics")
    args = parser.parse_args()

    print("=" * 80)
    print("MULTI-DATABASE ENTITY RESOLUTION & UNIFIED DATA REPOSITORY (PRJ-07)")
    print("=" * 80)

    stats = run_full_demonstration_pipeline(include_incremental_d=True)

    print("\n[SUCCESS] Pipeline Execution Completed Successfully:")
    print(f"  • Total Data Sources Ingested:    {stats['total_sources']}")
    print(f"  • Total Ingested Canonical Rows:  {stats['total_records']}")
    print(f"  • Total Unified Master Entities:  {stats['total_entities']}")
    print(f"  • Total Enrichment Edges Logged:  {stats['total_enrichment_edges']}")
    print(f"  • Total Unmatched Singletons:     {stats['total_singletons']}")
    print(f"  • Total Multi-Source Entities:    {stats['total_linked_entities']}")
    print(f"  • Average Entity Cluster Size:    {stats['average_cluster_size']}")
    print(f"  • Pipeline Execution Runtime:     {stats['elapsed_seconds']}s")
    print("\nTo launch the interactive web platform, execute:")
    print("  streamlit run app.py\n")


if __name__ == "__main__":
    main()
