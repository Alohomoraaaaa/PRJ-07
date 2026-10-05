"""
SQL Dump Importer & Schema Introspection Engine (Section 4 & 10).
Executes SQL database dumps inside an isolated SQLite environment (in-memory or file-backed
temporary database for large dumps) and introspects tables and columns without custom SQL parsing.
"""

import os
import sqlite3
import tempfile
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional


class SQLDumpImporter:
    """
    Safely executes uploaded .sql dumps in an isolated SQLite instance
    and extracts relational tables into DataFrames for schema inspection and ingestion.
    """

    @staticmethod
    def import_sql_dump(
        sql_file_path: str,
        max_in_memory_bytes: int = 10 * 1024 * 1024,  # 10 MB threshold
    ) -> Dict[str, pd.DataFrame]:
        """
        Execute SQL dump and return dictionary of {table_name: DataFrame}.
        Uses :memory: for small files and a file-backed temporary database for large files.
        """
        if not os.path.exists(sql_file_path):
            raise FileNotFoundError(f"SQL dump file not found: {sql_file_path}")

        file_size = os.path.getsize(sql_file_path)
        temp_db_path = None

        if file_size <= max_in_memory_bytes:
            conn = sqlite3.connect(":memory:")
        else:
            temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            temp_db_path = temp_db.name
            temp_db.close()
            conn = sqlite3.connect(temp_db_path)

        try:
            with open(sql_file_path, "r", encoding="utf-8", errors="replace") as f:
                sql_script = f.read()

            cursor = conn.cursor()
            cursor.executescript(sql_script)
            conn.commit()

            # Introspect created tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            tables = [row[0] for row in cursor.fetchall()]

            extracted_tables: Dict[str, pd.DataFrame] = {}
            for tbl in tables:
                df = pd.read_sql_query(f"SELECT * FROM {tbl}", conn)
                extracted_tables[tbl] = df

            return extracted_tables

        finally:
            conn.close()
            if temp_db_path and os.path.exists(temp_db_path):
                try:
                    os.unlink(temp_db_path)
                except Exception:
                    pass


if __name__ == "__main__":
    from src.demo_datasets import generate_demonstration_suite
    files = generate_demonstration_suite()
    sql_path = files["sample_sql_dump"]
    imported = SQLDumpImporter.import_sql_dump(sql_path)
    print("Imported SQL Dump Tables:")
    for tbl_name, df_data in imported.items():
        print(f"  • Table '{tbl_name}': {len(df_data)} rows, Columns: {list(df_data.columns)}")
