# Multi-Database Entity Resolution & Golden Master Platform

PRJ-07 is a domain-agnostic entity resolution and master data management platform. It turns heterogeneous CSV and SQL datasets into a trusted golden master: one row per resolved entity, consolidated attributes, and field-level lineage back to the source records.

---

## Core Capabilities

- **Arbitrary Ingestion**: Integrates any CSV or SQL dump without predefined column name assumptions.
- **AI Schema Mapping**: Gemini 2.5 Flash-Lite proposes canonical fields using column headers and 2–3 sample values; handles wide datasets (100+ cols) and flags ambiguous names.
- **Strict Disjoint Matching**: Inverted identifier index (Email, Phone, Aadhaar, Username, Member ID) + Union-Find clustering. Strict null/blank isolation guarantees zero false collapse.
- **Universal Field Preservation**: Every source attribute (canonical or unmapped passthrough) is preserved in the golden repository across all data types (Date, numeric, boolean, text).
- **Audit & Lineage**: Full field-level provenance records source dataset, source column, and resolution rule for every attribute.

## Technologies Used

Python, Streamlit, pandas, NumPy, DuckDB, scikit-learn, NetworkX, RapidFuzz, Plotly, and Gemini 2.5 Flash-Lite.

---

## Local Run

```bash
\.\.venv\Scripts\streamlit.exe run app.py
```

---

## Workflow Steps

1. **Overview**: Value proposition and 1-click interactive demo.
2. **Upload Datasets**: Upload 2+ CSV or SQL dump files.
3. **AI Schema Mapping**: Review and edit column mappings; ambiguous columns are flagged for review.
4. **Review & Process**: Run entity resolution and synthesize the master repository.
5. **Results & Master Dataset**: Review deduplication KPIs, inspect field lineage, and download golden master CSV.
6. **Search & Trace Lineage**: Query stored master entities and inspect progressive traversal trails.

---

## Database Architecture (DuckDB)

| Table | Description |
| :--- | :--- |
| `data_sources` | Metadata catalog of uploaded sources |
| `dataset_parts` | Multi-part and chunked file tracking |
| `canonical_records` | Dual storage of raw payload and normalized attributes |
| `identifier_index` | Inverted lookup index for exact identifier matching |
| `enrichment_edges` | Audit trail of all cross-record links |
| `master_entities` | Golden record repository with dynamic attributes |
| `entity_members` | Junction table linking master entities to source records |
| `field_provenance` | Field-level attribution, winning sources, and resolution rules |

---

