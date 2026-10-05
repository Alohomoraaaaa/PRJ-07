# Nexus • Multi-Database Entity Resolution & Master Data Platform (PRJ-07)

A production-grade, stakeholder-facing Master Data Management (MDM) and Entity Resolution platform that ingests multiple heterogeneous databases, automatically proposes AI-assisted schema mappings, performs deterministic record linkage, and synthesizes an authoritative **Golden Master Dataset** backed by embedded columnar DuckDB storage with full field-level provenance and CSV export.

---

## 🌟 Executive Product Highlights

1. **Self-Service Multi-Source Ingestion**:
   - Upload 2 or more CSV datasets or SQL database dumps with arbitrary, disparate schemas (e.g. `email_id` vs `contact_email`, `full_name` vs `person_name`).
   - No predefined schemas or hardcoded source names required.

2. **AI-Assisted Schema Mapping Studio**:
   - Deterministic semantic detection via RapidFuzz token matching and synonym dictionaries.
   - Shows explainable confidence scores and plain-English matching reasoning for every column.
   - Stakeholder review workflow: approve, reassign, or unmap fields.
   - Cross-dataset compatibility validation: alerts users if datasets share no common identity linkers.

3. **Deterministic Record Linkage & Conflict Resolution**:
   - Uses Inverted Identifier Indices + Disjoint Set Union (`UnionFind`) as the single source of truth.
   - 5-tier explainable conflict resolution (unanimous consensus, majority vote, source authority priority, completeness length heuristic, deterministic sort).
   - Zero record loss: unmatched records are preserved as standalone singleton entities (`cluster_size = 1`).

4. **Progressive Entity Enrichment Explorer**:
   - Search by any mapped identifier (Email, Phone, Username, ID) on **any user-uploaded dataset**.
   - Chronologically replays recursive multi-hop discovery across disconnected sources with step cards and interactive network subgraphs.

5. **Stakeholder Explainability & Field-Level Lineage**:
   - Detailed inspector explaining **WHY** records were merged and **WHY** each final attribute value won.
   - Traceable link to original raw strings (`raw_fields_json`) and normalized values (`norm_fields_json`).

6. **Instant Master Dataset Export**:
   - 1-click **Download Unified Master Dataset (CSV)** for downstream BI, CRM, or data warehouse pipelines.

---

## 🚀 Quick Start Guide

### 1. Activate Environment & Install Dependencies
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run Automated Test Suite
```powershell
python -m unittest discover tests
```

### 3. Launch the Interactive Web Platform
```powershell
streamlit run app.py
```

### 4. Optional: Run CLI Demonstration
```powershell
python main.py --demo
```

---

## 🧭 Application User Journey

```
┌────────────────────────────────────────────────────────────────────────────┐
│                       Nexus Stakeholder Workflow                           │
├────────────────────────────────────────────────────────────────────────────┤
│ 🚀 Project Overview         • Executive summary of MDM & Entity Resolution │
│                             • "Start New Project" or "Load Demo Suite"     │
│                                                                            │
│ 📂 1. Upload Datasets       • Upload 2+ CSV / SQL files with any schema    │
│                             • Live preview of row/column counts            │
│                                                                            │
│ 🧠 2. AI Schema Mapping     • Review semantic mapping suggestions          │
│                             • Confidence scores, reasoning & user approval │
│                             • Cross-dataset compatibility validation       │
│                                                                            │
│ ✨ 3. Master Dataset        • Summary KPIs: Total Input, Master Entities,  │
│                             • Duplicates Resolved, Deduplication Rate      │
│                             • 📥 Download Golden Master Dataset (CSV)      │
│                                                                            │
│ 🔍 4. Progressive Search    • Search by Email, Phone, Username, or ID      │
│                             • Multi-hop discovery cards & network graph    │
│                                                                            │
│ 🔬 5. Explainability        • Attribute provenance & winning rules         │
│                             • Contributing member record raw vs norm diff  │
│                                                                            │
│ 📊 Advanced: ML Benchmark   • Supervised Random Forest FEBRL evaluation    │
│                             • Decision threshold slider, ROC-AUC, matrix   │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 🏛️ Architecture & Database Model (DuckDB)

| Table | Purpose |
| :--- | :--- |
| `data_sources` | Metadata catalog of all uploaded files and databases |
| `dataset_parts` | Multi-part and chunked file tracking |
| `canonical_records` | Dual storage: unaltered raw JSON + cleaned normalized fields |
| `identifier_index` | High-performance inverted index on Email, Phone, Username, IDs |
| `enrichment_edges` | Audit trail of all graph merges for progressive replay |
| `master_entities` | Final unified golden entities with completeness scores |
| `entity_members` | Normalized junction table linking entities to source records |
| `field_provenance` | Attribute-level lineage tracking winning rules and source origins |
| `job_runs` | Persistent execution states and resumability checkpoints |
