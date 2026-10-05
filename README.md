# Nexus • Multi-Database Entity Resolution & Golden Master Platform (PRJ-07)

A production-grade, domain-agnostic Master Data Management (MDM) and Entity Resolution platform. Ingests heterogeneous datasets across arbitrary schemas, performs AI-assisted schema mapping using Gemini 2.5 Flash-Lite with deterministic fallback, executes high-precision record linkage via inverted identifier indexing and Union-Find graph clustering, and synthesizes an authoritative **Golden Master Dataset** (1 Row = 1 Entity, 1 Column = 1 Consolidated Semantic Attribute) backed by embedded columnar DuckDB storage with complete field-level provenance and CSV export.

---

## 🌟 Core System Capabilities

1. **Domain-Agnostic Multi-Source Ingestion**:
   - Ingest 2 or more CSV datasets or SQL database dumps with arbitrary, disparate structures (e.g., Customer Demographics, Social Media, Healthcare, HR/Employee).
   - Zero hardcoded schemas or column name assumptions.

2. **AI-Assisted Schema Mapping Studio**:
   - Analyzes column headers and 2–3 sample values per column using **Gemini 2.5 Flash-Lite** (`GEMINI_API_KEY`).
   - Deterministic rule-based taxonomy fallback for offline or air-gapped environments.
   - Interactive stakeholder review: approve, reassign, or customize canonical mappings.

3. **Deterministic Record Linkage & Conflict Resolution**:
   - Strict non-null inverted index matching on high-confidence identifiers (Email, Phone, Aadhaar, Username, Member ID) + composite rules.
   - Disjoint-Set Union (`UnionFind`) clustering with zero false collapse on missing/blank values.
   - 6-tier deterministic conflict resolution:
     1. Non-Null Filter
     2. Unanimous Consensus
     3. Majority Voting
     4. Source Authority Priority
     5. Completeness / Length Heuristic
     6. Deterministic Lexicographical Fallback.

4. **Dynamic Golden Master Dataset Synthesis**:
   - **1 Row = 1 Resolved Entity**: Cross-database linkages deduplicated, singletons preserved with zero data loss.
   - **1 Column = 1 Consolidated Semantic Attribute**: Multiple source column variations (e.g. `bio_text` and `user_bio` $\rightarrow$ `bio`) merge into one golden attribute column without raw column duplication.
   - 1-click **Download Master Dataset (CSV)**.

5. **Entity Search & Lineage Audit (Step 6)**:
   - Search the active master repository by any populated identifier (Email, Phone, Aadhaar, Username, Member ID).
   - Replays multi-hop graph discovery trails and displays field-level provenance showing which source dataset contributed each attribute and which resolution rule was applied.

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the project root:
```env
GEMINI_API_KEY="your-gemini-api-key-here"
```
*(Note: If `GEMINI_API_KEY` is not provided, the platform automatically utilizes its deterministic rule-based mapping engine without failing).*

### 3. Run Automated Test Suite
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 4. Launch the Interactive Web Application
```bash
streamlit run app.py
```

### 5. Run CLI Demonstration
```bash
python main.py --demo
```

---

## 🧭 Step-by-Step User Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Nexus Platform Flow                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ 🚀 Step 1: Overview          • Executive MDM summary & architecture guide   │
│                              • "Start Ingestion" or "Load Demo Suite"       │
│                                                                             │
│ 📂 Step 2: Ingest Datasets   • Upload 2+ CSV datasets with any schema       │
│                              • View row/column counts and preview tables    │
│                                                                             │
│ 🧠 Step 3: AI Schema Review  • Review Gemini AI canonical proposals         │
│                              • Confidence scores, reasoning & user dropdowns│
│                                                                             │
│ ⚡ Step 4: Run Resolution    • Inverted index matching & Union-Find cluster │
│                              • 6-tier deterministic conflict resolution     │
│                                                                             │
│ ✨ Step 5: Master Results    • Summary KPIs & deduplication metrics         │
│                              • 📥 DOWNLOAD MASTER DATASET (CSV)             │
│                              • Decision explainability & source attribution │
│                                                                             │
│ 🔍 Step 6: Search & Lineage  • Search active repository by any identifier   │
│                              • Multi-hop graph trail & field-level audit    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🏛️ Relational Database Schema (DuckDB)

| Table | Purpose |
| :--- | :--- |
| `data_sources` | Metadata catalog of uploaded datasets and file formats |
| `dataset_parts` | Multi-part and chunked file tracking |
| `canonical_records` | Dual storage: original raw row (`raw_fields_json`) + clean normalized fields (`norm_fields_json`) |
| `identifier_index` | High-performance inverted index on Email, Phone, Aadhaar, Username, Member ID |
| `enrichment_edges` | Audit trail of all graph merges and match confidence scores |
| `master_entities` | Final unified golden entities with dynamic attributes (`raw_attributes_json`) and quality score |
| `entity_members` | Normalized junction table linking golden entities to contributing source records |
| `field_provenance` | Attribute-level lineage tracking winning values, source datasets, source columns, and resolution rules |

---

## ☁️ Deployment Instructions

### Streamlit Community Cloud (Recommended)
1. Push repository to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), click **New App**.
3. Select your repository, branch (`main`), and set **Main file path** to `app.py`.
4. In **Advanced Settings $\rightarrow$ Secrets**, add your API key:
   ```toml
   GEMINI_API_KEY = "your-gemini-api-key-here"
   ```
5. Click **Deploy**.

### Docker / Cloud Server (Render / Railway / VM)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
Set environment variable `GEMINI_API_KEY` in your cloud platform dashboard settings.
