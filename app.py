"""
Multi-Database Entity Resolution & Unified Master Data Platform (PRJ-07).
Features:
- Gemini 2.5 Flash-Lite AI Schema Mapping with Deterministic Fallback
- Strict Null/Blank Isolation & Zero Collapse Guarantee
- Universal Passthrough of Dataset-Specific Columns (followers, rating, bio, etc.)
- Explainable Field Provenance & Conflict Resolution
- High-Performance In-Memory Session Caching
"""

import os
import sys
import json
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config import DEFAULT_DB_PATH, CANONICAL_FIELDS
from src.database import get_db_manager
from src.ai_mapper import GeminiSchemaMapper
from src.matcher import HierarchicalMatcher
from src.master_records import get_unified_master_dataframe
from src.clustering import UnionFind
from src.pipeline import run_dynamic_user_pipeline
from src.demo_datasets import generate_demonstration_suite
from src.sql_importer import SQLDumpImporter

# ==============================================================================
# 1. PAGE SETUP & CORPORATE STYLING
# ==============================================================================
st.set_page_config(
    page_title="Multi-Database Entity Resolution Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .stApp {
        background-color: #0B0F19;
        color: #F8FAFC;
    }
    
    .page-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    
    .page-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-bottom: 1.2rem;
        line-height: 1.5;
    }
    
    .card-panel {
        background: #131B2E;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }
    
    .card-highlight {
        background: linear-gradient(135deg, #131B2E 0%, #1E1B4B 100%);
        border: 1px solid #3B82F6;
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }
    
    .kpi-card {
        background: #131B2E;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 14px 18px;
        text-align: center;
    }
    
    .kpi-label {
        font-size: 0.75rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 4px;
    }
    
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.1;
    }
    
    .kpi-sub {
        font-size: 0.78rem;
        color: #34D399;
        font-weight: 600;
        margin-top: 4px;
    }
    
    .badge-tag {
        background: #1E293B;
        color: #E2E8F0;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid #334155;
    }
    
    .nav-stepper {
        display: flex;
        gap: 8px;
        background: #131B2E;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 8px 14px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. SESSION STATE CACHING
# ==============================================================================
db = get_db_manager(DEFAULT_DB_PATH)
ai_mapper = GeminiSchemaMapper()

if "current_step" not in st.session_state:
    st.session_state.current_step = 1

if "uploaded_datasets" not in st.session_state:
    st.session_state.uploaded_datasets = {}  # {name: {"df": df, "mapping": dict, "proposals": dict}}

if "schema_cache" not in st.session_state:
    st.session_state.schema_cache = {}  # {schema_hash: proposals_dict}

if "master_results" not in st.session_state:
    st.session_state.master_results = None

if "stats_cached" not in st.session_state:
    st.session_state.stats_cached = None


def query_db(sql: str, params: list = None) -> pd.DataFrame:
    return db.execute_query(sql, params)


# ==============================================================================
# 3. SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.markdown("### Workflow Navigation")

nav_steps = [
    "1. Overview",
    "2. Upload Datasets",
    "3. AI Schema Mapping",
    "4. Review & Process",
    "5. Results & Master Repository",
    "6. Search & Lineage Audit",
]

selected_nav = st.sidebar.radio(
    "Workflow Steps",
    nav_steps,
    index=st.session_state.current_step - 1,
)
selected_step_num = int(selected_nav.split(".")[0])
if selected_step_num != st.session_state.current_step:
    st.session_state.current_step = selected_step_num
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("PRJ-07 • Multi-Database Entity Resolution")
st.sidebar.caption("AI-Assisted Master Data Management")


# ==============================================================================
# STEP 1: OVERVIEW
# ==============================================================================
if st.session_state.current_step == 1:
    st.markdown('<div class="page-title">Multi-Database Entity Resolution & Unified Repository</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Integrate disparate databases, intelligently map column schemas with Gemini AI, match records across sources, and synthesize a clean Master Dataset.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="card-panel">
            <h4 style="color: #60A5FA; margin-top: 0;">1. Ingest & AI Map</h4>
            <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.5;">
                Upload multiple CSVs or SQL dumps with arbitrary schemas. Gemini 2.5 Flash-Lite automatically maps raw headers to canonical concepts.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card-panel">
            <h4 style="color: #818CF8; margin-top: 0;">2. Match & Disjoint Cluster</h4>
            <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.5;">
                Hierarchical inverted indexing and Union-Find graph clustering link verified identities (Email, Phone, Aadhaar, Username) without false collapse.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="card-panel">
            <h4 style="color: #34D399; margin-top: 0;">3. Master Dataset & Lineage</h4>
            <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.5;">
                Synthesizes a golden master repository preserving all source attributes, field provenance, and enables recursive progressive enrichment.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    b_col1, b_col2 = st.columns([1.5, 4])
    with b_col1:
        if st.button("Start Integration →", type="primary", use_container_width=True):
            st.session_state.current_step = 2
            st.rerun()
    with b_col2:
        if st.button("⚡ Quick Demo (Load Customer Datasets)", use_container_width=False):
            demo_files = generate_demonstration_suite()
            st.session_state.uploaded_datasets = {
                "customer_a.csv": {"df": pd.read_csv(demo_files["customer_a"]), "mapping": {}},
                "customer_b.csv": {"df": pd.read_csv(demo_files["customer_b"]), "mapping": {}},
            }
            st.session_state.current_step = 2
            st.rerun()


# ==============================================================================
# STEP 2: UPLOAD DATASETS
# ==============================================================================
elif st.session_state.current_step == 2:
    col_t1, col_t2 = st.columns([4, 1.2])
    with col_t1:
        st.markdown('<div class="page-title">Step 2: Upload Datasets</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Upload multiple CSV files or SQL database dumps to integrate.</div>', unsafe_allow_html=True)
    with col_t2:
        if st.button("Proceed to AI Mapping →", type="primary", use_container_width=True):
            if not st.session_state.uploaded_datasets:
                st.error("Please upload or load at least one dataset.")
            else:
                st.session_state.current_step = 3
                st.rerun()

    uploaded_files = st.file_uploader(
        "Upload Datasets (CSV or SQL)",
        type=["csv", "sql"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for f in uploaded_files:
            if f.name not in st.session_state.uploaded_datasets:
                if f.name.endswith(".csv"):
                    df = pd.read_csv(f)
                    st.session_state.uploaded_datasets[f.name] = {"df": df, "mapping": {}}
                elif f.name.endswith(".sql"):
                    temp_sql = os.path.join("data", f"temp_{f.name}")
                    with open(temp_sql, "wb") as out:
                        out.write(f.getbuffer())
                    imported = SQLDumpImporter.import_sql_dump(temp_sql)
                    for tbl, tbl_df in imported.items():
                        st.session_state.uploaded_datasets[f"{f.name} ({tbl})"] = {"df": tbl_df, "mapping": {}}
                    if os.path.exists(temp_sql):
                        os.remove(temp_sql)

    if not st.session_state.uploaded_datasets:
        st.info("No datasets loaded yet. Upload files above or load the demo datasets:")
        if st.button("Load Demo Datasets (customer_a.csv & customer_b.csv)"):
            demo_files = generate_demonstration_suite()
            st.session_state.uploaded_datasets = {
                "customer_a.csv": {"df": pd.read_csv(demo_files["customer_a"]), "mapping": {}},
                "customer_b.csv": {"df": pd.read_csv(demo_files["customer_b"]), "mapping": {}},
            }
            st.rerun()
    else:
        st.markdown("### Loaded Datasets Preview")
        for fname, dinfo in list(st.session_state.uploaded_datasets.items()):
            df_cur = dinfo["df"]
            with st.expander(f"📄 **{fname}** — {len(df_cur):,} rows • {len(df_cur.columns)} columns", expanded=True):
                c_head, c_del = st.columns([5, 1])
                with c_head:
                    st.dataframe(df_cur.head(3), use_container_width=True)
                with c_del:
                    if st.button(f"Remove", key=f"del_{fname}"):
                        del st.session_state.uploaded_datasets[fname]
                        st.rerun()

    st.markdown("---")
    b1, b2 = st.columns([1, 4])
    with b1:
        if st.button("← Back to Overview"):
            st.session_state.current_step = 1
            st.rerun()
    with b2:
        if st.button("Proceed to AI Mapping →", type="primary", key="btn_next_step2"):
            if not st.session_state.uploaded_datasets:
                st.error("Please upload or load at least one dataset.")
            else:
                st.session_state.current_step = 3
                st.rerun()


# ==============================================================================
# STEP 3: AI SCHEMA MAPPING
# ==============================================================================
elif st.session_state.current_step == 3:
    col_t1, col_t2 = st.columns([4, 1.2])
    with col_t1:
        st.markdown('<div class="page-title">Step 3: AI Schema Mapping</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Gemini 2.5 Flash-Lite analyzes column headers and sample values to propose canonical mappings. Review and edit as needed.</div>', unsafe_allow_html=True)
    with col_t2:
        if st.button("Review & Process →", type="primary", use_container_width=True):
            st.session_state.current_step = 4
            st.rerun()

    if not st.session_state.uploaded_datasets:
        st.warning("No datasets uploaded. Please return to Step 2.")
        if st.button("← Go to Upload"):
            st.session_state.current_step = 2
            st.rerun()
        st.stop()

    canonical_options = ["unmapped"] + list(CANONICAL_FIELDS.keys()) + ["followers", "platform", "bio"]

    for fname, dinfo in st.session_state.uploaded_datasets.items():
        df_cur = dinfo["df"]
        schema_key = f"{fname}_{'_'.join(df_cur.columns.tolist())}"

        # Run AI mapping ONCE per schema and cache
        if schema_key not in st.session_state.schema_cache:
            with st.spinner(f"Analyzing schema for {fname} with Gemini AI..."):
                proposals = ai_mapper.suggest_mappings(df_cur, dataset_name=fname)
                st.session_state.schema_cache[schema_key] = proposals
                dinfo["mapping"] = {c: p["suggested_canonical"] for c, p in proposals.items()}
                dinfo["proposals"] = proposals
        else:
            proposals = st.session_state.schema_cache[schema_key]
            if not dinfo.get("mapping"):
                dinfo["mapping"] = {c: p["suggested_canonical"] for c, p in proposals.items()}
            dinfo["proposals"] = proposals

        st.markdown(f"#### 📄 **{fname}** ({len(df_cur):,} rows)")

        # Compact Mapping Review Table
        table_rows = []
        for col_name in df_cur.columns.tolist():
            prop = proposals.get(col_name, {})
            current_target = dinfo["mapping"].get(col_name, "unmapped")
            sample_vals = ", ".join([str(v) for v in df_cur[col_name].dropna().astype(str).tolist()[:3] if str(v).strip()])

            c_c1, c_c2, c_c3, c_c4 = st.columns([2, 3, 2, 3])
            with c_c1:
                st.markdown(f"**`{col_name}`**")
            with c_c2:
                st.caption(f"Samples: {sample_vals[:40]}..." if len(sample_vals) > 40 else f"Samples: {sample_vals}")
            with c_c3:
                target_idx = canonical_options.index(current_target) if current_target in canonical_options else 0
                new_sel = st.selectbox(
                    f"Target {col_name}",
                    options=canonical_options,
                    index=target_idx,
                    key=f"map_{fname}_{col_name}",
                    label_visibility="collapsed",
                )
                dinfo["mapping"][col_name] = new_sel
            with c_c4:
                conf = prop.get("confidence", 1.0) * 100
                reason = prop.get("reason", "Exact Match")
                st.markdown(f"<span class='badge-tag'>{conf:.0f}% Confidence</span> <span style='color: #94A3B8; font-size: 0.8rem;'>{reason}</span>", unsafe_allow_html=True)

        st.markdown("<hr style='margin: 14px 0; border-color: #1E293B;'>", unsafe_allow_html=True)

    b1, b2 = st.columns([1, 4])
    with b1:
        if st.button("← Back to Upload"):
            st.session_state.current_step = 2
            st.rerun()
    with b2:
        if st.button("Review & Process →", type="primary", key="btn_next_step3"):
            st.session_state.current_step = 4
            st.rerun()


# ==============================================================================
# STEP 4: REVIEW & PROCESS
# ==============================================================================
elif st.session_state.current_step == 4:
    st.markdown('<div class="page-title">Step 4: Review & Process</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Confirm schema mappings and execute entity resolution to synthesize the Master Dataset.</div>', unsafe_allow_html=True)

    if not st.session_state.uploaded_datasets:
        st.warning("No datasets uploaded. Return to Step 2.")
        if st.button("← Go to Upload"):
            st.session_state.current_step = 2
            st.rerun()
        st.stop()

    # Pre-execution Manifest
    manifest = []
    total_input_rows = 0
    for fname, dinfo in st.session_state.uploaded_datasets.items():
        nr = len(dinfo["df"])
        total_input_rows += nr
        mapped_attrs = sum(1 for v in dinfo.get("mapping", {}).values() if v != "unmapped")
        manifest.append({
            "Dataset": fname,
            "Rows": f"{nr:,}",
            "Mapped Columns": mapped_attrs,
            "Total Columns": len(dinfo["df"].columns),
            "Status": "Ready",
        })

    st.dataframe(pd.DataFrame(manifest), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Big Process Button
    if st.button("🚀 Run Entity Resolution & Build Master Dataset", type="primary", use_container_width=True):
        with st.spinner("Processing records through entity resolution engine..."):
            specs = []
            for src_idx, (fname, dinfo) in enumerate(st.session_state.uploaded_datasets.items(), start=1):
                clean_id = f"src_{src_idx}_{fname.replace(' ', '_').replace('.', '_')}"
                specs.append({
                    "source_id": clean_id,
                    "source_name": fname,
                    "dataframe": dinfo["df"],
                    "mapping": dinfo["mapping"],
                })

            stats = run_dynamic_user_pipeline(datasets_spec=specs, reset_db=True)
            st.session_state.master_results = stats
            st.session_state.stats_cached = stats
            st.session_state.current_step = 5
            st.rerun()

    st.markdown("---")
    if st.button("← Back to AI Mapping"):
        st.session_state.current_step = 3
        st.rerun()


# ==============================================================================
# STEP 5: RESULTS & MASTER REPOSITORY
# ==============================================================================
elif st.session_state.current_step == 5:
    col_t1, col_t2 = st.columns([3.5, 1.5])
    with col_t1:
        st.markdown('<div class="page-title">Step 5: Results & Master Dataset</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Unified Master Dataset synthesized across all sources with complete field preservation and zero false collapse.</div>', unsafe_allow_html=True)

    stats = db.get_system_statistics()
    if stats["total_records"] == 0:
        st.info("No records in master repository. Please execute Step 4.")
        if st.button("← Go to Step 4"):
            st.session_state.current_step = 4
            st.rerun()
        st.stop()

    total_in = stats["total_records"]
    total_out = stats["total_entities"]
    dups_res = max(0, total_in - total_out)
    merged_cnt = stats["total_linked_entities"]
    singletons = stats["total_singletons"]

    # Stakeholder KPI Summary Banner
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Input Rows</div>
            <div class="kpi-value">{total_in:,}</div>
            <div class="kpi-sub">Total Uploaded</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Matched Rows</div>
            <div class="kpi-value">{total_in - singletons:,}</div>
            <div class="kpi-sub">Cross-Linked</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">New Entities</div>
            <div class="kpi-value">{singletons:,}</div>
            <div class="kpi-sub">Unique / Singletons</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Duplicates Resolved</div>
            <div class="kpi-value">{dups_res:,}</div>
            <div class="kpi-sub">Deduplicated</div>
        </div>
        """, unsafe_allow_html=True)
    with k5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Final Master Rows</div>
            <div class="kpi-value">{total_out:,}</div>
            <div class="kpi-sub">Golden Entities</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Master Table Retrieval
    df_master = get_unified_master_dataframe(db)

    # Prominent Download CSV Button
    csv_data = df_master.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 DOWNLOAD MASTER DATASET (CSV)",
        data=csv_data,
        file_name="unified_master_dataset.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Master Entity Repository")
    st.dataframe(df_master, use_container_width=True, hide_index=True)

    # Explainability Section
    st.markdown("---")
    st.markdown("### Decision Explainability & Lineage")

    if not df_master.empty:
        def format_entity_label(eid: str) -> str:
            match = df_master[df_master["entity_id"] == eid]
            if match.empty:
                return str(eid)
            row = match.iloc[0]
            name = row.get("name") or row.get("full_name") or "Entity"
            size = row.get("cluster_size", 1)
            return f"{eid} — {name} ({size} Sources)"

        selected_ent_id = st.selectbox(
            "Select an Entity to Inspect Merge Rationale:",
            options=df_master["entity_id"].tolist(),
            format_func=format_entity_label,
        )

        if selected_ent_id:
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                st.markdown("##### Why Were These Records Merged?")
                edges_for_ent = query_db("""
                    SELECT identifier_type, identifier_value, source_id_a, source_id_b, match_confidence
                    FROM enrichment_edges
                    WHERE record_uuid_a IN (SELECT record_uuid FROM entity_members WHERE entity_id = ?)
                       OR record_uuid_b IN (SELECT record_uuid FROM entity_members WHERE entity_id = ?)
                """, [selected_ent_id, selected_ent_id])

                if edges_for_ent.empty:
                    st.markdown("<p style='color: #94A3B8;'>• Unique record with no matching identifiers in other datasets. Preserved as an independent master entity with zero data loss.</p>", unsafe_allow_html=True)
                else:
                    for _, erow in edges_for_ent.drop_duplicates(subset=['identifier_type', 'identifier_value']).iterrows():
                        conf_str = f"{erow['match_confidence'] * 100:.0f}%"
                        st.markdown(f"""
                        <div style="margin-bottom: 8px; background: #131B2E; padding: 10px 14px; border-radius: 6px; border-left: 3px solid #10B981;">
                            <span style="color: #34D399; font-weight: 700;">✓ Merged on {erow['identifier_type'].upper()}:</span> <code>{erow['identifier_value']}</code><br>
                            <span style="color: #94A3B8; font-size: 0.82rem;">Linked records across <b>{erow['source_id_a']}</b> and <b>{erow['source_id_b']}</b> ({conf_str} confidence)</span>
                        </div>
                        """, unsafe_allow_html=True)

            with col_e2:
                st.markdown("##### Selected Attribute Lineage")
                prov_df = query_db("SELECT canonical_field, resolved_value, source_id, resolution_rule FROM field_provenance WHERE entity_id = ?", [selected_ent_id])
                for _, prow in prov_df.iterrows():
                    rule_desc = "Unanimous agreement" if prow['resolution_rule'] in ("SINGLE_NON_NULL", "UNANIMOUS_CONSENSUS") else prow['resolution_rule'].replace("_", " ").title()
                    st.markdown(f"""
                    <div style="margin-bottom: 6px; color: #E2E8F0; font-size: 0.88rem;">
                        <b>{prow['canonical_field'].title()}</b>: <span style="color: #60A5FA;">{prow['resolved_value']}</span> <span style="color: #94A3B8; font-size: 0.8rem;">(from <code>{prow['source_id']}</code> via {rule_desc})</span>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("---")
    b1, b2 = st.columns([1, 4])
    with b1:
        if st.button("Start New Integration"):
            st.session_state.current_step = 2
            st.session_state.uploaded_datasets = {}
            st.session_state.master_results = None
            st.rerun()
    with b2:
        if st.button("Search Entities & Recursive Enrichment →", type="primary"):
            st.session_state.current_step = 6
            st.rerun()


# ==============================================================================
# STEP 6: SEARCH & LINEAGE AUDIT
# ==============================================================================
elif st.session_state.current_step == 6:
    st.markdown('<div class="page-title">Step 6: Search & Recursive Entity Enrichment</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Search by any single identifier (Email, Phone, Username, or Aadhaar) to trace recursive multi-hop linkages across disparate databases.</div>', unsafe_allow_html=True)

    recs_df = query_db("SELECT * FROM canonical_records")
    edges_df = query_db("SELECT * FROM enrichment_edges")

    if recs_df.empty:
        st.info("No records loaded yet. Please complete Steps 2-4 first.")
    else:
        # Discover available identifier types from currently active canonical records
        available_types = set()
        for _, r in recs_df.iterrows():
            try:
                nd = json.loads(r["norm_fields_json"]) if isinstance(r.get("norm_fields_json"), str) else (r.get("norm_fields_json") or {})
            except Exception:
                nd = {}
            for k, v in nd.items():
                if v and str(v).strip():
                    available_types.add(k)

        preferred_order = ["email", "phone", "aadhaar", "username", "member_id", "name"]
        search_options = [t for t in preferred_order if t in available_types] + sorted([t for t in available_types if t not in preferred_order])
        if not search_options:
            search_options = ["email", "phone", "aadhaar", "username", "member_id"]

        col_s1, col_s2, col_s3 = st.columns([1.5, 3, 1])
        with col_s1:
            search_type = st.selectbox("Identifier Type", search_options)

        # Dynamic sample value from current repository
        dynamic_sample = ""
        for _, r in recs_df.iterrows():
            try:
                nd = json.loads(r["norm_fields_json"]) if isinstance(r.get("norm_fields_json"), str) else (r.get("norm_fields_json") or {})
            except Exception:
                nd = {}
            if nd.get(search_type) and str(nd[search_type]).strip():
                dynamic_sample = str(nd[search_type]).strip()
                break

        with col_s2:
            search_query = st.text_input(
                f"Enter {search_type.replace('_', ' ').title()} Value",
                value=dynamic_sample,
                placeholder=f"e.g. {dynamic_sample or 'Enter identifier...'}"
            )
        with col_s3:
            st.markdown("<br>", unsafe_allow_html=True)
            search_clicked = st.button("Search Entity", type="primary", use_container_width=True)

        if search_query.strip():
            matcher = HierarchicalMatcher()
            replay = matcher.replay_progressive_enrichment(search_type, search_query.strip(), edges_df, recs_df)

            if not replay["found"]:
                st.warning(replay["message"])
            else:
                cluster_uuids = replay["cluster_record_uuids"]
                ent_row = query_db("""
                    SELECT m.* FROM master_entities m
                    JOIN entity_members em ON m.entity_id = em.entity_id
                    WHERE em.record_uuid IN ({})
                    LIMIT 1
                """.format(", ".join(["?"] * len(cluster_uuids))), cluster_uuids)

                if not ent_row.empty:
                    ent = ent_row.iloc[0]
                    raw_d = json.loads(ent['raw_attributes_json']) if ent.get('raw_attributes_json') else {}
                    
                    # Construct dynamic attributes display
                    attr_pills = []
                    for ak, av in raw_d.items():
                        if av and str(av).strip() and ak not in ("name", "full_name"):
                            attr_pills.append(f"<b>{ak.replace('_', ' ').title()}:</b> {av}")
                    attrs_html = " &nbsp;|&nbsp; ".join(attr_pills) if attr_pills else "No additional attributes"
                    ent_title = raw_d.get("name") or raw_d.get("full_name") or ent.get("golden_name") or f"Entity {ent['entity_id']}"

                    st.markdown(f"""
                    <div class="card-highlight">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                            <div>
                                <span class="badge-tag" style="background: #10B981; color: white; border: none;">UNIFIED MASTER ENTITY #{ent['entity_id']}</span>
                                <span class="badge-tag" style="margin-left: 8px;">{ent['cluster_size']} Contributing Sources</span>
                                <h2 style="color: white; margin: 8px 0 6px 0; font-size: 1.7rem;">{ent_title}</h2>
                                <p style="color: #94A3B8; margin: 0; font-size: 0.92rem; line-height: 1.6;">
                                    {attrs_html}
                                </p>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Completeness</div>
                                <div style="font-size: 1.9rem; font-weight: 800; color: #34D399;">{ent['entity_quality_score'] * 100:.0f}%</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.subheader(f"Progressive Traversal Trail ({len(replay['steps'])} Discovery Steps)")
                for s in replay["steps"]:
                    st.markdown(f"""
                    <div class="card-panel" style="border-left: 4px solid #3B82F6; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between;">
                            <b>Step {s['step_number']}: {s['action']}</b>
                            <span class="badge-tag" style="background: #1E293B; color: #60A5FA;">{s['source_dataset']}</span>
                        </div>
                        <p style="margin: 4px 0 0 0; color: #E2E8F0; font-size: 0.92rem;">{s['details']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with st.expander("Contributing Source Rows (Payloads)"):
                    matched_records = query_db("""
                        SELECT cr.source_id, cr.source_record_id, cr.norm_fields_json, cr.raw_fields_json
                        FROM canonical_records cr
                        WHERE cr.record_uuid IN ({})
                    """.format(", ".join(["?"] * len(cluster_uuids))), cluster_uuids)
                    st.dataframe(matched_records, use_container_width=True, hide_index=True)

    st.markdown("---")
    if st.button("← Back to Results & Master Dataset"):
        st.session_state.current_step = 5
        st.rerun()
