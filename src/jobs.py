"""
Background Job Registry, Checkpointing & Lifecycle Recovery (PRJ-07).
Provides thread-safe singleton background execution, atomic chunk checkpointing,
and application-startup crash recovery for long-running batch ingestion pipelines.
"""

import os
import sys
import time
import threading
from typing import Dict, List, Any, Optional, Callable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    JOB_STATUS_QUEUED,
    JOB_STATUS_RUNNING,
    JOB_STATUS_PAUSED_RECOVERABLE,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_FAILED_INTERRUPTED,
)
from src.database import DatabaseManager, get_db_manager
from src.clustering import UnionFind


class JobRegistry:
    """
    Singleton thread-safe background job runner and lifecycle tracker.
    Prevents duplicate background threads across Streamlit reruns and handles startup crash recovery.
    """

    _instance: Optional["JobRegistry"] = None
    _registry_lock = threading.RLock()
    _active_threads: Dict[str, threading.Thread] = {}
    _job_progress: Dict[str, Dict[str, Any]] = {}

    def __new__(cls):
        with cls._registry_lock:
            if cls._instance is None:
                cls._instance = super(JobRegistry, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.db = get_db_manager()
        self._initialized = True

    @classmethod
    def is_job_running(cls, job_id: str) -> bool:
        """Check if a live background thread is actively executing for job_id."""
        with cls._registry_lock:
            thread = cls._active_threads.get(job_id)
            if thread is not None and thread.is_alive():
                return True
            if job_id in cls._active_threads:
                del cls._active_threads[job_id]
            return False

    def create_job(
        self,
        job_id: str,
        source_id: Optional[str],
        job_type: str,
        total_records: int = 0,
    ) -> str:
        """Register a new job in DuckDB with status QUEUED."""
        self.db.execute_write(
            """
            INSERT OR REPLACE INTO job_runs (
                job_id, source_id, job_type, current_stage, status,
                total_records, processed_records, last_checkpoint_chunk, error_message, start_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [job_id, source_id, job_type, "QUEUED", JOB_STATUS_QUEUED, total_records, 0, 0, None]
        )
        return job_id

    def start_background_job(
        self,
        job_id: str,
        worker_func: Callable[..., Any],
        *args,
        **kwargs
    ) -> bool:
        """
        Spawn a worker thread for job_id if not already running.
        Guards against duplicate thread creation on Streamlit reruns.
        """
        with self._registry_lock:
            if self.is_job_running(job_id):
                return False  # Already running; do not spawn a duplicate thread

            # Update status to RUNNING in DuckDB
            self.db.execute_write(
                "UPDATE job_runs SET status = ?, current_stage = ? WHERE job_id = ?",
                [JOB_STATUS_RUNNING, "PROCESSING", job_id]
            )

            def thread_wrapper():
                try:
                    worker_func(*args, **kwargs)
                    self.db.execute_write(
                        "UPDATE job_runs SET status = ?, current_stage = ?, end_time = CURRENT_TIMESTAMP WHERE job_id = ?",
                        [JOB_STATUS_COMPLETED, "COMPLETED", job_id]
                    )
                except Exception as e:
                    print(f"[JOB WORKER ERROR] Job {job_id} failed: {e}")
                    self.db.execute_write(
                        "UPDATE job_runs SET status = ?, current_stage = ?, error_message = ?, end_time = CURRENT_TIMESTAMP WHERE job_id = ?",
                        [JOB_STATUS_FAILED, "FAILED", str(e), job_id]
                    )
                finally:
                    with self._registry_lock:
                        if job_id in self._active_threads:
                            del self._active_threads[job_id]

            worker_thread = threading.Thread(target=thread_wrapper, daemon=True, name=f"Worker-{job_id}")
            self._active_threads[job_id] = worker_thread
            worker_thread.start()
            return True

    def recover_on_startup(self, union_find: Optional[UnionFind] = None) -> Dict[str, Any]:
        """
        Executed on application startup:
        1. Inspects persisted jobs in RUNNING or QUEUED state and transitions them to PAUSED_RECOVERABLE or FAILED_INTERRUPTED.
        2. Rebuilds in-memory UnionFind state by replaying persisted enrichment_edges.
        """
        # Step 1: Recover Jobs
        interrupted_jobs = self.db.execute_query(
            "SELECT job_id, last_checkpoint_chunk, total_records FROM job_runs WHERE status IN (?, ?)",
            [JOB_STATUS_RUNNING, JOB_STATUS_QUEUED]
        )

        recovered_count = 0
        if not interrupted_jobs.empty:
            for _, row in interrupted_jobs.iterrows():
                j_id = row["job_id"]
                last_chk = row.get("last_checkpoint_chunk", 0)
                new_status = JOB_STATUS_PAUSED_RECOVERABLE if last_chk > 0 else JOB_STATUS_FAILED_INTERRUPTED
                err_msg = "Application restarted while job was in-flight. State checkpoint preserved for resumption." if last_chk > 0 else "Application restarted before initial checkpoint."

                self.db.execute_write(
                    "UPDATE job_runs SET status = ?, error_message = ? WHERE job_id = ?",
                    [new_status, err_msg, j_id]
                )
                recovered_count += 1

        # Step 2: Rebuild UnionFind state from persisted edges
        rebuilt_edges_count = 0
        if union_find is not None:
            all_recs = self.db.execute_query("SELECT record_uuid FROM canonical_records")
            all_uuids = all_recs["record_uuid"].tolist() if not all_recs.empty else []

            edges_df = self.db.execute_query("SELECT record_uuid_a, record_uuid_b FROM enrichment_edges ORDER BY step_order ASC")
            edges_list = [(row["record_uuid_a"], row["record_uuid_b"]) for _, row in edges_df.iterrows()] if not edges_df.empty else []

            union_find.rebuild_from_persisted_edges(all_uuids, edges_list)
            rebuilt_edges_count = len(edges_list)

        return {
            "interrupted_jobs_recovered": recovered_count,
            "edges_replayed_into_union_find": rebuilt_edges_count,
        }


def get_job_registry() -> JobRegistry:
    """Helper to access singleton JobRegistry."""
    return JobRegistry()
