"""
Entity Clustering Module & Union-Find Graph Engine (Phases 4 & 8).
Maintains disjoint set connectivity for real-world entities with startup edge-replay recovery.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional
import pandas as pd


class UnionFind:
    """
    Disjoint Set Union (Union-Find) with path compression and union by rank.
    Serves as the single authoritative clustering structure for entity resolution.
    """

    def __init__(self, elements: Optional[List[str]] = None):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}
        if elements:
            for el in elements:
                self.add(el)

    def add(self, item: str):
        """Add a new standalone element to the disjoint set."""
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item: str) -> str:
        """Find root parent with path compression."""
        if item not in self.parent:
            self.add(item)
            return item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1: str, item2: str) -> bool:
        """
        Merge sets containing item1 and item2.
        Returns True if a new merge occurred, False if already in the same set.
        """
        root1 = self.find(item1)
        root2 = self.find(item2)

        if root1 != root2:
            if self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            elif self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1
            return True
        return False

    def get_clusters(self) -> Dict[str, List[str]]:
        """
        Group all elements by their canonical root entity.
        Returns: {root_id: [member_uuids]}
        """
        clusters = defaultdict(list)
        for item in self.parent:
            root = self.find(item)
            clusters[root].append(item)
        return dict(clusters)

    def rebuild_from_persisted_edges(self, all_record_uuids: List[str], edges: List[Tuple[str, str]]):
        """
        Reconstruct in-memory UnionFind state on application startup by replaying
        persisted canonical records and enrichment edges.
        """
        self.parent.clear()
        self.rank.clear()
        for rec_id in all_record_uuids:
            self.add(rec_id)
        for id_a, id_b in edges:
            self.union(id_a, id_b)
