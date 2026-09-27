from .schema import GraphData
import networkx as nx

from pathlib import Path
from typing import List

from ..manifest import _read_generation
from ..paths import current_gen_dir

from .utils import load_graph, to_networkx


class GraphCache:
    def __init__(self, index_root: Path, manifest_path: Path):
        self.index_root = index_root
        self.manifest_path = manifest_path
        self._gen_cached: int | None = None
        self.graph: GraphData | None = None
        self._nx: nx.DiGraph | None = None

    def _ensure_loaded(self) -> None:
        cur_gen = _read_generation(self.manifest_path) or 0
        cached_gen = self._gen_cached or 0
        if self.graph is None or cur_gen > cached_gen:
            gen_dir = current_gen_dir(self.index_root, self.manifest_path)
            self.graph = load_graph(gen_dir)
            self._nx = to_networkx(self.graph)
            self._gen_cached = cur_gen

    def expand(self, seed_positions: List[int], hops: int = 2)-> List[int]:
        self._ensure_loaded()
        if self._nx is None or self._nx.number_of_nodes() == 0:
            return []

        seed_nodes = {
            node for node, data in self._nx.nodes(data=True)
            if any(pos in data.get("source_chunks", []) for pos in seed_positions)
        }

        if not seed_nodes:
            return []

        result_positions: List[int] = []
        seen = set(seed_positions)
        frontier = seed_nodes
        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                next_frontier |= set(self._nx.successors(node)) | set(self._nx.predecessors(node))
            next_frontier -= seed_nodes
            for node in next_frontier:
                for pos in self._nx.nodes[node].get("source_chunks", []):
                    if pos not in seen:
                        result_positions.append(pos)
                        seen.add(pos)
            frontier = next_frontier
            if not frontier:
                break
        return result_positions