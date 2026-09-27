from .schema import GraphData
from pathlib import Path
import networkx as nx

from .schema import ExtractedGraph, StoredEntity, StoredRelation

# we work with cache generations and they have graphs in graph.json
def load_graph(gen_dir: Path) -> GraphData:
    graph_file = gen_dir / "graph.json"

    if graph_file.is_file():
        return GraphData.model_validate_json(graph_file.read_text(encoding="utf-8"))

    return GraphData()

def to_networkx(graph: GraphData) -> nx.DiGraph:
    g = nx.DiGraph()

    for id, entity in graph.entities.items():
        g.add_node(id, name=entity.name, type=entity.type, 
                   source_chunks=entity.source_chunks)

    for r in graph.relations.values():
        g.add_edge(r.source, r.target, relation=r.relation, 
                   source_chunks=r.source_chunks)

    return g

def save_graph(graph: GraphData, gen_dir: Path) -> None:
    (gen_dir / "graph.json").write_text(graph.model_dump_json(indent=2), encoding="utf-8")

import re

def normalize_entity_id(name: str) -> str:
    # can be substituted with embedding similarity matching 
    # or a resolution pass using transformers trained on 
    # real data.
    return re.sub(r"\s+", " ", name.strip().lower())

# pos is the chunks position in the data.
# for connecting chunk's location and its neighbors in the data.
def merge_extraction_into_graph(graph: GraphData, extracted_graph: ExtractedGraph, pos: int) -> GraphData:
    for entity in extracted_graph.entities:
        norm = normalize_entity_id(entity)
        if norm in graph.entities:
            existing = graph.entities[norm]
            if pos not in existing.source_chunks:
                existing.source_chunks.append(pos)
        else:
            graph.entities[norm] = StoredEntity(
                id=norm, name=entity.name, type=entity.type, source_chunks=[pos],
            )

    for relation in extracted_graph.relations:
        src = normalize_entity_id(relation.source)
        tgt = normalize_entity_id(relation.target)
        key = f"{src}|{relation.relation}|{tgt}"
        if key in graph.relations:
            existing = graph.relations[key]
            if pos not in existing.source_chunks:
                existing.source_chunks.append(pos)
        else:
            graph.relations[key] = StoredRelation(
                source=src, target=tgt, relation=relation.relation, source_chunks=[pos],
            )