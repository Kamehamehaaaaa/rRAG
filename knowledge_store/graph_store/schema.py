from pydantic import BaseModel
from typing import List, Dict

class Entity(BaseModel):
    type: str
    name: str
    description: str | None = None

class Relation(BaseModel):
    source: str
    target: str
    relation: str
    description: str | None = None

class ExtractedGraph(BaseModel):
    entities: List[Entity]
    relations: List[Relation]

# Graph store schema for storing extracted entities and relations
class StoredEntity(BaseModel):
    id: str
    name: str
    type: str
    source_chunks: List[str]

class StoredRelation(BaseModel):
    source: str
    target: str
    relation: str
    source_chunks: List[str]

class GraphData(BaseModel):
    entities: Dict[str, StoredEntity] = {}
    relations: Dict[str, StoredRelation] = {} 

