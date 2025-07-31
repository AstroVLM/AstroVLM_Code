from dataclasses import dataclass, field
from typing import TypedDict, Union, Literal, Generic, TypeVar, Optional

import numpy as np

from .utils import EmbeddingFunc

TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int},
)

T = TypeVar("T")


@dataclass
class QueryParam:
    only_need_context: bool = False
    only_need_prompt: bool = False
    response_type: str = "Multiple Paragraphs"
    stream: bool = False
    top_k: int =100
    max_token_for_text_unit: int = 4000
    max_token_for_global_context: int = 3000
    max_token_for_local_context: int = 5000
    min_similarity_score: Optional[float] = None 


@dataclass
class StorageNameSpace:
    namespace: str
    global_config: dict

    async def index_done_callback(self):
       
        pass

    async def query_done_callback(self):
        
        pass


@dataclass
class BaseVectorStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)

    async def query(self, query_text: str, top_k: int, min_similarity: Optional[float] = None) -> list[dict]: # <--- MODIFIED SIGNATURE
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):

        raise NotImplementedError

    async def save(self):
        raise NotImplementedError

@dataclass
class BaseGraphStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc = None

    async def has_node(self, node_id: str) -> bool:
        raise NotImplementedError

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        raise NotImplementedError

    async def node_degree(self, node_id: str) -> int:
        raise NotImplementedError

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        raise NotImplementedError
    
    async def get_pagerank(self,node_id:str) -> float:
        raise NotImplementedError

    async def get_node(self, node_id: str) -> Union[dict, None]:
        raise NotImplementedError

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        raise NotImplementedError

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        raise NotImplementedError
    
    async def get_node_in_edges(
        self,source_node_id:str
    ) -> Union[list[tuple[str,str]],None]:
        raise NotImplementedError
    async def get_node_out_edges(
        self,source_node_id:str
    ) -> Union[list[tuple[str,str]],None]:
        raise NotImplementedError

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        raise NotImplementedError

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        raise NotImplementedError

    async def delete_node(self, node_id: str):
        raise NotImplementedError

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        raise NotImplementedError("Node embedding is not used in PathRag.")
    
    async def save(self):
        raise NotImplementedError
