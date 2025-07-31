import networkx as nx
import os
import html
import numpy as np
import asyncio
from typing import Any, Union, cast, Optional
from tqdm.asyncio import tqdm as tqdm_async
from nano_vectordb import NanoVectorDB
from dataclasses import dataclass
from .utils import (
    compute_mdhash_id,
    logger
)
from .base import (
    BaseGraphStorage,
    BaseVectorStorage,
    QueryParam
)
@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )
        self.cosine_better_than_threshold = self.global_config.get(
            "cosine_better_than_threshold", self.cosine_better_than_threshold
        )

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        async def wrapped_task(batch):
            result = await self.embedding_func(batch)
            pbar.update(1)
            return result

        embedding_tasks = [wrapped_task(batch) for batch in batches]
        pbar = tqdm_async(
            total=len(embedding_tasks), desc="Generating embeddings", unit="batch"
        )
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list)
        if len(embeddings) == len(list_data):
            for i, d in enumerate(list_data):
                d["__vector__"] = embeddings[i]
            results = self._client.upsert(datas=list_data)
            return results
        else:
            logger.error(
                f"embedding is not 1-1 with data, {len(embeddings)} != {len(list_data)}"
            )

    async def query(self, query_text: str, top_k: int = 5, min_similarity: Optional[float] = None) -> list[dict]:
        logger.debug(f"NanoVectorDB query received: top_k={top_k}, min_similarity={min_similarity}, query_text='{query_text[:50]}...'")
        embedding_result = await self.embedding_func([query_text])
        query_embedding: np.ndarray

        if isinstance(embedding_result, np.ndarray) and embedding_result.ndim > 1 and embedding_result.shape[0] == 1:
            query_embedding = embedding_result[0]
        elif isinstance(embedding_result, list) and len(embedding_result) == 1 and isinstance(embedding_result[0], np.ndarray):
            query_embedding = embedding_result[0]
        elif isinstance(embedding_result, np.ndarray) and embedding_result.ndim == 1:
            query_embedding = embedding_result
        else:
            logger.error(f"Unexpected embedding format from embedding_func: {type(embedding_result)}")
            raise ValueError("Embedding function returned an unexpected format for a single query.")

        threshold_for_vdb_client = self.cosine_better_than_threshold
        if min_similarity is not None:
            threshold_for_vdb_client = min_similarity
            logger.debug(f"  Using provided min_similarity for VDB client: {threshold_for_vdb_client}")
        else:
            logger.debug(f"  Using instance default cosine_better_than_threshold for VDB client: {threshold_for_vdb_client}")

        raw_results = self._client.query(
            query=query_embedding,
            top_k=top_k,
            better_than_threshold=threshold_for_vdb_client
        )
        
        results = []
        if raw_results:
            for dp in raw_results:
                if isinstance(dp, dict):
                    clean_result = {
                        "id": dp.get("__id__"), 
                        "distance": dp.get("__metrics__") 
                    }
                    metadata = {k: v for k, v in dp.items() if k not in ["__id__", "__vector__", "__metrics__"]}
                    clean_result["metadata"] = metadata
                    results.append(clean_result)
                else:
                    logger.warning(f"Encountered non-dict item in raw_results from VDB: {dp}")
        return results


    @property
    def client_storage(self):
        return getattr(self._client, "_NanoVectorDB__storage")

    async def delete_entity(self, entity_name: str):
        try:
            entity_id = [compute_mdhash_id(entity_name, prefix="ent-")]

            if self._client.get(entity_id):
                self._client.delete(entity_id)
                logger.info(f"Entity {entity_name} have been deleted.")
            else:
                logger.info(f"No entity found with name {entity_name}.")
        except Exception as e:
            logger.error(f"Error while deleting entity {entity_name}: {e}")

    async def delete_relation(self, entity_name: str):
        try:
            relations = [
                dp
                for dp in self.client_storage["data"]
                if dp["src_id"] == entity_name or dp["tgt_id"] == entity_name
            ]
            ids_to_delete = [relation["__id__"] for relation in relations]

            if ids_to_delete:
                self._client.delete(ids_to_delete)
                logger.info(
                    f"All relations related to entity {entity_name} have been deleted."
                )
            else:
                logger.info(f"No relations found for entity {entity_name}.")
        except Exception as e:
            logger.error(
                f"Error while deleting relations for entity {entity_name}: {e}"
            )

    async def index_done_callback(self):
        self._client.save()

    async def save(self):
        """Saves the NanoVectorDB index to its file."""
        logger.info(f"Saving NanoVectorDB state for namespace '{self.namespace}' to {self._client_file_name}")
        self._client.save()
    
    def clear_data_sync(self):
        logger.debug(f"Attempting to clear data from NanoVectorDBStorage (namespace: {self.namespace})")
        if not hasattr(self, '_client') or self._client is None:
            logger.warning(f"NanoVectorDB client not found during clear_data_sync for namespace {self.namespace}. Initializing.")
            self._client = NanoVectorDB(
                embedding_dim=self.embedding_func.embedding_dim,
                storage_file=self._client_file_name
            )
        current_data = getattr(self._client, '_NanoVectorDB__storage', {}).get('data', [])
        if not isinstance(current_data, list):
            logger.warning(f"NanoVectorDB internal data format unexpected for namespace {self.namespace}. Clearing might be incomplete.")
            current_data = []

        all_ids = [dp["__id__"] for dp in current_data if isinstance(dp, dict) and "__id__" in dp]

        if all_ids:
            logger.debug(f"Found {len(all_ids)} IDs to delete in NanoVectorDBStorage (namespace: {self.namespace}).")
            try:
                deleted_ids = self._client.delete(all_ids)
                logger.debug(f"Successfully deleted {len(deleted_ids)} IDs from NanoVectorDB (namespace: {self.namespace}).")
            except Exception as e:
                logger.error(f"Error during NanoVectorDB delete operation for namespace {self.namespace}: {e}", exc_info=True)
        else:
            logger.debug(f"No IDs found in NanoVectorDBStorage (namespace: {self.namespace}). Already clear or empty.")
        

@dataclass
class NetworkXStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name) -> nx.DiGraph:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.DiGraph, file_name):
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
        """
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        graph = cast(nx.Graph, largest_connected_component(graph))
        node_mapping = {
            node: html.unescape(node.upper().strip()) for node in graph.nodes()
        }
        graph = nx.relabel_nodes(graph, node_mapping)
        return NetworkXStorage._stabilize_graph(graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.DiGraph()
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def index_done_callback(self):
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        return self._graph.degree(node_id)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return self._graph.degree(src_id) + self._graph.degree(tgt_id)

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None
    async def get_node_in_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.in_edges(source_node_id))
        return None
    async def get_node_out_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.out_edges(source_node_id))
        return None
    
    async def get_pagerank(self,source_node_id:str):
        pagerank_list=nx.pagerank(self._graph)
        if source_node_id in pagerank_list:
            return pagerank_list[source_node_id]
        else:
            print("pagerank failed")

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        self._graph.add_node(node_id, **node_data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def delete_node(self, node_id: str):
        """
        Delete a node from the graph based on the specified node_id.
        :param node_id: The node_id to delete
        """
        if self._graph.has_node(node_id):
            self._graph.remove_node(node_id)
            logger.info(f"Node {node_id} deleted from the graph.")
        else:
            logger.warning(f"Node {node_id} not found in the graph for deletion.")

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    async def _node2vec_embed(self):
        from graspologic import embed

        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids
    
    async def edges(self, nbunch=None, data=False, default=None):
        return self._graph.edges(nbunch=nbunch, data=data, default=default)

    async def nodes(self, data=False, default=None):
         return self._graph.nodes(data=data, default=default)

    async def save(self):
        """Saves the NetworkX graph to its GraphML file."""
        logger.info(f"Saving NetworkX graph for namespace '{self.namespace}' to {self._graphml_xml_file}")
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    def clear_data_sync(self): # Synchronous for easier calling during reset
        logger.debug(f"Clearing data from NetworkXStorage (namespace: {self.namespace})")
        self._graph = nx.DiGraph() # Reset the graph