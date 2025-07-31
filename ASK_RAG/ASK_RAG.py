import asyncio
import os
import json
import networkx as nx
import inspect
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Type, cast, List, Dict, Any, Optional, Tuple

from .utils import (
    EmbeddingFunc,
    limit_async_func_call,
    set_logger,
    logger
)
from .llm import (
    openai_embedding,
)
from .operate import(
    load_knowledge_graph,
    _subdivide_subgraph,
    generate_topic_wordlists,
    integrate_topic_groups_by_similarity
)
from .storage import(
    BaseGraphStorage,
    BaseVectorStorage,
    NetworkXStorage,
    NanoVectorDBStorage,
)
from .base import QueryParam
from .integrate import integrate
def lazy_external_import(module_name: str, class_name: str):
    """Lazily import a class from an external module based on the package of the caller."""
    caller_frame = inspect.currentframe().f_back
    module = inspect.getmodule(caller_frame)
    package = module.__package__ if module else None

    def import_class(*args, **kwargs):
        import importlib

  
        module = importlib.import_module(module_name, package=package)


        cls = getattr(module, class_name)
        return cls(*args, **kwargs)

    return import_class

Neo4JStorage = lazy_external_import(".kg.neo4j_impl", "Neo4JStorage")
OracleKVStorage = lazy_external_import(".kg.oracle_impl", "OracleKVStorage")
OracleGraphStorage = lazy_external_import(".kg.oracle_impl", "OracleGraphStorage")
OracleVectorDBStorage = lazy_external_import(".kg.oracle_impl", "OracleVectorDBStorage")
MilvusVectorDBStorge = lazy_external_import(".kg.milvus_impl", "MilvusVectorDBStorge")
MongoKVStorage = lazy_external_import(".kg.mongo_impl", "MongoKVStorage")
ChromaVectorDBStorage = lazy_external_import(".kg.chroma_impl", "ChromaVectorDBStorage")
TiDBKVStorage = lazy_external_import(".kg.tidb_impl", "TiDBKVStorage")
TiDBVectorDBStorage = lazy_external_import(".kg.tidb_impl", "TiDBVectorDBStorage")
AGEStorage = lazy_external_import(".kg.age_impl", "AGEStorage")

def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:

        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:

        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop

@dataclass
class ASK_RAG:
    """
    ASK_RAG: A model for topic-specific subgraph extraction from knowledge graphs.
    """
    # now adaptive to both windows and linux
    working_dir: str = field(
        default_factory=lambda: f"./ASK_RAG_cache_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )

    embedding_cache_config: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "similarity_threshold": 0.95,
            "use_llm_check": False,
        }
    )

    vector_storage: str = field(default="NanoVectorDBStorage")
    graph_storage: str = field(default="NetworkXStorage")

    current_log_level = logger.level
    log_level: str = field(default=current_log_level)

    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o-mini"


    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16

    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)

    enable_llm_cache: bool = True

    addon_params: dict = field(default_factory=dict)


    def __post_init__(self):
        log_file = os.path.join(self.working_dir,"ASK_RAG.log")
        set_logger(log_file)
        logger.setLevel(self.log_level)
        logger.info(f"Logger initialized for working directory: {self.working_dir}")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)
        
        self.vector_db_storage_cls: Type[BaseVectorStorage] = self._get_storage_class()[
            self.vector_storage
        ]
        self.graph_storage_cls: Type[BaseGraphStorage] = self._get_storage_class()[
            self.graph_storage
        ]  

        self._initialize_storage_components()

    def _get_storage_class(self) -> Any:
        return {
            # "JsonKVStorage": JsonKVStorage,
            "OracleKVStorage": OracleKVStorage,
            "MongoKVStorage": MongoKVStorage,
            "TiDBKVStorage": TiDBKVStorage,

            "NanoVectorDBStorage": NanoVectorDBStorage,
            "OracleVectorDBStorage": OracleVectorDBStorage,
            "MilvusVectorDBStorge": MilvusVectorDBStorge,
            "ChromaVectorDBStorage": ChromaVectorDBStorage,
            "TiDBVectorDBStorage": TiDBVectorDBStorage,

            "NetworkXStorage": NetworkXStorage,
            "Neo4JStorage": Neo4JStorage,
            "OracleGraphStorage": OracleGraphStorage,
            "AGEStorage": AGEStorage,
        }
    
    def _initialize_storage_components(self):
        """Initializes or re-initializes storage components."""
        logger.info("Initializing storage components...")
        current_global_config = asdict(self)

        self.entity_relation_graph = self.graph_storage_cls(
            namespace="graphs", global_config=current_global_config, embedding_func=self.embedding_func
        )
        self.entities_vdb = self.vector_db_storage_cls(
            namespace="entities", global_config=current_global_config, embedding_func=self.embedding_func,
            meta_fields={"entity_name", "structural_id"}
        )
        self.relationships_vdb = self.vector_db_storage_cls(
            namespace="relationships", global_config=current_global_config, embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id"}
        )
        logger.info("Storage components initialized.")
    
    def _clear_storage_components_data_sync(self):
        """Clears data from existing storage components synchronously."""
        logger.info("Clearing data from storage components (synchronously)...")
        
        storage_map = {
            "entity_relation_graph": self.entity_relation_graph,
            "entities_vdb": self.entities_vdb,
            "relationships_vdb": self.relationships_vdb
        }

        for name, component in storage_map.items():
            if hasattr(component, 'clear_data_sync'):
                logger.debug(f"Calling clear_data_sync() on {name} ({type(component).__name__})...")
                component.clear_data_sync()
            # Add specific fallbacks if clear_data_sync is not universally implemented yet
            elif isinstance(component, NetworkXStorage) and name == "entity_relation_graph":
                component._graph = nx.DiGraph()
                logger.debug(f"NetworkXStorage graph for {name} cleared directly.")
            else:
                logger.warning(f"No sync clear method or specific fallback for {name} of type {type(component).__name__}")
        logger.info("Data clearing attempted for storage components.")

    async def load_graph_as_current_context(self, graph_path: str):
        logger.info(f"Loading graph '{graph_path}' as current RAG context.")
        self._initialize_storage_components()
        self._clear_storage_components_data_sync()
        if not os.path.exists(graph_path):
            logger.error(f"GraphML file not found for context loading: {graph_path}")
            raise FileNotFoundError(f"GraphML file not found: {graph_path}")

        graph_obj = await self._aload_and_store_graph(graph_path)
        if graph_obj is None:
            logger.error(f"Failed to load graph into context: {graph_path}")
        else:
            logger.info(f"Graph '{graph_path}' successfully loaded as current context.")
        
    async def _aload_and_store_graph(self, path: str):
        """
        Load the graph from the given path, embed nodes using openai_embedding, 
        and store the graph structure and embeddings.
        """
        logger.info(f"Starting graph load and store process for: {path}")
        graph = await load_knowledge_graph(
            path,
            self.entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb
        )

        if graph is None:
             logger.error("Graph loading and storage process failed.")
             return None
        
        logger.info(f"Graph load and store process completed for: {path}")
        return graph
    
    def load_and_store_graph(self, path: str):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self._aload_and_store_graph(path))
    
    async def _subdivide_subgraph(self, query: str, param: Optional[QueryParam] = None) -> Optional[nx.DiGraph]:
        """
        Internal method to call the operate module's _subdivide_subgraph.
        Operates on the currently loaded RAG context.
        Returns the nx.DiGraph object or None.
        """
        if param is None:
            param = QueryParam()
        
        # operate_subdivide_subgraph is the imported function from operate.py
        # It uses the current state of self.entity_relation_graph and self.entities_vdb
        graph_nx = await _subdivide_subgraph(
            query=query,
            knowledge_graph_inst=self.entity_relation_graph,
            entities_vdb=self.entities_vdb,
            query_param=param
        )
        return graph_nx
    
    def run_hierarchical_subdivision(
        self,
        original_full_graph_path: str,
        threshold_config_for_agents: List[Dict[str, float]],
        max_depth: int = 4 
    ):
        loop = always_get_an_event_loop()
        logger.info(f"--- Starting Hierarchical Subdivision Orchestrator (Targeting up to G{max_depth}) ---")
        
        topic_wordlists_path = os.path.join(self.working_dir, "topic_wordlists.json")
        integration_groups_path = os.path.join(self.working_dir, "integration_group.json")
        try:
            with open(topic_wordlists_path, 'r', encoding='utf-8') as f: topic_wordlists_data = json.load(f)
            with open(integration_groups_path, 'r', encoding='utf-8') as f: integration_groups_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load prerequisite JSON files: {e}", exc_info=True); return
        if not isinstance(threshold_config_for_agents, list) or \
           len(threshold_config_for_agents) != len(topic_wordlists_data):
            logger.error("threshold_config_for_agents format/length mismatch. Aborting."); return

        initial_full_graph_abs_path = os.path.join(self.working_dir, original_full_graph_path)

        if not os.path.exists(initial_full_graph_abs_path):
            logger.error(f"Original full graph file not found at: {initial_full_graph_abs_path}. Aborting.")
            return

        all_topic_indices = list(range(1, len(topic_wordlists_data) + 1))
        current_level_parent_descriptors: List[Dict[str, Any]] = [{
            "filename": initial_full_graph_abs_path,
            "topic_indices_in_parent": all_topic_indices
        }]

        for n_current_parent_level in range(0, max_depth):
            n_graph_being_generated = n_current_parent_level + 1
            
            logger.info(f"--- Orchestrator: Starting Generation for G{n_graph_being_generated} (from G{n_current_parent_level} parents) ---")
            
            # --- Phase A: Partition ---
            logger.info(f"  Phase A: Partition of G{n_current_parent_level} parents into individual G{n_graph_being_generated}_K<topic_idx> files.")
            all_individual_graphs_generated_this_level: List[Dict[str, Any]] = []

            if not current_level_parent_descriptors:
                logger.info(f"No parent descriptors from G{n_current_parent_level} to generate G{n_graph_being_generated}. Stopping this level.")
                break 

            for parent_descriptor in current_level_parent_descriptors:
                graph_path_to_load = parent_descriptor["filename"]
                loop.run_until_complete(self.load_graph_as_current_context(graph_path_to_load))
                topics_to_use_for_subdivision = parent_descriptor["topic_indices_in_parent"]
                keywords_group_key = f"group_{n_graph_being_generated}" 

                for topic_idx_1based in topics_to_use_for_subdivision:
                    if not (0 <= topic_idx_1based - 1 < len(topic_wordlists_data)):
                        logger.warning(f"Topic index {topic_idx_1based} out of bounds for wordlists. Skipping G{n_graph_being_generated} generation for this topic from parent {parent_descriptor['filename']}.")
                        continue
                    wordlist_for_topic = topic_wordlists_data[topic_idx_1based - 1]
                    keywords = wordlist_for_topic.get("keywords", {}).get(keywords_group_key, [])
                    if not keywords:
                        logger.warning(f"    No '{keywords_group_key}' keywords for topic {topic_idx_1based}. Skipping G{n_graph_being_generated}_K{topic_idx_1based} from parent {parent_descriptor['filename']}.")
                        continue
                    query_string = ", ".join(list(set(keywords)))

                    if 0 <= topic_idx_1based - 1 < len(threshold_config_for_agents):
                        agent_thresh_config = threshold_config_for_agents[topic_idx_1based - 1]
                        current_min_sim = agent_thresh_config[keywords_group_key]
                    else:
                        logger.warning(f"Topic index {topic_idx_1based} for threshold config is out of bounds. Using default: {current_min_sim}")

                    query_param = QueryParam(
                        min_similarity_score=current_min_sim,
                        top_k=getattr(self, 'addon_params', {}).get(f"g{n_graph_being_generated}_subdivision_top_k", 100)
                    )
                    
                    logger.info(f"      Generating G{n_graph_being_generated}_K{topic_idx_1based}.graphml (min_sim={current_min_sim}) from parent {parent_descriptor['filename']} using keywords for topic {topic_idx_1based} from {keywords_group_key}. Quert String; {query_string}.")
                    subgraph_nx = loop.run_until_complete(self._subdivide_subgraph(query=query_string, param=query_param))
                    
                    if subgraph_nx and subgraph_nx.number_of_nodes() > 0:
                        logger.info(f"        Generated individual G{n_graph_being_generated}_K{topic_idx_1based} graph in memory.")
                        all_individual_graphs_generated_this_level.append({
                            "graph_obj": subgraph_nx,
                            "topic_idx": topic_idx_1based
                        })
                    else:
                        logger.warning(f"        Generated empty G{n_graph_being_generated}_K{topic_idx_1based}.graphml. Not saving.")
            
            if not all_individual_graphs_generated_this_level:
                logger.info(f"  Phase A for G{n_graph_being_generated} did not produce any individual subgraphs. Stopping further G-level generation.")
                break

            # --- Phase B: Aggregation ---
            logger.info(f"  Phase B: Aggregation of G{n_graph_being_generated} files based on integration_group.json.")
            descriptors_for_next_iteration_parents: List[Dict[str, Any]] = []
            integration_clusters_key = f"group_{n_graph_being_generated}"
            clusters_for_this_level_integration = integration_groups_data.get(integration_clusters_key, [])
            processed_indices_in_this_level_integration = set()

            for topic_indices_cluster in clusters_for_this_level_integration:
                sorted_cluster_indices = sorted(list(set(topic_indices_cluster)))
                integrated_filename_stem = "K" + "-".join(map(str, sorted_cluster_indices))
                full_integrated_filename = os.path.join(self.working_dir, f"G{n_graph_being_generated}_{integrated_filename_stem}.graphml")
                parent_descriptor_for_next_iteration = {
                    "filename": full_integrated_filename,
                    "topic_indices_in_parent": sorted_cluster_indices
                }
                
                if len(sorted_cluster_indices) == 1:
                    topic_idx = sorted_cluster_indices[0]
                    processed_indices_in_this_level_integration.add(topic_idx)
                    file_entry = next((f for f in all_individual_graphs_generated_this_level if f["topic_idx"] == topic_idx), None)
                    if file_entry:
                        individual_graph_obj = file_entry["graph_obj"]
                        individual_filename = f"G{n_graph_being_generated}_K{topic_idx}.graphml"
                        individual_full_path = os.path.join(self.working_dir, individual_filename)
                        nx.write_graphml(individual_graph_obj, individual_full_path)
                        logger.info(f"    G{n_graph_being_generated} Cluster of size 1: [{topic_idx}]. Saved to {individual_full_path} as parent for G{n_graph_being_generated+1}.")

                        parent_descriptor_for_next_iteration["filename"] = individual_full_path
                        descriptors_for_next_iteration_parents.append(parent_descriptor_for_next_iteration)
                    else:
                        logger.warning(f"    Individual graph G{n_graph_being_generated}_K{topic_idx}.graphml (for cluster [{topic_idx}]) not found in Phase A results. Cannot be parent.")
                
                elif len(sorted_cluster_indices) > 1:
                    logger.info(f"    Integrating G{n_graph_being_generated} graphs for topic cluster: {sorted_cluster_indices}")
                    graphs_to_union_list: List[nx.DiGraph] = []
                    for topic_idx in sorted_cluster_indices:
                        processed_indices_in_this_level_integration.add(topic_idx)
                        
                        file_entry = next((f for f in all_individual_graphs_generated_this_level if f["topic_idx"] == topic_idx), None)
                        if file_entry and file_entry.get("graph_obj") and file_entry["graph_obj"].number_of_nodes() > 0:
                            graphs_to_union_list.append(file_entry["graph_obj"])
                        else:
                            logger.warning(f"    Individual graph G{n_graph_being_generated}_K{topic_idx} for cluster {sorted_cluster_indices} was empty or not generated in Phase A and cannot be integrated.")
                    
                    if graphs_to_union_list:
                        integrated_graph = loop.run_until_complete(integrate(graphs_to_union_list))

                        if integrated_graph and integrated_graph.number_of_nodes() > 0:
                            nx.write_graphml(integrated_graph, parent_descriptor_for_next_iteration["filename"])
                            logger.info(f"      Saved integrated G{n_graph_being_generated} graph: {parent_descriptor_for_next_iteration['filename']}")
                            descriptors_for_next_iteration_parents.append(parent_descriptor_for_next_iteration)
                        else: 
                            logger.warning(f"      Integrated G{n_graph_being_generated} graph for cluster {sorted_cluster_indices} is empty or integration failed. Not saving or using as parent.")
                    else: 
                        logger.warning(f"    No G{n_graph_being_generated} graphs to union for cluster {sorted_cluster_indices}. Skipping integration for this cluster.")
            
            current_level_parent_descriptors = descriptors_for_next_iteration_parents
            if not current_level_parent_descriptors:
                logger.info(f"--- Orchestrator: No parent graphs available after integration for G{n_graph_being_generated}. Stopping hierarchy at G{n_graph_being_generated}. ---")
                break

        logger.info(f"--- Orchestrator: Hierarchical subdivision process finished (completed up to G{n_graph_being_generated}). ---")


    def save_storage(self):
        """Synchronous wrapper to save storage state."""
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self._asave_storage())

    async def _asave_storage(self):
        """Asynchronously saves the state of all storage components."""
        logger.info("Saving storage state...")
        save_tasks = []
        if hasattr(self.entities_vdb, 'save'):
            save_tasks.append(self.entities_vdb.save())
        if hasattr(self.relationships_vdb, 'save'):
            save_tasks.append(self.relationships_vdb.save())

        if save_tasks:
            await asyncio.gather(*save_tasks)
            logger.info("Storage state saved.")
        else:
            logger.warning("No storage components found with a save() method.")


    def ask(self, with_reference, question: str, system_prompt:str = None, param: Optional[QueryParam] = None) -> str:
        """
        Synchronously answer a user question by retrieving relevant knowledge and calling the LLM.
        """
        sys_prompt = system_prompt if system_prompt is not None else "You are an expert assistant. Use the provided context to answer the question."
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self._aask(with_reference, sys_prompt, question, param or QueryParam()))
    
    
    async def _aask(self, with_reference:bool, system_prompt:str, question: str, param: QueryParam = QueryParam()) -> str:
        """
        Asynchronously retrieve a subgraph for the question and call the LLM to generate an answer.
        """

        # 1. Retrieve the relevant subgraph from the knowledge graph
        subgraph = await _subdivide_subgraph(
            question,
            self.entity_relation_graph,
            self.entities_vdb,
            param
        )

        # 2. Serialize subgraph into a text context
        context_text = self._format_subgraph(with_reference, subgraph)
        
        # 3. Build messages for the LLM
        system_msg = {
            "role": "system",
            "content":[{"type": "text", "text": system_prompt}] 
        }
        user_msg = {
            "role": "user",
            "content": [{"type": "text", "text": f"Context:\n{context_text}\n\nQuestion: {question}"}]
        }

        # The 4th step will be called in Multi-Agent service

        # client = OpenAI()
        # # 4. Call OpenAI ChatCompletion
        # response = client.chat.completions.create(
        #     model=self.tiktoken_model_name,
        #     messages=[system_msg, user_msg],
        #     temperature=0.2,
        #     max_tokens= self.chunk_token_size // 2
        # )
        # answer = response.choices[0].message.content.strip()
        # return answer
        return [system_msg, user_msg]


    def _format_subgraph(self, subgraph) -> str:
        """
        Convert the returned subgraph object into a human-readable text context.
        Customize this based on your subgraph representation.
        """
        triples=[]
        for u, v, data in subgraph.edges(data=True):
            rel = data.get("relation", data.get("description", ""))  # relation or weight
            triples.append((subgraph.nodes[u]['name'], rel, subgraph.nodes[v]['name']))
        lines = []
        for src, rel, tgt in triples:
            lines.append(f"- {src} {rel} {tgt}")
        return "\n".join(lines)

    def generate_wordlists(self):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self._agenerate_wordlists())
    
    async def _agenerate_wordlists(
        self,
        integration_similarity_threshold: float = 0.70
    ) -> Optional[str]:
        """
        Generates topic wordlists, saves them, then integrates them by similarity.
        """

        wordlists_saved_path = await generate_topic_wordlists(
            output_dir=self.working_dir,
        )
        wordlists_saved_path = os.path.join(self.working_dir, "topic_wordlists.json")

        if wordlists_saved_path:
            logger.info(f"Wordlists successfully generated and saved to: {wordlists_saved_path}")
            
            logger.info("Starting similarity-based integration of topic groups...")
            if not hasattr(self, 'embedding_func') or self.embedding_func is None:
                logger.error("Embedding function is not configured in ASK_RAG instance.")
                return wordlists_saved_path

            integration_saved_path = await integrate_topic_groups_by_similarity(
                topic_wordlists_path=wordlists_saved_path,
                embedding_func=self.embedding_func,
                output_dir=self.working_dir,
                similarity_threshold=integration_similarity_threshold
            )
            if integration_saved_path:
                logger.info(f"Topic group integration by similarity saved to: {integration_saved_path}")
                return integration_saved_path
            else:
                logger.error("Failed to perform topic group integration by similarity.")
                return wordlists_saved_path
        else:
            logger.error("Failed to generate or save wordlists. Skipping integration.")
            return None
