import asyncio
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Dict, List, Tuple

import networkx as nx
import numpy as np
from openai import AsyncOpenAI
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.asyncio import tqdm as tqdm_async

from .utils import (
    logger,
    EmbeddingFunc,
    UnlimitedSemaphore,
    compute_mdhash_id,
    wrap_embedding_func_with_attrs,
)
from .storage import (
    BaseGraphStorage,
)
from .base import (
    QueryParam,
    BaseVectorStorage,
)
from .llm import (
    call_llm_for_keywords
)
from .prompt import PROMPTS, topics


async def generate_topic_wordlists(
    topics_list: list[str] = topics,
    prompt_template: str = PROMPTS["keywords_extraction"],
    model: str = "gpt-4.1",
    client: AsyncOpenAI = None,
    output_dir: str = ".",
    output_filename: str = "topic_wordlists.json"
) -> list[dict]:
    """
    Generates hierarchical keyword lists for a list of topics using an LLM.
    """
    if client is None:
        client = AsyncOpenAI()

    all_wordlists_results = []
    logger.info(f"Starting keyword generation for {len(topics_list)} topics using model {model}...")

    tasks = []
    for topic in topics_list:
        formatted_prompt = prompt_template.format(topic=topic)
        tasks.append(call_llm_for_keywords(client, model, formatted_prompt))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(results):
        topic = topics_list[i]
        if isinstance(result, Exception):
            logger.error(f"Failed to generate keywords for topic '{topic}': {result}")
            all_wordlists_results.append({"topic": topic, "error": str(result), "keywords": None})
        elif result is None:
            logger.error(f"Failed to generate or parse keywords for topic '{topic}' (result was None).")
            all_wordlists_results.append({"topic": topic, "error": "Generation or Parsing Failed", "keywords": None})
        else:
            all_wordlists_results.append({"topic": topic, "error": None, "keywords": result})

    successful_count = sum(1 for wl in all_wordlists_results if wl['keywords'] is not None)
    logger.info(f"Finished keyword generation. Successfully generated for {successful_count}/{len(topics_list)} topics.")

    if not all_wordlists_results:
        logger.warning("No wordlist results were generated to save.")
        return None

    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_wordlists_results, f, indent=4, ensure_ascii=False)

    logger.info(f"Successfully saved topic wordlists to: {output_path}")
    return output_path


async def integrate_topic_groups_by_similarity(
    topic_wordlists_path: str,
    embedding_func: EmbeddingFunc,
    output_dir: str,
    output_filename: str = "integration_group.json",
    thresholds_by_group_level: Optional[Dict[str, float]] = None,
    similarity_threshold: float = 0.75  # Group if cosine similarity > threshold
) -> Optional[str]:
    """
    Embeds keyword groups from topic_wordlists.json, clusters them by similarity
    within each group level (group_1, group_2, etc.), and saves the integrated
    grouping of topic indices.
    """

    logger.info(f"Starting topic group integration from: {topic_wordlists_path}")
    try:
        with open(topic_wordlists_path, 'r', encoding='utf-8') as f:
            topic_wordlists_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Topic wordlists file not found: {topic_wordlists_path}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from: {topic_wordlists_path}")
        return None

    texts_to_embed_map = []  # Stores tuples of (text_string, original_topic_index_0_based, group_name)

    for i, topic_data in enumerate(topic_wordlists_data):
        if topic_data.get("keywords") and isinstance(topic_data["keywords"], dict):
            for group_name, keywords_list in topic_data["keywords"].items():
                if isinstance(keywords_list, list) and keywords_list: 
                    text_string = ", ".join(str(kw) for kw in keywords_list)
                    texts_to_embed_map.append({
                        "text": text_string,
                        "topic_index_0_based": i,
                        "group_name": group_name
                    })
        else:
            logger.warning(f"No keywords found or keywords format incorrect for topic index {i} ('{topic_data.get('topic', 'Unknown Topic')}'). Skipping.")

    if not texts_to_embed_map:
        logger.warning("No valid keyword groups found to embed for similarity integration.")
        return None
    
    all_text_strings = [item["text"] for item in texts_to_embed_map]

    try:
        logger.info(f"Generating embeddings for {len(all_text_strings)} keyword group strings...")
        all_embeddings = await embedding_func(all_text_strings)
        if len(all_embeddings) != len(all_text_strings):
            logger.error(f"Mismatch in number of embeddings ({len(all_embeddings)}) and texts ({len(all_text_strings)}).")
            return None
    except Exception as e:
        logger.error(f"Error during embedding generation: {e}", exc_info=True)
        return None

    embeddings_by_group_level = defaultdict(list)
    for i, item_map in enumerate(texts_to_embed_map):
        embeddings_by_group_level[item_map["group_name"]].append({
            "embedding": all_embeddings[i],
            "topic_index_0_based": item_map["topic_index_0_based"]
        })

    final_integrated_groups = {}
    logger.info(f"Clustering topic groups using custom or default thresholds...")
    for group_name, items_in_group_level in embeddings_by_group_level.items():
        if len(items_in_group_level) < 1:
            final_integrated_groups[group_name] = []
            continue
        if len(items_in_group_level) == 1:
            topic_idx_1_based = items_in_group_level[0]["topic_index_0_based"] + 1
            final_integrated_groups[group_name] = [[topic_idx_1_based]]
            continue

        current_embeddings_matrix = np.array([item["embedding"] for item in items_in_group_level])

        # Determine the similarity threshold for the current group_name
        current_sim_thresh = similarity_threshold
        if thresholds_by_group_level and group_name in thresholds_by_group_level:
            current_sim_thresh = thresholds_by_group_level[group_name]
            logger.info(f"  For {group_name}, using custom similarity threshold: {current_sim_thresh}")
        elif thresholds_by_group_level is not None:  # Dict provided but group_name not in it
            logger.info(f"  Threshold for {group_name} not in custom map. Using default: {current_sim_thresh}")

        # distance_threshold for AgglomerativeClustering: 1 - cosine_similarity_threshold
        # Linkage occurs if distance < distance_threshold
        distance_threshold = 1.0 - current_sim_thresh
        logger.info(f"  Calculated distance_threshold for {group_name}: {distance_threshold} (from sim_thresh: {current_sim_thresh})")

        cluster_model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='cosine',
            linkage='average'
        )

        try:
            cluster_labels = cluster_model.fit_predict(current_embeddings_matrix)
        except Exception as e:
            logger.error(f"  Error during clustering for {group_name}: {e}", exc_info=True)
            final_integrated_groups[group_name] = []
            continue

        # ... (Organize results into clusters - same as before) ...
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            topic_idx_1_based = items_in_group_level[i]["topic_index_0_based"] + 1
            clusters[label].append(topic_idx_1_based)

        processed_clusters = []
        for cluster_label in sorted(clusters.keys()):  # Sort by label for consistent order
            processed_clusters.append(sorted(list(set(clusters[cluster_label]))))  # Sort indices within cluster

        final_integrated_groups[group_name] = sorted(processed_clusters, key=lambda x: x[0] if x else 0)  # Sort clusters by first element
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_integrated_groups, f, ensure_ascii=False)
        logger.info(f"Successfully saved integrated topic groups to: {output_path}")
        return output_path
    except IOError as e:
        logger.error(f"Error saving integrated topic groups to file {output_path if 'output_path' in locals() else output_filename}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving integrated topic groups: {e}", exc_info=True)
        return None


async def load_knowledge_graph(path: str, knowledge_graph_inst: BaseGraphStorage, entities_vdb: BaseVectorStorage, relationships_vdb: BaseVectorStorage):
    """
    Load graph from GraphML path, store structure, extract data using GraphML keys
    (v_name, v_description, e_description), and upsert into VDBs.
    Structural node_id is stored as 'entity_name' metadata in entities_vdb.
    """
    # 1. Load GraphML
    try:
        G = nx.read_graphml(path)
        logger.info(f"Graph loaded successfully from {path} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    except FileNotFoundError:
        logger.error(f"GraphML file not found at {path}")
        return None
    except Exception as e:
        logger.error(f"Error loading graph from {path}: {e}")
        return None

    # 2. Store Graph Structure (using structural node_id)
    try:
        nodes_added = 0
        for node_id, node_data in G.nodes(data=True):
            processed_node_data = {}

            processed_node_data['v_type'] = node_data.get('v_type', node_data.get('type', ''))
            processed_node_data['v_name'] = node_data.get('v_name', node_data.get('name', node_id))
            processed_node_data['v_description'] = node_data.get('v_description', node_data.get('description', ''))

            for k, v in node_data.items():
                if k not in ['v_type', 'type', 'v_name', 'name', 'v_description', 'description']:
                    processed_node_data[k] = v

            await knowledge_graph_inst.upsert_node(node_id, processed_node_data)
            nodes_added += 1
        edges_added = 0
        for u, v, edge_data in G.edges(data=True):
            await knowledge_graph_inst.upsert_edge(u, v, edge_data)
            edges_added += 1
        logger.info(f"Stored graph structure: {nodes_added} nodes, {edges_added} edges.")
    except Exception as e:
        logger.error(f"Error storing graph structure: {e}")
        return None

    all_entities_data = []
    for node_id, node_data in G.nodes(data=True):
        entity_name = node_data.get("v_name", node_id) 
        description = node_data.get("v_description", "") 
        entity_info = {
            "node_id": node_id, 
            "entity_name": entity_name, 
            "description": description
        }
        all_entities_data.append(entity_info)

    all_relationships_data = []
    for u, v, edge_data in G.edges(data=True):
        description = edge_data.get("e_description", "") 
        rel_info = {
            "src_id": u,
            "tgt_id": v,
            "description": description
        }
        all_relationships_data.append(rel_info)
    logger.info(f"Extracted data for {len(all_entities_data)} entities and {len(all_relationships_data)} relationships using GraphML keys.")

    if entities_vdb is not None:
        if all_entities_data:
            logger.info(f"Preparing {len(all_entities_data)} entities for vector DB upsert.")
            data_for_entity_vdb = {}
            for dp in all_entities_data:
                node_id = dp.get("node_id")
                entity_name = dp.get("entity_name").upper()
                description = dp.get("description", "")
                if node_id is None: continue

                vdb_key = compute_mdhash_id(str(entity_name), prefix="ent-")

                content = str(entity_name) + " " + str(description)

                metadata_for_vdb = {"entity_name": entity_name, "structural_id": node_id}

                data_for_entity_vdb[vdb_key] = {
                    "content": content,
                    **metadata_for_vdb
                }

            if data_for_entity_vdb:
                logger.info(f"Calling entity vector storage upsert for {len(data_for_entity_vdb)} items...")
                try:
                    await entities_vdb.upsert(data_for_entity_vdb)
                    logger.info(f"Entity vector storage upsert call finished.")
                except Exception as e:
                    logger.error(f"Error during entity vector DB upsert: {e}", exc_info=True) 
            else:
                logger.info("No valid entity data prepared for vector DB upsert.")
        else:
            logger.info("No entity data extracted, skipping entity vector DB upsert.")
    else:
        logger.warning("Entities vector DB (entities_vdb) is None, skipping entity upsert.")


    if relationships_vdb is not None:
        if all_relationships_data:
            logger.info(f"Preparing {len(all_relationships_data)} relationships for vector DB upsert.")
            data_for_rel_vdb = {}
            for dp in all_relationships_data:
                src_id = dp.get("src_id")
                tgt_id = dp.get("tgt_id")
                description = dp.get("description", "")

                if src_id is None or tgt_id is None: continue 

                vdb_key = compute_mdhash_id(src_id + tgt_id, prefix="rel-")

                content = src_id + tgt_id + description

                metadata_for_vdb = {"src_id": src_id, "tgt_id": tgt_id}

                data_for_rel_vdb[vdb_key] = {
                    "content": content,
                    **metadata_for_vdb
                }

            if data_for_rel_vdb:
                logger.info(f"Calling relationship vector storage upsert for {len(data_for_rel_vdb)} items...")
                try:
                    await relationships_vdb.upsert(data_for_rel_vdb)
                    logger.info(f"Relationship vector storage upsert call finished.")
                except Exception as e:
                    logger.error(f"Error during relationship vector DB upsert: {e}", exc_info=True) 
            else:
                logger.info("No valid relationship data prepared for vector DB upsert.")
        else:
            logger.info("No relationship data extracted, skipping relationship vector DB upsert.")
    else:
        logger.warning("Relationships vector DB (relationships_vdb) is None, skipping relationship upsert.")

    logger.info("load_knowledge_graph finished successfully.")
    return G


# temporary
async def get_subgraph(
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    query_param: QueryParam,
    wordlists: list,
):
    query = ", ".join(wordlists[0]["low_level_keywords"])
    await _subdivide_subgraph(query, knowledge_graph_inst, entities_vdb, query_param)


async def _subdivide_subgraph(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    query_param: QueryParam,
) -> Optional[nx.DiGraph]:
    min_sim_display = query_param.min_similarity_score if query_param.min_similarity_score is not None else 'VDB_default'
    logger.info(f"Querying VDB for top {query_param.top_k} entities (min_sim={min_sim_display}) for query: '{query[:100]}...'")

    results = await entities_vdb.query(
        query_text=query,
        top_k=query_param.top_k,
        min_similarity=query_param.min_similarity_score
    )

    if not results:
        logger.warning("VDB query returned no results for the given query.")
        return None  # Nothing more to do

    structural_ids_found = [
        r['metadata'].get('structural_id')
        for r in results
        if r.get('metadata') and r['metadata'].get('structural_id')
    ]

    if not structural_ids_found:
        logger.warning("No structural_ids found in VDB results metadata.")
        return None

    logger.info(f"Attempting to fetch {len(structural_ids_found)} nodes from graph storage using structural IDs...")

    nodes_fetched_map = {}
    for struct_id in structural_ids_found:
        if struct_id:
            node_data = await knowledge_graph_inst.get_node(struct_id)
            if node_data is not None:
                nodes_fetched_map[struct_id] = node_data
            else:
                logger.warning(f"Node with structural_id '{struct_id}' not found in graph storage.")

    if not nodes_fetched_map:
        logger.error("Failed to fetch any nodes from graph storage using structural IDs found in VDB.")
        return

    logger.info(f"Successfully fetched data for {len(nodes_fetched_map)} nodes from graph storage.")

    logger.info(f"Fetching node degrees for {len(nodes_fetched_map)} nodes...")

    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(struct_id) for struct_id in nodes_fetched_map.keys()]
    )
    degrees_map = dict(zip(nodes_fetched_map.keys(), node_degrees))

    node_info_for_pathfinding = []
    vdb_results_map = {
        r['metadata']['structural_id']: r
        for r in results
        if r.get('metadata') and r['metadata'].get('structural_id')
    }

    for struct_id, graph_data in nodes_fetched_map.items():
        info = {
            "graph_node_id": struct_id,
            "graph_node_data": graph_data,
            "degree": degrees_map.get(struct_id),
            "vdb_result": vdb_results_map.get(struct_id)
        }
        node_info_for_pathfinding.append(info)

    if not node_info_for_pathfinding:
        logger.error("No valid node information available for pathfinding step.")
        return None

    logger.info(f"Proceeding to find related edges/paths for {len(node_info_for_pathfinding)} source nodes.")

    sub_G, final_result_edges = await _find_most_related_edges_from_entities(
        node_info_for_pathfinding, query_param, knowledge_graph_inst
    )

    if sub_G is not None and sub_G.number_of_nodes() > 0:
        logger.info(f"Generated subgraph with {sub_G.number_of_nodes()} nodes and {sub_G.number_of_edges()} edges.")
        return sub_G
    else:
        logger.warning("Subgraph generation in _subdivide_subgraph resulted in an empty or None graph.")
        return None


async def find_paths_and_edges_with_stats(graph, target_nodes, max_paths_limit: int = 5000):

    result = defaultdict(lambda: {"paths": [], "edges": set()})
    path_stats = {"1-hop": 0, "2-hop": 0, "3-hop": 0}
    one_hop_paths = []
    two_hop_paths = []
    three_hop_paths = []

    async def dfs(current, target, path, depth):

        if depth > 3:
            return
        if current == target:
            result[(path[0], target)]["paths"].append(list(path))
            for u, v in zip(path[:-1], path[1:]):
                result[(path[0], target)]["edges"].add(tuple(sorted((u, v))))
            if depth == 1:
                path_stats["1-hop"] += 1
                one_hop_paths.append(list(path))
            elif depth == 2:
                path_stats["2-hop"] += 1
                two_hop_paths.append(list(path))
            elif depth == 3:
                path_stats["3-hop"] += 1
                three_hop_paths.append(list(path))
            return
        neighbors = graph.neighbors(current)
        for neighbor in neighbors:
            if neighbor not in path:
                await dfs(neighbor, target, path + [neighbor], depth + 1)

    for node1 in target_nodes:
        for node2 in target_nodes:
            if node1 != node2:
                await dfs(node1, node2, [node1], 0)

    for key in result:
        result[key]["edges"] = list(result[key]["edges"])

    return dict(result), path_stats, one_hop_paths, two_hop_paths, three_hop_paths


def bfs_weighted_paths(G, path, source, target, threshold, alpha):
    results = []
    edge_weights = defaultdict(float)
    node = source
    follow_dict = {}

    for p in path:
        for i in range(len(p) - 1):
            current = p[i]
            next_num = p[i + 1]

            if current in follow_dict:
                follow_dict[current].add(next_num)
            else:
                follow_dict[current] = {next_num}

    for neighbor in follow_dict[node]:
        edge_weights[(node, neighbor)] += 1 / len(follow_dict[node])

        if neighbor == target:
            results.append(([node, neighbor]))
            continue

        if edge_weights[(node, neighbor)] > threshold:

            for second_neighbor in follow_dict[neighbor]:
                weight = edge_weights[(node, neighbor)] * alpha / len(follow_dict[neighbor])
                edge_weights[(neighbor, second_neighbor)] += weight

                if second_neighbor == target:
                    results.append(([node, neighbor, second_neighbor]))
                    continue

                if edge_weights[(neighbor, second_neighbor)] > threshold:

                    for third_neighbor in follow_dict[second_neighbor]:
                        weight = edge_weights[(neighbor, second_neighbor)] * alpha / len(follow_dict[second_neighbor])
                        edge_weights[(second_neighbor, third_neighbor)] += weight

                        if third_neighbor == target:
                            results.append(([node, neighbor, second_neighbor, third_neighbor]))
                            continue
    path_weights = []
    for p in path:
        path_weight = 0
        for i in range(len(p) - 1):
            edge = (p[i], p[i + 1])
            path_weight += edge_weights.get(edge, 0)
        path_weights.append(path_weight / (len(p) - 1))

    combined = [(p, w) for p, w in zip(path, path_weights)]

    return combined


async def _find_most_related_edges_from_entities(
    node_info_list: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    """
    Builds a graph G from storage, finds paths between source nodes (identified by graph_node_id),
    weights paths, and constructs a final subgraph sub_G containing the most relevant paths/edges.
    """
    sub_G = nx.Graph()
    source_node_ids = [
        node_info["graph_node_id"]
        for node_info in node_info_list
        if "graph_node_id" in node_info
    ]

    if len(source_node_ids) < 2:
        logger.warning(f"Need at least 2 source nodes for pathfinding, received {len(source_node_ids)}. Adding received nodes to subgraph and returning.")
        for node_info in node_info_list:
            node_id = node_info.get("graph_node_id")
            node_data = node_info.get("graph_node_data", {})
            if node_id:
                sub_G.add_node(node_id, **node_data)
        return sub_G, []

    logger.info("Building temporary NetworkX graph G from storage for pathfinding...")
    G = nx.Graph()
    try:
        nodes_iter = await knowledge_graph_inst.nodes(data=True)
        edges_iter = await knowledge_graph_inst.edges(data=True)
        G.add_nodes_from(nodes_iter)
        G.add_edges_from(edges_iter)
        logger.info(f"Built temporary graph G with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except Exception as e:
        logger.error(f"Failed to build temporary graph G from storage: {e}")
        return sub_G, []

    valid_source_node_ids = [node_id for node_id in source_node_ids if G.has_node(node_id)]

    if len(valid_source_node_ids) < 2:
        logger.warning(f"After building G, only {len(valid_source_node_ids)} source nodes exist in it. Need at least 2 for pathfinding.")
        for node_id in valid_source_node_ids:
            sub_G.add_node(node_id, **G.nodes.get(node_id, {}))
        return sub_G, []

    logger.info(f"Finding paths between {len(valid_source_node_ids)} source nodes in G: {valid_source_node_ids}")
    result_paths_data, path_stats, _, _, _ = await find_paths_and_edges_with_stats(G, valid_source_node_ids)
    logger.info(f"Path stats between source nodes: {path_stats}")
    if not result_paths_data:
        logger.warning("No paths found between any pair of source nodes up to depth 3.")
        for node_id in valid_source_node_ids:
            sub_G.add_node(node_id, **G.nodes.get(node_id, {}))
        return sub_G, []

    threshold = 0.3
    alpha = 0.8
    all_weighted_paths = []

    for node_id in valid_source_node_ids:
        node_data = G.nodes.get(node_id, {})
        node_attrs_for_subgraph = {}
        node_attrs_for_subgraph['v_name'] = node_data.get('v_name', node_data.get('name', node_id))
        node_attrs_for_subgraph['v_description'] = node_data.get('v_description', node_data.get('description', ''))

        for k, v in node_data.items():
            if k not in ['v_name', 'v_description', 'name', 'description']:
                node_attrs_for_subgraph[k] = v

        sub_G.add_node(node_id, **node_attrs_for_subgraph)

    logger.info(f"Calculating weighted paths between node pairs...")
    processed_pairs_count = 0
    # Iterate through pairs using the validated source node IDs
    for node1 in valid_source_node_ids:
        for node2 in valid_source_node_ids:
            if node1 != node2:
                path_data = result_paths_data.get((node1, node2)) 

                if path_data and path_data['paths']:
                    processed_pairs_count += 1
                    paths = path_data['paths']
                    results_for_pair = bfs_weighted_paths(G, paths, node1, node2, threshold, alpha)
                    all_weighted_paths.extend(results_for_pair)

    logger.info(f"Calculated weights for paths between {processed_pairs_count} node pairs.")
    if not all_weighted_paths:
        logger.warning("No weighted paths generated (threshold might be too high?). Returning subgraph with only source nodes.")
        return sub_G, []

    all_weighted_paths.sort(key=lambda x: x[1], reverse=True)

    # --- Uncomment this part if the query is badly time-costing ---
    # max_paths_to_add = 5000 
    # if len(all_weighted_paths) > max_paths_to_add:
    #     logger.info(f"Limiting weighted paths from {len(all_weighted_paths)} to top {max_paths_to_add} for subgraph construction.")
    #     all_weighted_paths = all_weighted_paths[:max_paths_to_add]
    # ----------------------------------------------

    # Add Edges (and intermediate nodes) based on weighted paths to the single sub_G
    seen_edges = set()
    final_result_edges = []

    logger.info(f"Adding top weighted path edges to subgraph...")
    edges_added_count = 0

    for path, weight in all_weighted_paths:
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]

            if not sub_G.has_node(u):
                node_data_u = G.nodes.get(u, {})
                node_attrs_for_subgraph_u = {}
                node_attrs_for_subgraph_u['v_name'] = node_data_u.get('v_name', node_data_u.get('name', u))
                node_attrs_for_subgraph_u['v_description'] = node_data_u.get('v_description', node_data_u.get('description', ''))
                for k, val in node_data_u.items():
                    if k not in ['v_name', 'v_description', 'name', 'description']:
                        node_attrs_for_subgraph_u[k] = val
                sub_G.add_node(u, **node_attrs_for_subgraph_u)

            if not sub_G.has_node(v):
                node_data_v = G.nodes.get(v, {})
                node_attrs_for_subgraph_v = {}
                node_attrs_for_subgraph_v['v_name'] = node_data_v.get('v_name', node_data_v.get('name', v))
                node_attrs_for_subgraph_v['v_description'] = node_data_v.get('v_description', node_data_v.get('description', ''))
                for k, val in node_data_v.items():
                    if k not in ['v_name', 'v_description', 'name', 'description']:
                        node_attrs_for_subgraph_v[k] = val
                sub_G.add_node(v, **node_attrs_for_subgraph_v)

            edge_tuple = tuple(sorted((u, v)))

            if edge_tuple not in seen_edges:
                edge_data = G.get_edge_data(u, v, default={})

                unified_edge_attrs = {}
                unified_edge_attrs['e_description'] = edge_data.get('e_description', edge_data.get('description', edge_data.get('relation', '')))
                unified_edge_attrs['e_chunks_id'] = edge_data.get('e_chunks_id', edge_data.get('chunks_id', None))

                sub_G.add_edge(u, v, **unified_edge_attrs)
                seen_edges.add(edge_tuple)
                final_result_edges.append((u, v, sub_G.edges[u, v]))
                edges_added_count += 1

    logger.info(f"Added {edges_added_count} unique edges from {len(all_weighted_paths)} weighted paths considered.")
    logger.info(f"Final subgraph has {sub_G.number_of_nodes()} nodes and {sub_G.number_of_edges()} edges.")

    return sub_G, final_result_edges