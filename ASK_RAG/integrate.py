import asyncio
import logging
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
from openai import AsyncOpenAI
from typing import List, Optional, Dict, Any, Tuple

from .llm import call_llm_for_edge_prediction
from .prompt import PROMPTS
from .utils import logger

# Model Hyperparameters
LLM_MODEL = "gpt-4.1"
EMBEDDING_DIM = 128
HIDDEN_CHANNELS = 64
OUT_CHANNELS = 32
LEARNING_RATE = 0.01
EPOCHS = 200
NUM_PREDICTIONS = 10


class GCNLinkPredictor(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden_channels, out_channels):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, edge_index, node_ids=None):
        x = self.embedding.weight if node_ids is None else self.embedding(node_ids)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

    def decode_all(self, z):
        return z @ z.t()

def train(model: GCNLinkPredictor, data: Data, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    optimizer.zero_grad()
    node_indices = torch.arange(data.num_nodes, device=data.edge_index.device)
    z = model.encode(data.edge_index, node_ids=node_indices)
    pos_logits = model.decode(z, data.edge_label_index[:, data.edge_label == 1])
    neg_logits = model.decode(z, data.edge_label_index[:, data.edge_label == 0])
    logits = torch.cat([pos_logits, neg_logits])
    labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model: GCNLinkPredictor, data: Data) -> float:
    model.eval()
    node_indices = torch.arange(data.num_nodes, device=data.edge_index.device)
    z = model.encode(data.edge_index, node_ids=node_indices)
    pos_logits = model.decode(z, data.edge_label_index[:, data.edge_label == 1])
    neg_logits = model.decode(z, data.edge_label_index[:, data.edge_label == 0])
    logits = torch.cat([pos_logits, neg_logits])
    labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
    predictions = (logits > 0).float()
    return (predictions == labels).float().mean().item()

async def generate_edge_descriptions(
    candidate_edges: List[Tuple[int, int]],
    original_graph: nx.Graph,
    idx_to_id_map: Dict[int, Any],
    prompt_template: str,
    model_llm: str,
    client: AsyncOpenAI
) -> List[Dict[str, Any]]:
    logger.info(f"Starting description generation for {len(candidate_edges)} candidate edges using LLM: {model_llm}...")
    tasks = []
    for node1_idx, node2_idx in candidate_edges:
        node1_id = idx_to_id_map.get(node1_idx)
        node2_id = idx_to_id_map.get(node2_idx)
        if node1_id is None or node2_id is None:
            logger.warning(f"Could not map indices {node1_idx}, {node2_idx} back to original IDs. Skipping edge description.")
            tasks.append(asyncio.create_task(asyncio.sleep(0, result={"nodes_id": (node1_idx, node2_idx), "description": "Error: Node ID mapping failed"})))
            continue
        try:
            node1_attrs = original_graph.nodes[node1_id]
            node2_attrs = original_graph.nodes[node2_id]
        except KeyError as e:
            logger.warning(f"Node ID {e} not found in original_graph. Skipping edge {node1_id}-{node2_id} description.")
            tasks.append(asyncio.create_task(asyncio.sleep(0, result={"nodes_id": (node1_id, node2_id), "description": f"Error: Node {e} not found"})))
            continue
        node1_info_str = (
            f"Name: {node1_attrs.get('name', node1_attrs.get('v_name', 'N/A'))}\n"
            f"Description: {node1_attrs.get('description', node1_attrs.get('v_description', 'N/A'))}"
        )
        node2_info_str = (
            f"Name: {node2_attrs.get('name', node2_attrs.get('v_name', 'N/A'))}\n"
            f"Description: {node2_attrs.get('description', node2_attrs.get('v_description', 'N/A'))}"
        )
        formatted_prompt = prompt_template.format(node1_info_str=node1_info_str, node2_info_str=node2_info_str)
        tasks.append(call_llm_for_edge_prediction(client, model_llm, formatted_prompt))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    processed_results = []
    for i, res in enumerate(results):
        edge_idx_pair = candidate_edges[i]
        node1_orig_id = idx_to_id_map.get(edge_idx_pair[0], f"idx_{edge_idx_pair[0]}")
        node2_orig_id = idx_to_id_map.get(edge_idx_pair[1], f"idx_{edge_idx_pair[1]}")
        edge_orig_id_pair = (node1_orig_id, node2_orig_id)
        description = None
        if isinstance(res, Exception):
            logger.error(f"LLM call for edge {edge_orig_id_pair[0]}<->{edge_orig_id_pair[1]} failed: {res}")
            description = f"Error during LLM call: {str(res)[:100]}"
        elif isinstance(res, dict) and res.get("description") and "Error:" in res.get("description", ""):
             description = res.get("description")
             logger.warning(f"Pre-packaged error for edge {edge_orig_id_pair[0]}<->{edge_orig_id_pair[1]}: {description}")
        else:
            description = res
            if isinstance(description, dict):
                 description_str = description.get("relationship_type", str(description))
                 logger.info(f"LLM output for edge {edge_orig_id_pair[0]}<->{edge_orig_id_pair[1]}: {description_str}")
                 description = description_str
            else:
                 logger.info(f"LLM output for edge {edge_orig_id_pair[0]}<->{edge_orig_id_pair[1]}: {str(description)[:100]}")
        processed_results.append({"nodes_id": edge_orig_id_pair, "description": description})
    return processed_results

async def integrate(
    graphs_to_integrate: List[nx.DiGraph],
) -> Optional[nx.DiGraph]:
    """
    Integrates a list of knowledge subgraphs, predicts new links using GCN,
    generates textual descriptions for them using an LLM, and returns the
    final augmented graph.

    Args:
        graphs_to_integrate: A list of NetworkX DiGraph objects to integrate.

    Returns:
        The final integrated and augmented NetworkX DiGraph, or None on failure.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Starting integration process for {len(graphs_to_integrate)} graphs on device: {device}")

    if not graphs_to_integrate:
        logger.warning("No graphs provided to integrate. Returning None.")
        return None

    logger.info("Composing input graphs...")
    nx_merged_graph = graphs_to_integrate[0].copy()
    for i in range(1, len(graphs_to_integrate)):
        nx_merged_graph = nx.compose(nx_merged_graph, graphs_to_integrate[i])
    
    if nx_merged_graph.number_of_nodes() == 0:
        logger.warning("Merged graph has no nodes. Aborting integration.")
        return nx_merged_graph 

    logger.info(f"Composed merged graph: {nx_merged_graph.number_of_nodes()} nodes, {nx_merged_graph.number_of_edges()} edges")

    original_ids = list(nx_merged_graph.nodes())
    id_to_idx_map = {node_id: i for i, node_id in enumerate(original_ids)}
    idx_to_id_map = {i: node_id for i, node_id in enumerate(original_ids)}
    reindexed_nx = nx.relabel_nodes(nx_merged_graph, id_to_idx_map, copy=True)
    
    if reindexed_nx.number_of_edges() > 0:
        edge_index = from_networkx(reindexed_nx).edge_index
    else:
        edge_index = torch.empty((2,0), dtype=torch.long)
    pyg_data_merged = Data(edge_index=edge_index)
    pyg_data_merged.num_nodes = reindexed_nx.number_of_nodes()
    logger.info(f"Prepared PyG data: {pyg_data_merged.num_nodes} nodes, {pyg_data_merged.num_edges // 2 if pyg_data_merged.is_undirected() else pyg_data_merged.num_edges} unique edges for PyG.")

    if pyg_data_merged.num_nodes == 0 or pyg_data_merged.num_edges == 0 : # Added num_nodes check
        logger.warning("Merged graph has no nodes or no edges for GCN. Skipping GCN link prediction.")
        return nx_merged_graph # Return the merged graph without GCN augmentation

    try:
        undirected_converter = ToUndirected()
        pyg_data_undirected = undirected_converter(pyg_data_merged)
        if pyg_data_undirected.num_edges == 0:
            logger.warning("Merged graph has no edges after ToUndirected. GCN link prediction will be skipped.")
            return nx_merged_graph
        transform = RandomLinkSplit(is_undirected=True, add_negative_train_samples=True, num_val=0.1, num_test=0.1)
        train_data, val_data, test_data = transform(pyg_data_undirected)
    except Exception as e:
        logger.error(f"Error during graph transformation for GCN: {e}", exc_info=True)
        return nx_merged_graph 

    train_data, val_data, test_data = train_data.to(device), val_data.to(device), test_data.to(device)

    gcn_model = GCNLinkPredictor(
        num_nodes=pyg_data_merged.num_nodes,
        embedding_dim=EMBEDDING_DIM,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUT_CHANNELS
    ).to(device)
    optimizer = torch.optim.Adam(params=gcn_model.parameters(), lr=LEARNING_RATE)

    logger.info("Starting GCN model training...")
    for epoch in range(1, EPOCHS + 1):
        loss = train(gcn_model, train_data, optimizer)
        if epoch % 20 == 0 or epoch == EPOCHS:
            val_acc = test(gcn_model, val_data)
            logger.info(f'Epoch: {epoch:03d}/{EPOCHS}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')
    logger.info("GCN Training finished.")

    logger.info("Predicting hidden relationships with trained GCN...")
    gcn_model.eval()
    with torch.no_grad():
        all_node_indices = torch.arange(pyg_data_merged.num_nodes, device=device)
        final_embeddings = gcn_model.encode(train_data.edge_index, node_ids=all_node_indices)
        adj_pred_scores = gcn_model.decode_all(final_embeddings)

    mask = torch.ones_like(adj_pred_scores, dtype=torch.bool, device=device)
    existing_edges_for_mask = pyg_data_undirected.edge_index.to(device) 
    mask[existing_edges_for_mask[0], existing_edges_for_mask[1]] = False
    mask.fill_diagonal_(False)
    potential_scores = adj_pred_scores[mask]

    if potential_scores.numel() == 0:
        logger.info("No potential new edges found after masking existing ones.")
        return nx_merged_graph
        
    potential_indices = mask.nonzero(as_tuple=False)
    num_candidates = min(NUM_PREDICTIONS, len(potential_scores))

    if num_candidates == 0:
        logger.info("No potential new edges found to process based on NUM_PREDICTIONS or available scores.")
        return nx_merged_graph

    top_k_scores, top_k_indices_in_potential = torch.topk(potential_scores, num_candidates)
    top_candidate_edges_pyg_indices = [tuple(potential_indices[i].tolist()) for i in top_k_indices_in_potential]

    edge_prediction_prompt = PROMPTS.get("edge_prediction")
    if not edge_prediction_prompt:
        logger.error("PROMPTS['edge_prediction'] not found. Cannot generate edge descriptions.")
        return nx_merged_graph # Return the current graph if prompt is missing

    client = AsyncOpenAI()
    edge_descriptions = await generate_edge_descriptions(
        candidate_edges=top_candidate_edges_pyg_indices,
        original_graph=nx_merged_graph,
        idx_to_id_map=idx_to_id_map,
        prompt_template=edge_prediction_prompt,
        model_llm=LLM_MODEL,
        client=client
    )
    
    output_graph = nx_merged_graph.copy() # output_graph 已经是合并后的图，其节点/边属性应该已经处理过
    added_count = 0
    for result in edge_descriptions:
        if result.get("description") and not str(result["description"]).startswith("Error:"):
            node1_id, node2_id = result["nodes_id"]
            if output_graph.has_node(node1_id) and output_graph.has_node(node2_id):
                if not output_graph.has_edge(node1_id, node2_id):
                    new_edge_attrs = {
                        'e_description': str(result["description"]),
                        'e_chunks_id': -1 
                    }

                    output_graph.add_edge(node1_id, node2_id, **new_edge_attrs)
                    added_count += 1
                    logger.info(f"Added new edge '{node1_id}'<->'{node2_id}' with description: {str(result['description'])[:100]}")
                else:
                    logger.info(f"Edge '{node1_id}'<->'{node2_id}' already exists. Not adding predicted link.")
            else:
                logger.warning(f"Nodes {node1_id} or {node2_id} not in output_graph. Cannot add edge.")


    if added_count > 0:
        logger.info(f"Integration complete. {added_count} new edges were described and added to the graph.")
    else:
        logger.warning("No new edges with valid descriptions were added. Graph structure from GCN prediction phase remains.")
    
    return output_graph