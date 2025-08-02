import os
import json
import sys
from ASK_RAG import ASK_RAG

# Configuration
WORKING_DIR = "./my_rag"
CONFIG_FILE = "threshold_config.json"
GRAPH_FILE = "YOUR_GRAPH.graphml" # <-- Specify your graph file here
API_KEY = "YOUR_API_KEY" # <-- Specify your API key here, or follow the readme to export in the terminal

# Setup Environment
os.environ["LLM_API_KEY"] = API_KEY
os.environ["LLM_BASE_URL"] = "https://api.openai.com/v1"

if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

# Load Configuration File
try:
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        threshold_config_for_agents = json.load(f)
except FileNotFoundError:
    print(f"Error: Configuration file not found at '{CONFIG_FILE}'")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from '{CONFIG_FILE}'. Please check the file format.")

# Initialize and Run the ASK_RAG instance
rag = ASK_RAG(working_dir=WORKING_DIR)

# Create Relevant Wordlists
try:
    rag.generate_wordlists()
    print("Successfully generate wordlists")
except Exception as e:
    print(f"Error: {e}")

# Run ASK_RAG
max_processing_depth = 4

print(f"Starting hierarchical subdivision for graph: '{GRAPH_FILE}'")
try:
    rag.run_hierarchical_subdivision(
        original_full_graph_path=GRAPH_FILE,
        threshold_config_for_agents=threshold_config_for_agents,
        max_depth=max_processing_depth,
    )
    print("Processing complete. Check the working directory for results.")
except Exception as e:
    print(f"An exception occurred during processing: {e}")