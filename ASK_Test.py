import os
from ASK_RAG import ASK_RAG

WORKING_DIR = "./my_rag"  # Set the appropriate directory path

api_key = "YOUR_API_KEY"
os.environ["OPENAI_API_KEY"] = api_key
base_url = "https://api.openai.com/v1"
os.environ["OPENAI_API_BASE"] = base_url

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Initialize the TGraphRAG instance
rag = ASK_RAG(working_dir=WORKING_DIR)

threshold_config_for_agents = [
    # Agent 1: Device Selection and Compatibility for Astronomical Imaging
    { "group_1": 0.20, "group_2": 0.15, "group_3": 0.10, "group_4": 0.05 },
    # Agent 2: Shooting Time, Location, and Target Compatibility in Astronomy
    { "group_1": 0.20, "group_2": 0.15, "group_3": 0.10, "group_4": 0.05 },
    # Agent 3: Exposure Time and Parameter Evaluation in Astrophotography
    { "group_1": 0.20, "group_2": 0.15, "group_3": 0.10, "group_4": 0.05 },
    # Agent 4: Star Point Morphology and Tracking Accuracy in Astrophotography
    { "group_1": 0.20, "group_2": 0.15, "group_3": 0.10, "group_4": 0.05 },
    # Agent 5: Halo and Artifact Detection in Astronomical Images
    { "group_1": 0.10, "group_2": 0.10, "group_3": 0.05, "group_4": 0.05 },
    # Agent 6: Calibration Frame Effectiveness and Quality in Astronomy
    { "group_1": 0.20, "group_2": 0.15, "group_3": 0.10, "group_4": 0.05 },
    # Agent 7: Color Balance and Ground Truth Comparison in Image Processing
    { "group_1": 0.10, "group_2": 0.10, "group_3": 0.05, "group_4": 0.05 },
    # Agent 8: Calibration Result Evaluation in Astrophotography
    { "group_1": 0.20, "group_2": 0.15, "group_3": 0.10, "group_4": 0.05 },
    # Agent 9: High Contrast Region and Signal-to-Noise Ratio Analysis
    { "group_1": 0.20, "group_2": 0.15, "group_3": 0.10, "group_4": 0.05 },
    # Agent 10: Background Gradient and Color Block Analysis in Astronomical Images
    { "group_1": 0.10, "group_2": 0.10, "group_3": 0.05, "group_4": 0.05 },
    # Agent 11: Star Point Morphology Integrity in Post-processing
    { "group_1": 0.20, "group_2": 0.15, "group_3": 0.10, "group_4": 0.05 },
    # Agent 12: Signal-to-Noise Ratio Analysis and Feedback in Image Processing
    { "group_1": 0.20, "group_2": 0.15, "group_3": 0.10, "group_4": 0.05 },
    # Agent 13: Brightness, Contrast, and Saturation Evaluation in Final Image
    { "group_1": 0.10, "group_2": 0.10, "group_3": 0.05, "group_4": 0.05 },
]

original_full_graph_file = "YOUR_GRAPH.graphml"
max_processing_depth = 4

try:
    rag.run_hierarchical_subdivision(
        original_full_graph_path=original_full_graph_file,
        threshold_config_for_agents=threshold_config_for_agents,
        max_depth=max_processing_depth,
    )
except Exception as e:
    print(f"Exception: {e}")
    import traceback
    traceback.print_exc()