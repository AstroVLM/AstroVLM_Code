PROMPTS = {}

PROMPTS["keywords_extraction"] = """
**Role**
You are a helpful assistant tasked with identifying keywords related to a given **topic** and organizing them into progressively more specific **conceptual categories**. These categories should range from general concepts to more detailed and focused sub-domains, suitable for use in a system of agents designed for complex tasks, particularly for knowledge graph organization and querying.

**Goal**
For a given **topic**, you need to extract **4 groups** of keywords. The keywords should be organized in a way that the first group contains general, overarching **concepts or domains**, and each subsequent group becomes progressively more specific by drilling down into finer **conceptual details or sub-topics** related to the **topic**.

**Instructions**
- Return the keywords in **JSON format**.
- The output should consist of **4 groups of keywords**:
  - The first group contains the most general, high-level **conceptual keywords or domain areas**.
  - Each subsequent group provides more specific keywords, offering progressively more focused **conceptual sub-categories or aspects** related to the **topic**.
- Each group should represent a logical step toward more precise and focused **conceptual terms**.
- Ensure that the keywords are relevant to the **topic** and reflect its different conceptual aspects (e.g., general technologies as concepts, types of parameters, classes of methods, theoretical principles, defining characteristics, etc.).
***- Crucially, all keywords, even in the more specific groups (group_3, group_4), must remain conceptual and general in nature. Think of them as 'chapter headings' or key 'index terms' for a knowledge domain. They should represent categories, types, principles, methods, processes, properties, or abstract parameters rather than specific, concrete individual items, named products, particular measurements, or singular instances. This is vital as these keywords will be used for embedding and querying a knowledge graph to identify related conceptual nodes.***
- All keyword strings within the JSON values should be human-readable and in the same language as the input **topic**.
- The provided examples below demonstrate this conceptual hierarchy well (e.g., "FWHM of star" as a type of metric, not a specific value; "CCD sensor" as a type of sensor, not a particular model).
- Example structure for the output:

{{
  "group_1": ["general_concept1", "general_concept2", "general_concept3"],
  "group_2": ["more_specific_concept_category1", "more_specific_concept_category2", "more_specific_concept_category3"],
  "group_3": ["detailed_conceptual_area1", "detailed_conceptual_area2", "detailed_conceptual_area3"],
  "group_4": ["highly_focused_sub_domain1", "highly_focused_sub_domain2"]
}}


**Examples**

Example 1:
**Topic:** "Astronomical Imaging"
**Output:**
{{
  "group_1": ["Astronomy", "Imaging Principles", "Space Observation Techniques"],
  "group_2": ["Telescope Systems", "Imaging Sensor Technology", "Focal Parameters", "Camera Systems", "Mounting and Tracking Concepts"],
  "group_3": ["CCD/CMOS Sensor Types", "Pixel Characteristics", "Exposure Control Methods", "Optical System Parameters"],
  "group_4": ["Stellar Profile Analysis (e.g., FWHM)", "Guiding Accuracy Principles", "Sensor Noise Characterization"]
}}

Example 2:
**Topic:** "Device Selection for Photography"
**Output:**
{{
  "group_1": ["Photography Equipment Categories", "Lens Principles", "Camera Technology Overview"],
  "group_2": ["Camera Body Types (e.g., DSLR, Mirrorless)", "Sensor Format Concepts", "Lens Type Categories (e.g., Prime, Zoom)", "Focal Length Principles"],
  "group_3": ["Image Sensor Technologies (e.g., Full-frame, Crop)", "Lens Optical Design Concepts", "Autofocus System Principles"],
  "group_4": ["Aperture Control Mechanisms", "Lens Aberration Types", "Image Stabilization Technologies"]
}}

**Real Data**
**Topic:** {topic}
**Output:**
"""

topics = [
    "Device Selection and Compatibility for Astronomical Imaging",
    "Shooting Time, Location, and Target Compatibility in Astronomy",
    "Exposure Time and Parameter Evaluation in Astrophotography",
    "Star Point Morphology and Tracking Accuracy in Astrophotography",
    "Halo and Artifact Detection in Astronomical Images",
    "Calibration Frame Effectiveness and Quality in Astronomy",
    "Color Balance and Ground Truth Comparison in Image Processing",
    "Calibration Result Evaluation in Astrophotography",
    "High Contrast Region and Signal-to-Noise Ratio Analysis",
    "Background Gradient and Color Block Analysis in Astronomical Images",
    "Star Point Morphology Integrity in Post-processing",
    "Signal-to-Noise Ratio Analysis and Feedback in Image Processing",
    "Brightness, Contrast, and Saturation Evaluation in Final Image"
]

PROMPTS["edge_prediction"] = """
You are an expert in knowledge graph construction for astrophotography.
Based on the information for the two nodes below, please generate a single, concise sentence that describes a plausible relationship between them. This sentence will be used as the edge description in a knowledge graph.

Node 1:
{node1_info_str}

Node 2:
{node2_info_str}

Generated Relationship Sentence:
"""