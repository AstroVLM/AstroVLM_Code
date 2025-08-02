from agent import AgentConfig
import os

API_KEY = os.getenv("LLM_API_KEY")
API_URL = os.getenv("LLM_BASE_URL")
predict_model = os.getenv("BASE_CHAT_MODEL")

Device = AgentConfig(
    name="Device Selection and Compatibility for Astronomical Imaging",
    model=predict_model,
    system_prompt="You are an expert in the analysis of astronomical imaging equipment, prioritizing the images themselves as the primary source of information for conducting evaluations and analyses.",
    user_prompt="It appears that there may be a potential issue regarding the compatibility of the equipment I used during the imaging process. Based on the information provided above, the relevant shooting details, and your own knowledge base, please analyze this image and conduct a rigorous assessment to determine whether such an issue exists. Rely solely on the image’s encoding data to evaluate whether the problem is present. If any incompatibility is detected, clearly specify the nature of the issue and provide a confidence level for its occurrence. If all equipment appears compatible, please conclude after analysis that no issue is present.",
    description="Device Agent",
    input_type=["text", "image"],
    api_type="openai",
    api_key=API_KEY,
    base_url=API_URL,
    sub_graph_path=r"knowledge_graph\G4_K1.graphml",
    working_dir=r"work_dir\Device Selection and Compatibility for Astronomical Imaging",
    with_reference=False
)

Shooting = AgentConfig(
    name="Shooting Time, Location, and Target Compatibility in Astronomy",
    model=predict_model,
    system_prompt="You are an expert in astrophotography analysis, prioritizing the image itself as the primary basis for conducting your assessment.",
    user_prompt="There may be potential issues regarding shooting time, location, and target compatibility in my imaging process. Based on the information provided above, the relevant shooting details, and your own knowledge base, please analyze this image and conduct a rigorous, high-standard examination to determine whether such issues exist. Rely solely on the image’s encoding data to make this assessment. If any issues are detected, specify them clearly and provide a confidence level for each. If all parameters are appropriate, conclude after analysis that there are no problems.",
    description="Shooting Time, Location, and Target Agent",
    input_type=["text", "image"],
    api_type="openai",
    api_key=API_KEY,
    base_url=API_URL,
    sub_graph_path=r"knowledge_graph\G4_K2.graphml",
    working_dir="work_dir\Shooting Time, Location, and Target Compatibility in Astronomy",
    with_reference=False
)

Exposure = AgentConfig(
    name="Exposure Time and Parameter Evaluation in Astrophotography",
    model=predict_model,
    system_prompt="You are an expert in astrophotography analysis, prioritizing the image itself as the primary basis for conducting your assessment.",
    user_prompt="There may be potential issues with overexposure or underexposure in my photographs. Based on the information provided above, the relevant shooting details, and your own knowledge base, please analyze this image and conduct the most rigorous, high-standard examination to determine whether any exposure-related problems are present. Make your assessment solely based on the image’s encoding data. If any issues are detected, clearly identify them and provide a confidence level for each. If all exposure parameters are appropriate, conclude after analysis that there are no problems.",
    description="Exposure Agent",
    input_type=["text", "image"],
    api_type="openai",
    api_key=API_KEY,
    base_url=API_URL,
    sub_graph_path=r"knowledge_graph\G4_K3.graphml",
    working_dir="work_dir\Exposure Time and Parameter Evaluation in Astrophotography",
    with_reference=False
)

Star_Point_Front = AgentConfig(
    name="Star Point Morphology and Tracking Accuracy in Astrophotography",
    model=predict_model,
    system_prompt="You are an expert in astrophotography analysis, prioritizing the image itself as the primary basis for conducting your assessment.",
    user_prompt="There may be potential issues with the shape of the star points in my photograph. Based on the information provided above, the relevant shooting details, and your own knowledge base, please analyze this image and conduct the most rigorous, high-standard examination to determine whether any problems exist with the star point shapes. Make your assessment solely on the basis of the image’s encoding data. If any issues are detected, clearly identify them and provide a confidence level for each. If all aspects are appropriate, conclude after analysis that there are no problems.",
    description="Star Point Agent Front",
    input_type=["text", "image"],
    api_type="openai",
    api_key=API_KEY,
    base_url=API_URL,
    sub_graph_path=r"knowledge_graph\G4_K4.graphml",
    working_dir="work_dir\Star Point Morphology and Tracking Accuracy in Astrophotography",
    with_reference=False
)

Halo = AgentConfig(
    name="Halo and Artifact Detection in Astronomical Images",
    model=predict_model,
    system_prompt="You are an expert in astrophotography analysis, prioritizing the image itself as the primary basis for conducting your assessment.",
    user_prompt="There may be potential issues with halo effects in the image I captured. Based on the information provided above, the relevant shooting details, and your own knowledge base, please analyze this image and conduct the most rigorous, high-standard examination to determine whether any halo-related problems are present. Make your assessment solely on the basis of the image’s encoding data. If any issues are detected, clearly identify them and provide a confidence level for each. If all parameters are appropriate, conclude after analysis that there are no problems.",
    description="Halo Agent",
    input_type=["text", "image"],
    api_type="openai",
    api_key=API_KEY,
    base_url=API_URL,
    sub_graph_path=r"knowledge_graph\G4_K5.graphml",
    working_dir="work_dir\Halo and Artifact Detection in Astronomical Images",
    with_reference=False
)

Calibration  = AgentConfig(
    name="Calibration Frame Effectiveness and Quality in Astronomy",
    model=predict_model,
    system_prompt="You are an expert in astrophotography analysis, prioritizing the image itself as the primary basis for conducting your assessment.",
    user_prompt="There may be potential issues related to vignetting or image degradation caused by the use of calibration frames during my imaging process. Based on the information provided above, the relevant shooting details, and your own knowledge base, please analyze this image and conduct the most rigorous, high-standard examination to determine whether any such issues are present. Make your assessment solely on the basis of the image’s encoding data. If any problems are detected, clearly identify them and provide a confidence level for each. If all parameters are appropriate, conclude after analysis that there are no problems.",
    description="Calibration Agent",
    input_type=["text", "image"],
    api_type="openai",
    api_key=API_KEY,
    base_url=API_URL,
    sub_graph_path=r"knowledge_graph\G4_K6.graphml",
    working_dir="work_dir\Calibration Frame Effectiveness and Quality in Astronomy",
    with_reference=False
)

Color  = AgentConfig(
    name="Color Balance and Ground Truth Comparison in Image Processing",
    model=predict_model,
    system_prompt="You are an expert in astrophotography analysis, prioritizing the image itself as the primary basis for conducting your assessment.",
    user_prompt="There may be potential color cast or color deviation issues in the processed version of this sample image. Based on the information provided above, the relevant shooting details, and your own knowledge base, please analyze this image—using standard RGB values as a reference—and conduct the most rigorous, high-standard examination to determine whether any color-related issues exist. Make your assessment solely on the basis of the image’s encoding data. If any issues are detected, clearly identify them and provide a confidence level for each. If all color parameters are appropriate, conclude after analysis that there are no problems.",
    description="RGB Agent",
    input_type=["text", "image"],
    api_type="openai",
    api_key=API_KEY,
    base_url=API_URL,
    sub_graph_path=r"knowledge_graph\G4_K7.graphml",
    working_dir="work_dir\Color Balance and Ground Truth Comparison in Image Processing",
    with_reference=False
)

Calibration_Result  = AgentConfig(
    name="Calibration Result Evaluation in Astrophotography",
    model=predict_model,
    system_prompt="You are an expert in astrophotography analysis, prioritizing the image itself as the primary basis for conducting your assessment.",
    user_prompt="The results after applying calibration frames may exhibit issues such as ineffective calibration, uneven background, and low signal-to-noise ratio. Based on the information provided above, the relevant shooting details, and your own knowledge base, please analyze this image and conduct the most rigorous, high-standard examination to determine whether any of these issues are present. Make your assessment solely on the basis of the image’s encoding data. If any problems are detected, clearly identify them and provide a confidence level for each. If all parameters are appropriate, conclude after analysis that there are no problems.",
    description="Calibration Result Agent",
    input_type=["text", "image"],
    api_type="openai",
    api_key=API_KEY,
    base_url=API_URL,
    sub_graph_path=r"knowledge_graph\G4_K8.graphml",
    working_dir="work_dir\Calibration Result Evaluation in Astrophotography",
    with_reference=False
)

Signal_to_Noise  = AgentConfig(
    name="High Contrast Region and Signal-to-Noise Ratio Analysisg",
    model=predict_model,
    system_prompt="You are an expert in astrophotography analysis, prioritizing the image itself as the primary basis for conducting your assessment.",
    user_prompt="There may be potential issues with high-contrast areas of the subject and the signal-to-noise ratio (SNR) in my photograph. Based on the information provided above, the relevant shooting details, and your own knowledge base, please analyze this image and conduct the most rigorous, high-standard examination to determine whether any SNR-related problems exist. Make your assessment solely on the basis of the image’s encoding data. If any issues are detected, clearly identify them and provide a confidence level for each. If all parameters are appropriate, conclude after analysis that there are no problems.",
    description="SNR Agent",
    input_type=["text", "image"],
    api_type="openai",
    api_key=API_KEY,
    base_url=API_URL,
    sub_graph_path=r"knowledge_graph\G4_K9.graphml",
    working_dir="work_dir\High Contrast Region and Signal-to-Noise Ratio Analysis",
    with_reference=False
)

Gradient  = AgentConfig(
    name="Background Gradient and Color Block Analysis in Astronomical Images",
    model=predict_model,
    system_prompt="You are an expert in astrophotography analysis, prioritizing the image itself as the primary basis for conducting your assessment.",
    user_prompt="The image I captured may have potential issues with background gradients and color blotches. Based on the information provided above, the relevant shooting details, and your own knowledge base, please analyze this image and conduct the most rigorous, high-standard examination to determine whether any of these issues are present. Make your assessment solely on the basis of the image’s encoding data. If any problems are detected, clearly identify them and provide a confidence level for each. If all parameters are appropriate, conclude after analysis that there are no problems.",
    description="BG and CB Agent",
    input_type=["text", "image"],
    api_type="openai",
    api_key=API_KEY,
    base_url=API_URL,
    sub_graph_path=r"knowledge_graph\G4_K10.graphml",
    working_dir="work_dir\Background Gradient and Color Block Analysis in Astronomical Images",
    with_reference=False
)

Star_Point_After  = AgentConfig(
    name="Star Point Morphology Integrity in Post-processing",
    model=predict_model,
    system_prompt="You are an expert in astrophotography analysis, prioritizing the image itself as the primary basis for conducting your assessment.",
    user_prompt="The star point shapes in the post-processed image I captured may have potential issues. Based on the information provided above, the relevant shooting details, and your own knowledge base, please analyze this image and conduct the most rigorous, high-standard examination to determine whether any of these issues are present. Make your assessment solely on the basis of the image’s encoding data. If any problems are detected, clearly identify them and provide a confidence level for each. If all parameters are appropriate, conclude after analysis that there are no problems.",
    description="Star Point After Agent",
    input_type=["text", "image"],
    api_type="openai",
    api_key=API_KEY,
    base_url=API_URL,
    sub_graph_path=r"knowledge_graph\G4_K11.graphml",
    working_dir="work_dir\Star Point Morphology Integrity in Post-processing",
    with_reference=False
)

SNR_After = AgentConfig(
    name="Signal-to-Noise Ratio Analysis and Feedback in Image Processing",
    model=predict_model,
    system_prompt="You are an expert in astrophotography analysis, prioritizing the image itself as the primary basis for conducting your assessment.",
    user_prompt="The signal-to-noise ratio in the post-processed image I captured may have potential issues. Based on the information provided above, the relevant shooting details, and your own knowledge base, please analyze this image and conduct the most rigorous, high-standard examination to determine whether any such issues are present. Make your assessment solely on the basis of the image’s encoding data. If any problems are detected, clearly identify them and provide a confidence level for each. If all parameters are appropriate, conclude after analysis that there are no problems.",
    description="Star Point After Agent",
    input_type=["text", "image"],
    api_type="openai",
    api_key=API_KEY,
    base_url=API_URL,
    sub_graph_path=r"knowledge_graph\G4_K12.graphml",
    working_dir="work_dir\Signal-to-Noise Ratio Analysis and Feedback in Image Processing",
    with_reference=False
)

BCS  = AgentConfig(
    name="Brightness, Contrast, and Saturation Evaluation in Final Image",
    model=predict_model,
    system_prompt="You are an expert in astrophotography analysis, prioritizing the image itself as the primary basis for conducting your assessment.",
    user_prompt="There may be potential issues with the brightness, contrast, and saturation of the subject in my photograph. Based on the information provided above, the relevant shooting details, and your own knowledge base, please analyze this image and conduct the most rigorous, high-standard examination to determine whether any problems exist with brightness, contrast, or saturation. Make your assessment solely on the basis of the image’s encoding data. If any issues are detected, clearly identify them and provide a confidence level for each. If all parameters are appropriate, conclude after analysis that there are no problems.",
    description="BCS Agent",
    input_type=["text", "image"],
    api_type="openai",
    api_key=API_KEY,
    base_url=API_URL,
    sub_graph_path=r"knowledge_graph\G4_K13.graphml",
    working_dir="work_dir/Brightness, Contrast, and Saturation Evaluation in Final Image",
    with_reference=False
)