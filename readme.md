<p align="center">
  <img src="https://github.com/user-attachments/assets/1db1baa0-62b4-4469-b92b-a8083cf39796" width="345px" style="vertical-align:middle;">
</p>

## Project Introduction
AstroVLM is an expert multi-agent collaborative framework designed for diagnosing the quality of astronomical images. This system addresses the complex challenges inherent in astronomical imaging, a field that requires a blend of multidisciplinary knowledge and involves numerous, intricate sub-tasks. Even for world-class organizations and seasoned enthusiasts, diagnosing image quality is a significant investment of time and effort due to the complex correlations between different processes. 

The goal of AstroVLM is to automate the difficult and time-consuming task of identifying the root causes of low-quality astronomical images, a challenge that has not been effectively addressed by previous studies. By leveraging a multi-agent system, AstroVLM provides a robust solution for comprehensive quality diagnosis and precise error localization in the astronomical imaging workflow. 



## Key Features

- Multi-Agent Framework (AstroSight)
- Agent-Specific Knowledge RAG (ASK-RAG)
- Reasonging with Backtracking (RwB): This novel process enhances the accuracy of error localization.  It features:
  - Chain-of-Backtracking (CoB)
  - Collaborative Reasoning Tree (CRT)
-  Superior Performance
-  Proven Stability
-  Lastest Novel RAG Method

## Installation 
1.  Clone the repo:
```bash
git clone https://github.com/AstroVLM/AstroVLM_Code.git
```
2. Create a new environment to install the libraries and activate it:
```bash
conda create -n your_environment python=3.12
conda activate your_environment
```
3. Install the required libraries.
```bash
pip install -r requirements.txt
```
4. Set your API key:
```bash
export LLM_API_KEY="YOUR_API_KEY"
```

## Deployment
Run the ASK_RAG to get subgraphs for all agents
```bash
python ASK_Test.py
```

## Run
Run the Astrosight to get use of AstroVLM
```bash
python AstroSight/multi-process.py
```