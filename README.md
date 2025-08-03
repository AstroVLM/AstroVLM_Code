<p align="center">
  <img src="https://github.com/user-attachments/assets/1db1baa0-62b4-4469-b92b-a8083cf39796" width="345px" style="vertical-align:middle;">
</p>

## Project Introduction
AstroVLM is an expert multi-agent collaborative framework designed for diagnosing the quality of astronomical images. This system addresses the complex challenges inherent in astronomical imaging, a field that requires a blend of multidisciplinary knowledge and involves numerous, intricate sub-tasks. By leveraging a multi-agent system, AstroVLM provides a robust solution for comprehensive quality diagnosis and precise error localization in the astronomical imaging workflow. 

## Key Features

- Multi-Agent Framework designed for Astronomical imaging diagnosis. **(AstroSight)**
- Agent-Specific Knowledge RAG **(ASK-RAG)** for multi-agent system. 
- Reasonging with Backtracking **(RwB)** for enhancing the accuracy of error localisation:  
  - Chain-of-Backtracking **(CoB)**
  - Collaborative Reasoning Tree **(CRT)**
-  Superior performance and proven stability in experiments.
-  Sufficient dataset collected from real-world observation.

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

## Run
Run the ASKRAG to get subgraphs for all agents.
```bash
python ASK_Main.py
```

Run the AstroSight to get use of AstroVLM.
```bash
python AstroSight/multi-process.py
```

## PS

The current repository only contains some key codes. After the paper is accepted, all the codes will be opensourced in this repository.
