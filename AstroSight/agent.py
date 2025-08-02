import sys
import os
import json
sys.path.append("./")
from typing import List, Dict, Union, Any
from pydantic import BaseModel
import asyncio
import base64
from openai import OpenAI
import openai
from ASK_RAG.ASK_RAG import (
    ASK_RAG,
    QueryParam
)
import re
API_KEY = os.getenv("LLM_API_KEY")
API_URL = os.getenv("LLM_BASE_URL")

class AgentConfig(BaseModel):
    name: str
    model: str
    description: str
    input_type: List[str]  # ["text", "image"]
    system_prompt: str
    user_prompt: str
    api_type: str
    api_key:str
    base_url:str
    sub_graph_path:Union[str,None]
    working_dir:Union[str,None]
    with_reference: bool

coordinator = AgentConfig(
    name="coordinator",
    model="deepseek-v3-241226",
    system_prompt="",
    user_prompt="Only summarize and extract the errors found in this shooting based on the content of the Agent’s response. Do not attempt to identify potential issues on your own, and do not evaluate the Agent. Instead, summarize the Agent’s analysis and place it into the \"Analysis\" field of the dictionary mentioned below.\
    My previous question was: “{question}”, and the Agent’s response was: {answer}. If errors are identified, determine which aspect(s) they fall under from the provided agent_names list: [{agent_names}].\
    Return the corresponding index number(s) from the list, where the numbering should follow Python indexing conventions (the first item in the list is numbered 0, the n-th item is numbered n-1). If no errors are found or if the agent_names list is empty, return -1 for the index.\
    Finally, write the entire content in the form of a JSON dictionary with five keys: \"General\", \"Extract Questions\", \"Analysis\", \"Whether Question Exist\", and “agent_index”. The value types for these keys should be str, str, str, bool, and list[int] respectively, with “agent_index” containing the list of returned index numbers.\
    ",
    description="coordinator",
    input_type=["text"],
    api_type="openai",
    api_key=API_KEY,
    base_url=API_URL,
    sub_graph_path=None,
    working_dir=None,
    with_reference=False
)

class Agent_Tree_Node:
    def __init__(self, value:list[str]):
        self.name = value[0]
        self.value = value[1]
        self.children = []
        self.parent = None  

    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)

    def __repr__(self):
        return f"TreeNode({self.value})"
    
class Agent_Tree:
    def __init__(self, root_value):
        self.root = Agent_Tree_Node(root_value)
        self.current = self.root  

    def set_current(self, node):
        self.current = node

    def get_parent(self):
        return self.current.parent
    
    def add_child_to_current(self, value):
        new_node = Agent_Tree_Node(value)
        self.current.add_child(new_node)
        return new_node

    def print_tree(self, node=None, level=0):
        if node is None:
            node = self.root
        print("  " * level + str(node.value))
        for child in node.children:
            self.print_tree(child, level + 1)
    
    def get_tree_string(self, node=None, level=0):
        if node is None:
            node = self.root

        result = str(node.value)
        for child in node.children:
            result += " " + self.get_tree_string(child, level + 1)
        return result

class AgentSystem:
    def __init__(self, config:AgentConfig):
        self.config = config
        self.RAG = ASK_RAG(working_dir=config.working_dir) if config.working_dir and config.sub_graph_path else None
        
        if self.RAG:
            if not len(os.listdir(self.RAG.working_dir)):
                self.RAG.load_and_store_graph(config.sub_graph_path)
                self.RAG.save_storage()
    
class MultiAgentSystem:
    def __init__(self):
        self.agents = {}
        self.coordinator_tree = Agent_Tree([coordinator.name, ""])
        self.prompt_usage = 0
        self.completion_usage = 0
        self.total_tokens_usage = 0
    def add_agent(self, agent:AgentSystem):
        self.agents[agent.config.name] = agent

    def clear_token_usage(self):
        self.prompt_usage = 0
        self.completion_usage = 0
        self.total_tokens_usage = 0

    def process_input(self, input_data: Dict[str, Any]):
        processed = {}
        if 'text' in input_data:
            processed['text'] = input_data['text']
        if 'image' in input_data:
            processed['image'] = self._encode_image(input_data['image'])
        return processed

    def run_agent(self, agent_name: str, input_data: Dict, front_names:list):
        agent:AgentSystem = self.agents[agent_name]
        if agent.config.api_type.startswith("openai"):
            result = self._call_openai(agent, input_data)
            val = self.coordinate(result, front_names)
            return (result, val)
        

    def coordinate(self, result: Dict, names:list):
        temp = self.agents["coordinator"].config.user_prompt
        self.agents["coordinator"].config.user_prompt = self._build_coordinator_prompt(self.agents["coordinator"].config, result, names)
        if self.agents["coordinator"].config.api_type.startswith("openai"):
            res = self._call_openai(self.agents["coordinator"], None)
            self.agents["coordinator"].config.user_prompt = temp
            return res

    # def accuracy(self, agent_answer:str, expert_answer: str):
    #     temp = self.agents["accuracy"].config.user_prompt
    #     self.agents["accuracy"].config.user_prompt = self._build_accuracy_prompt(self.agents["accuracy"].config, agent_answer, expert_answer)
    #     if self.agents["accuracy"].config.api_type.startswith("openai"):
    #         res = self._call_openai(self.agents["accuracy"], None)
    #         self.agents["accuracy"].config.user_prompt = temp
    #         return res
        
    # def Reasoning_Rationality(self, agent_answer:str):
    #     temp = self.agents["Reasoning Rationality"].config.user_prompt
    #     self.agents["Reasoning Rationality"].config.user_prompt = self._build_rr_prompt(self.agents["Reasoning Rationality"].config, agent_answer)
    #     if self.agents["Reasoning Rationality"].config.api_type.startswith("openai"):
    #         res = self._call_openai(self.agents["Reasoning Rationality"], None)
    #         self.agents["Reasoning Rationality"].config.user_prompt = temp
    #         return res
    
    # def Diversity(self, agent_answer:str):
    #     temp = self.agents["Diversity"].config.user_prompt
    #     self.agents["Diversity"].config.user_prompt = self._build_diversity_prompt(self.agents["Diversity"].config, agent_answer)
    #     if self.agents["Diversity"].config.api_type.startswith("openai"):
    #         res = self._call_openai(self.agents["Diversity"], None)
    #         self.agents["Diversity"].config.user_prompt = temp
    #         return res
    
    def execute(self, input_data: Dict, agent_names: List[str]):
        processed_input = self.process_input(input_data)
        results = []
        for index,name in enumerate(agent_names):
            res = self.run_agent(name, processed_input, agent_names[:index])
            match = re.search(r'{.*}', res[1]["content"], re.DOTALL)
            if match:
                coordinator_content = json.loads(match.group())
            else:
                raise ValueError("ERR JSON")
            
            child_node = Agent_Tree_Node([name, coordinator_content["General"]])
            self.coordinator_tree.current.add_child(child_node)
            results.append(res)

        return results
    def execute_tree(self, input_data: Dict, agent_names: List[str]):
        processed_input = self.process_input(input_data)
        results = []
        for index,name in enumerate(agent_names):
            res = self.run_agent(name, processed_input, agent_names[:index])
            match = re.search(r'{.*}', res[1]["content"], re.DOTALL)
            if match:
                coordinator_content = json.loads(match.group())
            else:
                raise ValueError("ERR JSON")
            if coordinator_content["Whether Question Exist"]:
                child_node = Agent_Tree_Node([name, coordinator_content["Analysis"]])
                self.coordinator_tree.current.add_child(child_node)
                new_input_data = input_data
                new_input_data["text"] += "Find this Problem in this picture:" + coordinator_content["Extract Questions"]
                new_agent_names = []
                for i in coordinator_content["agent_index"]:
                    if i==-1:
                        continue
                    else:
                        print(i, agent_names)
                        new_agent_names.append(agent_names[i])
                self.coordinator_tree.current = child_node
                self.execute_tree(new_input_data, new_agent_names)
                self.coordinator_tree.current = self.coordinator_tree.get_parent()
            results.append(res)

        return results

    def _build_coordinator_prompt(self, config: AgentConfig, result: Dict, names:list[str]):
        names_string = ", ".join(names)
        text_dict = {
            "question": result.get("origin_messages"),
            "answer":result.get("content"),
            "agent_names":names_string
            }
        return config.user_prompt.format(**text_dict)
    
    def _build_accuracy_prompt(self, config: AgentConfig, agent_answer:str, expert_answer: str):
        text_dict = {
            "agent_answer": agent_answer,
            "expert_answer": expert_answer
            }
        return config.user_prompt.format(**text_dict)
    
    def _build_rr_prompt(self, config: AgentConfig, agent_answer:str):
        text_dict = {
            "agent_answer": agent_answer,
            }
        return config.user_prompt.format(**text_dict)
    
    def _build_diversity_prompt(self, config: AgentConfig, agent_answer:str):
        text_dict = {
            "agent_answer": agent_answer,
            }
        return config.user_prompt.format(**text_dict)

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _call_openai(self, agent: AgentSystem, input_data):
        config:AgentConfig = agent.config
        try:
            max_tokens = 1000  
            chat_client = OpenAI(api_key=config.api_key, base_url=config.base_url)
            if agent.RAG is None:
                messages = [
                    {"role":"system", "content": [{"type": "text", "text": config.system_prompt}]} ,
                    {"role":"user", "content": [{"type": "text", "text": config.user_prompt}]}
                ]
            else:
                messages = agent.RAG.ask(config.with_reference, config.user_prompt + input_data['text'], config.system_prompt)
            if input_data:
                if 'text' in input_data and input_data['text']:
                    messages[1]["content"].append({
                        "type": "text",
                        "text": input_data['text']})
                    
                if 'image' in input_data and input_data['image']:
                    messages[1]["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{input_data['image']}",
                            "detail": "high"  # low/high
                        }
                    })

            response = chat_client.chat.completions.create(
                model=config.model,
                messages=messages,
                max_tokens=max_tokens
            )
            usage = dict(response.usage)
            self.prompt_usage += usage['prompt_tokens']
            self.completion_usage += usage['completion_tokens']
            self.total_tokens_usage = self.prompt_usage + self.completion_usage

            return {
                "agent": config.name,
                "content": response.choices[0].message.content,
                "model": config.model,
                "usage": usage,
                "origin_messages":config.user_prompt
            }
        except openai.APIError as e:
            print(f"OpenAI API Error: {e}")
            return {"error": str(e)}
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {"error": "Service unavailable"}


