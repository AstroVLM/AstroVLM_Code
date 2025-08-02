import os
import glob
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from agent import MultiAgentSystem, AgentConfig, AgentSystem, coordinator, accuracy
from agent_dict import *
load_dotenv()
import multiprocessing as mp
from multiprocessing import Queue, Event
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

agent_names=[
                "Device Selection and Compatibility for Astronomical Imaging",
                "Shooting Time, Location, and Target Compatibility in Astronomy",
                "Exposure Time and Parameter Evaluation in Astrophotography",
                "Star Point Morphology and Tracking Accuracy in Astrophotography",
                "Halo and Artifact Detection in Astronomical Images",
                "Calibration Frame Effectiveness and Quality in Astronomy",
                "Color Balance and Ground Truth Comparison in Image Processing",
                "Calibration Result Evaluation in Astrophotography",
                "High Contrast Region and Signal-to-Noise Ratio Analysisg",
                "Background Gradient and Color Block Analysis in Astronomical Images",
                "Star Point Morphology Integrity in Post-processing",
                "Signal-to-Noise Ratio Analysis and Feedback in Image Processing",
                "Brightness, Contrast, and Saturation Evaluation in Final Image"
                ]

def system_add_agent(system:MultiAgentSystem):

    # Multi-Agent
    system.add_agent(AgentSystem(Device))
    system.add_agent(AgentSystem(Shooting))
    system.add_agent(AgentSystem(Exposure))
    system.add_agent(AgentSystem(Star_Point_Front))
    system.add_agent(AgentSystem(Halo))
    system.add_agent(AgentSystem(Calibration))
    system.add_agent(AgentSystem(Color))
    system.add_agent(AgentSystem(Calibration_Result))
    system.add_agent(AgentSystem(Signal_to_Noise))
    system.add_agent(AgentSystem(Gradient))
    system.add_agent(AgentSystem(Star_Point_After))
    system.add_agent(AgentSystem(SNR_After))
    system.add_agent(AgentSystem(BCS))

    # coordinator
    system.add_agent(AgentSystem(coordinator))


def summary_analysis(results:list):
    analysis = ""
    for res in results:
        match = re.search(r'{.*}', res[1]["content"], re.DOTALL)
        if match:
            coordinator_content = json.loads(match.group())
        else:
            raise ValueError("ERR JSON")
        analysis += coordinator_content["Analysis"]
    return analysis

def extract_accuracy(result):

    match = re.search(r'{.*}', result["content"], re.DOTALL)
    if match:
        r = json.loads(match.group())
    else:
        raise ValueError("ERR JSON")
    acc=r['acc']
    return acc

def write_list_to_xlsx(list_data, list_name: str):
    filename = f"{list_name}.xlsx"
    df = pd.DataFrame([list_data])
    df.to_excel(filename, index=False, header=False)

def write_2dlist_to_xlsx(list_data, list_name: str):
    filename = f"{list_name}.xlsx"
    df = pd.DataFrame(list_data)
    df.to_excel(filename, index=False, header=False)



def process_run_agent(task_queue:Queue, answer_queue:Queue, multi_system:MultiAgentSystem, copy_names):
    
    if not task_queue.empty():

        # step 1: Initial an image
        multi_system.clear_token_usage()
        input_dict = task_queue.get()
        input_data = {
            "text":input_dict['input_text'],
            "image":input_dict['path']
            }

        # step 2: Run AstroVLM
        try:
            result = multi_system.execute_tree(input_data, copy_names)
            tree_string = multi_system.coordinator_tree.get_tree_string()
            if tree_string == "":
                agent_answer = summary_analysis(results=result)
            else:
                agent_answer = tree_string
            
        
            expert_answer = input_dict['expert_answer']

            # Analysis agent answer
            # acc_result = multi_system.accuracy(agent_answer, expert_answer)
            # acc = extract_accuracy(acc_result)

        # step 3: Write Answer
            output_data = [input_dict['path'], agent_answer, expert_answer]
            answer_queue.put(output_data)
        except:
            task_queue.put(input_dict)


def system_1(task_queue:Queue, answer_queue:Queue, stop_event):

    multi_system = MultiAgentSystem()

    system_add_agent(multi_system)
    copy_names = agent_names
    while not stop_event.is_set():
        process_run_agent(task_queue, answer_queue, multi_system, copy_names)



def system_2(task_queue:Queue, answer_queue:Queue, stop_event):

    multi_system = MultiAgentSystem()

    system_add_agent(multi_system)
    copy_names = agent_names
    while not stop_event.is_set():
        process_run_agent(task_queue, answer_queue, multi_system, copy_names)

def system_3(task_queue:Queue, answer_queue:Queue, stop_event):

    multi_system = MultiAgentSystem()

    system_add_agent(multi_system)
    copy_names = agent_names
    while not stop_event.is_set():
        process_run_agent(task_queue, answer_queue, multi_system, copy_names)

def system_4(task_queue:Queue, answer_queue:Queue, stop_event):

    multi_system = MultiAgentSystem()

    system_add_agent(multi_system)
    copy_names = agent_names
    while not stop_event.is_set():
        process_run_agent(task_queue, answer_queue, multi_system, copy_names)

def system_5(task_queue:Queue, answer_queue:Queue, stop_event):

    multi_system = MultiAgentSystem()

    system_add_agent(multi_system)
    copy_names = agent_names
    while not stop_event.is_set():
        process_run_agent(task_queue, answer_queue, multi_system, copy_names)

def system_6(task_queue:Queue, answer_queue:Queue, stop_event):

    multi_system = MultiAgentSystem()

    system_add_agent(multi_system)
    copy_names = agent_names
    while not stop_event.is_set():
        process_run_agent(task_queue, answer_queue, multi_system, copy_names)

def system_7(task_queue:Queue, answer_queue:Queue, stop_event):

    multi_system = MultiAgentSystem()

    system_add_agent(multi_system)
    copy_names = agent_names
    while not stop_event.is_set():
        process_run_agent(task_queue, answer_queue, multi_system, copy_names)

def system_8(task_queue:Queue, answer_queue:Queue, stop_event):

    multi_system = MultiAgentSystem()

    system_add_agent(multi_system)
    copy_names = agent_names
    while not stop_event.is_set():
        process_run_agent(task_queue, answer_queue, multi_system, copy_names)

def system_9(task_queue:Queue, answer_queue:Queue, stop_event):

    multi_system = MultiAgentSystem()

    system_add_agent(multi_system)
    copy_names = agent_names
    while not stop_event.is_set():
        process_run_agent(task_queue, answer_queue, multi_system, copy_names)

def system_10(task_queue:Queue, answer_queue:Queue, stop_event):

    multi_system = MultiAgentSystem()

    system_add_agent(multi_system)
    copy_names = agent_names
    while not stop_event.is_set():
        process_run_agent(task_queue, answer_queue, multi_system, copy_names)

def system_11(task_queue:Queue, answer_queue:Queue, stop_event):

    multi_system = MultiAgentSystem()

    system_add_agent(multi_system)
    copy_names = agent_names
    while not stop_event.is_set():
        process_run_agent(task_queue, answer_queue, multi_system, copy_names)

def system_12(task_queue:Queue, answer_queue:Queue, stop_event):

    multi_system = MultiAgentSystem()

    system_add_agent(multi_system)
    copy_names = agent_names
    while not stop_event.is_set():
        process_run_agent(task_queue, answer_queue, multi_system, copy_names)

def system_13(task_queue:Queue, answer_queue:Queue, stop_event):

    multi_system = MultiAgentSystem()

    system_add_agent(multi_system)
    copy_names = agent_names
    while not stop_event.is_set():
        process_run_agent(task_queue, answer_queue, multi_system, copy_names)

def system_14(task_queue:Queue, answer_queue:Queue, stop_event):

    multi_system = MultiAgentSystem()

    system_add_agent(multi_system)
    copy_names = agent_names
    while not stop_event.is_set():
        process_run_agent(task_queue, answer_queue, multi_system, copy_names)

def system_15(task_queue:Queue, answer_queue:Queue, stop_event):

    multi_system = MultiAgentSystem()

    system_add_agent(multi_system)
    copy_names = agent_names
    while not stop_event.is_set():
        process_run_agent(task_queue, answer_queue, multi_system, copy_names)

def system_16(task_queue:Queue, answer_queue:Queue, stop_event):

    multi_system = MultiAgentSystem()

    system_add_agent(multi_system)
    copy_names = agent_names
    while not stop_event.is_set():
        process_run_agent(task_queue, answer_queue, multi_system, copy_names)

def master(task_queue:Queue, answer_queue:Queue, stop_event):
 
    answers = []
    new_files = []
    files = glob.glob("dataset/*.json")

    # step 1: Load untreated image
    for json_file in files:
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        txt_path = os.path.join("dataset", f"{base_name}.xlsx")

        if os.path.exists(txt_path):
            continue

    # step 2: Put image and its information into task process
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            new_files.append(json_file)
            task_queue.put(data)
            
    pbar = tqdm(total=len(new_files), bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')

    # step 3: Wait answer process 
    while len(answers) < len(new_files):
        if not answer_queue.empty():
            ans = answer_queue.get()
            # print(ans)
            write_list_to_xlsx(ans, ans[0][:-4])
            answers.append(ans)
            pbar.update(1)

    # step 4: Close multiprocess
    pbar.close
 
    stop_event.set()
    
    # step 5: Write answer
    write_2dlist_to_xlsx(answers, "dataset/results")

if __name__ == "__main__":

    task_queue = Queue(maxsize=200)
    answer_queue = Queue(maxsize=1000)
    stop_event = Event()

    # create process
    master_process = mp.Process(target=master, args=(task_queue, answer_queue, stop_event))
    s1 = mp.Process(target=system_1, args=(task_queue, answer_queue, stop_event,))
    s2 = mp.Process(target=system_2, args=(task_queue, answer_queue, stop_event,))
    s3 = mp.Process(target=system_3, args=(task_queue, answer_queue, stop_event,))
    s4 = mp.Process(target=system_4, args=(task_queue, answer_queue, stop_event,))
    s5 = mp.Process(target=system_5, args=(task_queue, answer_queue, stop_event,))
    s6 = mp.Process(target=system_6, args=(task_queue, answer_queue, stop_event,))
    s7 = mp.Process(target=system_7, args=(task_queue, answer_queue, stop_event,))
    s8 = mp.Process(target=system_8, args=(task_queue, answer_queue, stop_event,))
    # s9 = mp.Process(target=system_9, args=(task_queue, answer_queue, stop_event,))
    # s10 = mp.Process(target=system_10, args=(task_queue, answer_queue, stop_event,))
    # s11 = mp.Process(target=system_11, args=(task_queue, answer_queue, stop_event,))
    # s12 = mp.Process(target=system_12, args=(task_queue, answer_queue, stop_event,))
    # s13 = mp.Process(target=system_13, args=(task_queue, answer_queue, stop_event,))
    # s14 = mp.Process(target=system_14, args=(task_queue, answer_queue, stop_event,))
    # s15 = mp.Process(target=system_15, args=(task_queue, answer_queue, stop_event,))
    # s16 = mp.Process(target=system_16, args=(task_queue, answer_queue, stop_event,))
    


    # start process
    master_process.start()
    s1.start()
    s2.start()
    s3.start()
    s4.start()
    s5.start()
    s6.start()
    s7.start()
    s8.start()
    # s9.start()
    # s10.start()
    # s11.start()
    # s12.start()
    # s13.start()
    # s14.start()
    # s15.start()
    # s16.start()
    

    # end process
    master_process.join()
    s1.join()
    s2.join()
    s3.join()
    s4.join()
    s5.join()
    s6.join()
    s7.join()
    s8.join()
    # s9.join()
    # s10.join()
    # s11.join()
    # s12.join()
    # s13.join()
    # s14.join()
    # s15.join()
    # s16.join()