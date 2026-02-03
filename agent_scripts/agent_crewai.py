import os
import pandas as pd
import inspect
from typing import Any
import statool
from crewai import Crew, Task, Agent, LLM, Process
from crewai_tools import FileReadTool, CodeInterpreterTool
from crewai.tools import tool
import traceback
import json
from tqdm import tqdm
from utils import get_mcp_prompt, get_model_cfg

def auto_wrap_tools(module):
    wrapped_tools = []
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_") or inspect.getmodule(func) != module:
            continue

        sig = inspect.signature(func)
        params_str_list = []
        for param_name, param in sig.parameters.items():
            param_type = param.annotation
            if param_type not in [int, float, str, bool]:
                param_type = Any
            params_str_list.append(f"{param_name}: {param_type.__name__}")
        params_str = ", ".join(params_str_list)

        wrapper_code = f"""
@tool("{func.__name__}")
def wrapper({params_str}):
    r\"\"\"{func.__doc__}\"\"\"
    return func({', '.join(sig.parameters.keys())})
"""
        local_ns = {"func": func, "tool": tool, "Any": Any}
        exec(wrapper_code, local_ns)
        wrapped_tools.append(local_ns["wrapper"])
    return wrapped_tools

if __name__ == "__main__":

    # ================= Configuration Area =================
    models = ["deepseek"]
    model_map = {"deepseek": "deepseek3"}
    
    # Filter Configuration
    needs = ["a21"]  # Only run these indices
    begin_index = 0
    
    # Path Configuration
    input_path = "./agents_allq/crewai.json"
    output_path = "./agents_allq/crewai.json"
    
    # Tool Initialization
    tools = auto_wrap_tools(statool)
    code_interpreter = CodeInterpreterTool(unsafe_mode=True)

    for model in models:
        print(f"Using model: {model}")
        
        cfg = get_model_cfg(model, model_map)

        if model == "deepseek":
            llm = LLM(
                model=f"dashscope/{cfg['model']}",
                base_url=cfg["base_url"],
                api_key=cfg["api_key"]
            )
        else:
            llm = LLM(
                model=f"dashscope/{cfg['model']}" if "qwen" in cfg['model'].lower() else cfg['model'],
                base_url=cfg["base_url"],
                api_key=cfg["api_key"]
            )

        data = pd.read_json(input_path, encoding='utf-8')
        data_tmp = data.copy()

        col = f"crewai_{model}"
        if col not in data_tmp.columns:
            data_tmp[col] = None

        # Logic preserved as original to check for resume index
        null_mask = data_tmp[col].isna()
        if null_mask.any():
            resume_index = null_mask.idxmax()
            print(f"[Info] Resume from index {resume_index}")
        else:
            print("[Info] All rows already finished.")
            # begin_index = len(data_tmp) # Preserved commented out line

        # Loop processing
        for index, row in tqdm(data.iterrows(), total=len(data)):
            
            # Filter logic
            if index < begin_index or row["index"] not in needs:
                continue
            
            print("-" * 20 + f"agent{model},row{index}" + "-" * 20)
            
            try:
                # Create Agent
                dataset_inference_agent = Agent(
                    role="Statistics Expert",
                    goal=(
                        "Answer user's questions, and you can use tools if necessary."
                    ),
                    backstory=(
                        "There are lots of tools you can use to help you answer questions about datasets."
                    ),
                    tools=tools,
                    llm=llm,
                    verbose=False,
                    max_iter=5
                )

                query = get_mcp_prompt(row)
                coding_task = Task(
                    description=query,
                    expected_output="Use the tools' results to answer the question",
                    agent=dataset_inference_agent,
                )

                # Create and run Crew
                crew = Crew(
                    agents=[dataset_inference_agent],
                    tasks=[coding_task],
                    verbose=True,
                    process=Process.sequential
                )
                result = crew.kickoff()

                # Safely retrieve result
                data_tmp.loc[index, col] = getattr(result, "raw", None)

            except Exception:
                print(f"[Error] Row {index} failed:\n{traceback.format_exc()}")
                message = f"[Error] Row {index} failed:\n{traceback.format_exc()}"
                data_tmp.loc[index, col] = message

            # Save progress every 10 rows
            if (index + 1) % 10 == 0:
                data_tmp.to_json(output_path, force_ascii=False, orient='records', indent=2)
                print(f"[Info] Saved progress at index {index + 1}")

        # Save final result
        data_tmp.to_json(output_path, force_ascii=False, orient='records', indent=2)
        print("[Info] Finished processing all rows and saved final result.")