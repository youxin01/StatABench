import os
import inspect
import functools
import numpy as np
import pandas as pd
import datetime
import decimal
from typing import Any, List, Union
import asyncio
import traceback
import statool
import json
from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_mcp_prompt,get_model_cfg
import argparse

def _clean_result(obj):
    if obj is None:
        return None
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, datetime.timedelta):
        return obj.total_seconds()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _clean_result(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_result(v) for v in obj]
    return obj

def make_function_tool(func):
    sig = inspect.signature(func)
    
    @functools.wraps(func)
    def wrapper(**kwargs):
        result = func(**kwargs)
        return _clean_result(result)
    new_annotations = {}
    new_params = []
    DATA_TYPE = Union[str, List] 

    for param in sig.parameters.values():
        if param.name == "data":
            new_annotations[param.name] = DATA_TYPE
            new_params.append(param.replace(annotation=DATA_TYPE))
        else:
            new_annotations[param.name] = Any
            new_params.append(param.replace(annotation=Any))
    
    wrapper.__annotations__ = new_annotations
    wrapper.__signature__ = sig.replace(parameters=new_params)
    
    return FunctionTool(wrapper, name=func.__name__, description=func.__doc__ or func.__name__)

def make_tools_from_module(module):
    tools = []
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if inspect.getmodule(func) != module:
            continue
        tools.append(make_function_tool(func))
    return tools

sys="""
You are a precise and careful Statistics Analyst. And you can use the tools provided to help you answer questions.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run autogen processing.")

    parser.add_argument("--models", nargs="+", default=["deepseek"],
                        help="List of models to use, e.g., --models deepseek model2")
    parser.add_argument("--begin_index", type=int, default=0,
                        help="Starting index")
    parser.add_argument("--input_path", type=str, default="./autogen2.json",
                        help="Input JSON file path")
    parser.add_argument("--output_path", type=str, default="./autogen2.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    models = args.models
    input_path = args.input_path 
    output_path = args.output_path
    begin_index =args.begin_index

    tools = make_tools_from_module(statool)
    
    for model in models:
        
        data = pd.read_json(input_path, encoding='utf-8')
        data_tmp = data.copy()
        col = f"autogen_{model}"
        if col not in data_tmp.columns:
            data_tmp[col] = None

        cfg = get_model_cfg(model)
        model_info = {
            "name": cfg["model"],
            "family": "qwen" if "qwen" in cfg["model"].lower() else "openai",
            "functions": [], 
            "vision": False,  
            "json_output": True, 
            "function_calling": True
        }

        for index, row in tqdm(data.iterrows(), total=len(data)):
            if index < begin_index:
                continue
            print("-"*20+f"agent{model},row{index}"+"-"*20)

            model_client = OpenAIChatCompletionClient(
                model=cfg["model"],
                api_key=cfg["api_key"],
                base_url=cfg["base_url"],
                model_info=model_info,
                llm_cfg = {
                    "max_retry": 5
                }
            )     
            agent = AssistantAgent(
                "assistant",
                model_client,
                tools=tools,
                reflect_on_tool_use=True,
                system_message=sys,
                max_tool_iterations=5
            )

            try:
                query = get_mcp_prompt(row)
                result = asyncio.run(agent.run(task=query))
                print(result)
                content = None
                if hasattr(result, "messages") and result.messages:
                    last_msg = result.messages[-1]
                    if hasattr(last_msg, "content"):
                        content = last_msg.content

                data_tmp.loc[index, f"autogen_{model}"] = content

            except Exception:
                print(f"[Error] Row {index} failed:\n{traceback.format_exc()}")
                message = f"[Error] Row {index} failed:\n{traceback.format_exc()}"
                data_tmp.loc[index, f"autogen_{model}"] = message

            if (index + 1) % 10 == 0:
                data_tmp.to_json(output_path, force_ascii=False, orient='records', indent=2)
                print(f"[Info] Saved progress at index {index + 1}")

        data_tmp.to_json(output_path, force_ascii=False, orient='records', indent=2)
        print("[Info] Finished processing all rows and saved final result.")