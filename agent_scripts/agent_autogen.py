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
import allfuncscode
import json
from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from tqdm import tqdm

def _clean_result(obj):
    """递归清洗返回值，保证 JSON 可序列化"""
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
    # 获取原函数签名
    sig = inspect.signature(func)
    
    # 创建 wrapper
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

mcp_prompt = """
Your task is to answer the following question. Please follow the instructions carefully:

## Question

{questions}

## Instructions
1. Analyze the question and determine whether you can answer.
2. There are lots of useful tools available, and you can use them to answer the user's questions.(Please use the tools when necessary.)
3. If the question is choice or judgment, you should satisfy the required answer format and don't provide explanations.
4. For other questions, you should provide a concise and complete answer.
5. Finally, the output **must always** follow this format **(do not omit <>):**
    The answer is <your answer>.
"""

def get_mcp_prompt(row):
    if row["code"] == 0:
        return mcp_prompt.format(questions=row["question"])
    else:
        ques = row["question"]
        if row["dataset"] != 0:
            dataset = str(row["dataset"])
            if not dataset.endswith(".csv"):
                dataset += ".csv"
            dataset_path = f"./datasets83/{dataset}"
            ques += f" The relevant dataset is located at: \"{dataset_path}\"\n"
        return mcp_prompt.format(questions=ques)

if __name__ == "__main__":
    # model = "qwen72"
    # models = ["qwen32", "qwen72"]
    # models = ["qwen7"]
    # models = ["qwen7","qwen32","qwen72"]
    # models = ["deepseek"]
    models = ["qwen7","qwen32","qwen72","gpt-4o-mini","deepseek"]
    for model in models:
        with open("keys.json", "r") as f:
            config = json.load(f)
            
        if model not in config:
            raise ValueError(f"Unknown provider: {model}")
        
        model_map = {"deepseek":"deepseek3"}
        cfg = config[model_map[model]] if model in model_map else config[model]

        model_info = {
            "name": cfg["model"],
            "family": "qwen" if "qwen" in cfg["model"].lower() else "openai",
            "functions": [], 
            "vision": False,  
            "json_output": True, 
            "function_calling": True
        }

        # 创建工具
        tools = make_tools_from_module(allfuncscode)
        # code_tool = PythonCodeExecutionTool(LocalCommandLineCodeExecutor(work_dir="coding"))

        sys="""
You are a precise and careful Statistics Analyst. And you can use the tools provided to help you answer questions.
"""

        data1 = pd.read_json("./agents_allq/autogen.json", encoding='utf-8')
        data = data1.copy()
        data_tmp = data.copy()
        col = f"autogen_{model}"
        if col not in data_tmp.columns:
            data_tmp[col] = None

        null_mask = data_tmp[col].isna()
        if null_mask.any():
            begin_index = null_mask.idxmax()
            print(f"[Info] Resume from index {begin_index}")
        else:
            print("[Info] All rows already finished.")
            begin_index = len(data_tmp)
        needs = ["a57","a224","a237"]
        begin_index =0
        for index, row in tqdm(data.iterrows(), total=len(data)):
            # if index < begin_index or ("[Error] Row" not in str(data.loc[index,col])):
            if index < begin_index or row["index"] not in needs:
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

            # 每10条保存一次
            if (index + 1) % 10 == 0:
                data_tmp.to_json("./agents_allq/autogen.json", force_ascii=False, orient='records', indent=2)
                print(f"[Info] Saved progress at index {index + 1}")

        # 保存最终结果
        data_tmp.to_json("./agents_allq/autogen.json", force_ascii=False, orient='records', indent=2)
        print("[Info] Finished processing all rows and saved final result.")