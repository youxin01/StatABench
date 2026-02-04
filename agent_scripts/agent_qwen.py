import os
import pandas as pd
import json5
import traceback
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
import statool
import inspect
import numpy as np
import datetime
import decimal
import json
from tqdm import tqdm
import re
from qwen_agent import settings

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_mcp_prompt,get_model_cfg

# Global Settings
settings.MAX_LLM_CALL_PER_RUN = 5

def clean_result(obj):
    """Recursively cleans the result to ensure it is JSON5 serializable."""
    if obj is None:
        return None
    # numpy scalar
    if isinstance(obj, np.generic):
        return obj.item()
    # numpy array
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # pandas Series
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    # datetime objects
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, datetime.timedelta):
        return obj.total_seconds()
    # set types
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    # Decimal
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    # dict / list / tuple
    if isinstance(obj, dict):
        return {k: clean_result(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_result(v) for v in obj]
    # Fallback
    return obj

def parse_docstring_parameters(docstring: str):
    """
    Parses parameter descriptions from the function docstring and returns them as a dictionary.
    """
    if not docstring:
        return {}
    
    param_pattern = re.compile(r'\s*(\w+)\s*:\s*(.+)')
    params = {}
    in_params = False
    for line in docstring.splitlines():
        line = line.strip()
        if line.lower().startswith("parameters"):
            in_params = True
            continue
        if in_params:
            if line == "":
                continue
            if line.lower().startswith("returns"):
                break
            m = param_pattern.match(line)
            if m:
                name, desc = m.groups()
                params[name] = desc.strip()
    return params

def create_qwen_tools_with_doc(module):
    tools = []

    for name, func in inspect.getmembers(module, inspect.isfunction):
        if inspect.getmodule(func) != module:
            continue

        class_name = f"{name}_Tool"

        # Parse docstring
        param_docs = parse_docstring_parameters(func.__doc__)

        def make_call(f):
            def call(self, params: str, **kwargs) -> str:
                parsed = json5.loads(params)
                sig = inspect.signature(f)
                args = {}
                for param in sig.parameters.values():
                    if param.name in parsed:
                        args[param.name] = parsed[param.name]
                    elif param.default != inspect.Parameter.empty:  # Has default value
                        args[param.name] = param.default
                    else:
                        raise ValueError(f"Missing required parameter: {param.name}")
                result = f(**args)
                result = clean_result(result)
                return json5.dumps(result, ensure_ascii=False)
            return call

        # Build parameter list
        parameters = []
        sig = inspect.signature(func)
        for param in sig.parameters.values():
            parameters.append({
                "name": param.name,
                "type": "string",  # Default to string, parsed internally
                "description": param_docs.get(param.name, f"{param.name} parameter"),
                "required": param.default == inspect.Parameter.empty
            })

        attrs = {
            "name": name,
            "description": func.__doc__ or f"{name} function tool",
            "parameters": parameters,
            "call": make_call(func)
        }

        ToolClass = type(class_name, (BaseTool,), attrs)
        register_tool(ToolClass)
        tools.append(ToolClass)

    return tools

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run qwenagent processing.")

    parser.add_argument("--models", nargs="+", default=["deepseek"],
                        help="List of models to use, e.g., --models deepseek model2")
    parser.add_argument("--begin_index", type=int, default=0,
                        help="Starting index")
    parser.add_argument("--input_path", type=str, default="./qwen_agent.json",
                        help="Input JSON file path")
    parser.add_argument("--output_path", type=str, default="./qwen_agent.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    models = args.models
    input_path = args.input_path 
    output_path = args.output_path
    begin_index =args.begin_index

    # System Prompt
    system_instruction = "You are a professional statistics analyst, and could answer user's questions with or without using tools."

    # Tool Initialization
    tool_classes = create_qwen_tools_with_doc(statool)
    tools_instances = [t() for t in tool_classes]

    for model in models:
        print(f"Using model: {model}")
        
        cfg = get_model_cfg(model)
        llm_cfg = {
            'model': cfg["model"],
            'model_server': cfg["base_url"],
            'api_key': cfg["api_key"],
            'generate_cfg': {'temperature': 0.0}
        }

        # Load Data
        data = pd.read_json(input_path, encoding='utf-8')
        data_tmp = data.copy()
        
        col = f"qwen_agent_{model}"
        if col not in data_tmp.columns:
            data_tmp[col] = None

        # Loop processing
        for index, row in tqdm(data.iterrows(), total=len(data)):
            if index < begin_index:    
            # if index < begin_index or row["index"] not in needs:
                continue
            
            print("-" * 20 + f"agent{model},row{index}" + "-" * 20)
            
            try:
                bot = Assistant(
                    llm=llm_cfg,
                    system_message=system_instruction,
                    function_list=tools_instances,
                )
                
                query = get_mcp_prompt(row)
                dataset = str(row["dataset"])

                messages = [
                    {'role': 'user', 'content': [{'text': query}]}
                ]

                last_response = None
                for response in bot.run(messages=messages):
                    last_response = response

                if last_response and "content" in last_response[-1]:
                    result = last_response[-1]["content"]
                else:
                    result = None

                data_tmp.at[index, col] = result

            except Exception:
                error_msg = f"[Error] Row {index} failed:\n{traceback.format_exc()}"
                print(error_msg)
                data_tmp.at[index, col] = error_msg

            # Save progress every 10 rows
            if (index + 1) % 10 == 0:
                data_tmp.to_json(output_path, force_ascii=False, orient='records', indent=2)
                print(f"[Info] Saved progress at index {index + 1}")

        # Save final result
        data_tmp.to_json(output_path, force_ascii=False, orient='records', indent=2)
        print("[Info] Finished processing all rows and saved final result.")