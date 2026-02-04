import pandas as pd
import json5
import traceback
import statool
import inspect
import numpy as np
import datetime
import decimal
import json
from tqdm import tqdm
import re
import importlib.util
import inspect
from smolagents import tool
import os
from smolagents import OpenAIServerModel, CodeAgent, LogLevel
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_mcp_prompt,get_model_cfg

def create_smolagent_tools_with_doc(TOOL_FILE):
    spec = importlib.util.spec_from_file_location("my_tools", TOOL_FILE)
    my_tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(my_tools)
    def prepare_func_for_tool(func):
        sig = inspect.signature(func)
        params = []
        annotations = func.__annotations__.copy() if hasattr(func, '__annotations__') else {}

        # Complete missing parameter type annotations, default to str
        for p in sig.parameters.values():
            if p.annotation == inspect.Parameter.empty:
                p = p.replace(annotation=str)
                annotations[p.name] = str
            params.append(p)

        # Complete return value annotations
        if sig.return_annotation == inspect.Signature.empty:
            return_annotation = str
            annotations['return'] = str
        else:
            return_annotation = sig.return_annotation
            annotations['return'] = return_annotation

        func.__signature__ = sig.replace(parameters=params, return_annotation=return_annotation)
        func.__annotations__ = annotations

        # Process docstring
        doc = func.__doc__ or ""
        lines = doc.splitlines()
        new_lines = []
        in_parameters = False
        args_seen = set()
        param_names = list(sig.parameters.keys())

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("Parameters:"):
                new_lines.append("Args:")
                in_parameters = True
                continue
            elif stripped.startswith("Returns:"):
                # Discard the entire Returns section
                in_parameters = False
                continue

            if in_parameters:
                if stripped and ":" in stripped:
                    # Parse into name (type): desc
                    parts = stripped.split(":", 1)
                    left = parts[0].strip()
                    desc = parts[1].strip()

                    # Attempt to extract name and type
                    if "(" in left and ")" in left:
                        arg_name = left.split("(")[0].strip()
                        arg_type = left[left.find("(")+1:left.find(")")]
                    else:
                        arg_name = left
                        arg_type = "str"

                    # Truncate overly long descriptions
                    if len(desc) > 120:
                        desc = desc[:120] + "..."

                    new_lines.append(f"    {arg_name} ({arg_type}): {desc}")
                    args_seen.add(arg_name)
                else:
                    # Skip empty lines or incorrect formats
                    continue
            else:
                new_lines.append(line)

        # Add default descriptions for parameters missing descriptions
        for arg in param_names:
            if arg not in args_seen:
                new_lines.append(f"    {arg} (str): auto-generated.")
                print(f"[Info] Added missing description for parameter '{arg}' in function '{func.__name__}'.")

        func.__doc__ = "\n".join(new_lines)
        return func

    # Iterate through functions and register as tools, only registering functions within the module
    tools = []
    for name, func in inspect.getmembers(my_tools, inspect.isfunction):
        if func.__module__ == "my_tools" and not name.startswith("_"):  # Only register public functions written in the module
            func_prepared = prepare_func_for_tool(func)
            tool_func = tool(func_prepared)
            tools.append(tool_func)
    return tools


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run smolagent processing.")

    parser.add_argument("--models", nargs="+", default=["deepseek"],
                        help="List of models to use, e.g., --models deepseek model2")
    parser.add_argument("--begin_index", type=int, default=0,
                        help="Starting index")
    parser.add_argument("--input_path", type=str, default="./smolagent.json",
                        help="Input JSON file path")
    parser.add_argument("--output_path", type=str, default="./smolagent.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    models = args.models
    input_path = args.input_path 
    output_path = args.output_path
    begin_index =args.begin_index

    tool_file_path = "./statool.py"

    for model in models:
        cfg = get_model_cfg(model)

        llm = OpenAIServerModel(
            model_id=cfg["model"],
            api_base=cfg["base_url"],
            api_key=cfg["api_key"],
        )

        tools = create_smolagent_tools_with_doc(tool_file_path)

        data = pd.read_json(input_path, encoding='utf-8')
        data_tmp = data.copy()

        col = f"smolagent_{model}"
        if col not in data_tmp.columns:
            data_tmp[col] = None
        
        # Loop Processing
        for index, row in tqdm(data.iterrows(), total=len(data)):
            if index < begin_index:
            # if index < begin_index or row["index"] not in needs:
                continue
            
            print("-" * 20 + f"model{model},row{index}" + "-" * 20)
            
            try:
                agent = CodeAgent(tools=tools, 
                        model=llm,
                        # verbosity_level=LogLevel.ERROR,
                        max_steps=5,
                )
                query = get_mcp_prompt(row)

                result = agent.run(query)
                result = str(result)

                data_tmp.at[index, col] = result

            except Exception:
                print(f"[Error] Row {index} failed:\n{traceback.format_exc()}")
                message = f"[Error] Row {index} failed:\n{traceback.format_exc()}"
                data_tmp.at[index, col] = message

            # Save progress every 10 rows (as per original code)
            if (index + 1) % 10 == 0:
                try:
                    data_tmp.to_json(output_path, force_ascii=False, orient='records', indent=2)
                    print(f"[Info] Saved progress at index {index + 1}")
                except Exception as e:
                    print(f"[Warning] Save failed at index {index + 1}. Continuing execution. Error: {e}")

        # Save final result
        try:
            data_tmp.to_json(output_path, force_ascii=False, orient='records', indent=2)
            print("[Info] Finished processing all rows and saved final result.")
        except Exception as e:
            print(f"[Error] Final save failed: {e}")