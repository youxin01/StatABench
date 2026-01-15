import pandas as pd
import json5
import traceback
import allfuncscode
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

def create_smolagent_tools_with_doc(TOOL_FILE):
    spec = importlib.util.spec_from_file_location("my_tools", TOOL_FILE)
    my_tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(my_tools)
    def prepare_func_for_tool(func):
        sig = inspect.signature(func)
        params = []
        annotations = func.__annotations__.copy() if hasattr(func, '__annotations__') else {}

        # 补全缺失参数类型注解，默认 str
        for p in sig.parameters.values():
            if p.annotation == inspect.Parameter.empty:
                p = p.replace(annotation=str)
                annotations[p.name] = str
            params.append(p)

        # 补全返回值注解
        if sig.return_annotation == inspect.Signature.empty:
            return_annotation = str
            annotations['return'] = str
        else:
            return_annotation = sig.return_annotation
            annotations['return'] = return_annotation

        func.__signature__ = sig.replace(parameters=params, return_annotation=return_annotation)
        func.__annotations__ = annotations

        # 处理 docstring
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
                # 丢弃整个 Returns 部分
                in_parameters = False
                continue

            if in_parameters:
                if stripped and ":" in stripped:
                    # 解析成 name (type): desc
                    parts = stripped.split(":", 1)
                    left = parts[0].strip()
                    desc = parts[1].strip()

                    # 尝试取 name 和 type
                    if "(" in left and ")" in left:
                        arg_name = left.split("(")[0].strip()
                        arg_type = left[left.find("(")+1:left.find(")")]
                    else:
                        arg_name = left
                        arg_type = "str"

                    # 截断过长的描述
                    if len(desc) > 120:
                        desc = desc[:120] + "..."

                    new_lines.append(f"    {arg_name} ({arg_type}): {desc}")
                    args_seen.add(arg_name)
                else:
                    # 空行/格式不对直接跳过
                    continue
            else:
                new_lines.append(line)

        # 对缺失描述的参数补充默认描述
        for arg in param_names:
            if arg not in args_seen:
                new_lines.append(f"    {arg} (str): auto-generated.")
                print(f"[Info] Added missing description for parameter '{arg}' in function '{func.__name__}'.")

        func.__doc__ = "\n".join(new_lines)
        return func

    # 遍历函数并注册为 tool，只注册自己模块里的函数
    tools = []
    for name, func in inspect.getmembers(my_tools, inspect.isfunction):
        if func.__module__ == "my_tools" and not name.startswith("_"):  # 只注册自己写的公开函数
            func_prepared = prepare_func_for_tool(func)
            tool_func = tool(func_prepared)
            tools.append(tool_func)
    return tools

mcp_prompt = """"
Your task is to answer the following question. Please follow the instructions carefully:

## Question

{questions}

## Instructions
1. Analyze the question and determine whether you can answer.
2. If the question is choice or judgment, you should satisfy the required answer format and don't provide explanations.
3. For other questions, you should provide a concise and complete answer.
4. Finally, the output **must always** follow this format **(do not omit <>):**
    The answer is <your answer>.

"""

def get_mcp_prompt(row):
    if row["code"] == 0:
        eval_prompt = mcp_prompt.format(questions=row["question"])
    else:
        if row["dataset"] != 0:
            if not str(row["dataset"]).endswith(".csv"):
                dataset = str(row["dataset"]) + ".csv"
            else:
                dataset = str(row["dataset"])
            dataset_path = f"./datasets83/{dataset}"
            ques = row["question"]+" The relevant dataset is located at: " + f'"{dataset_path}"' + "\n"
        else:
            ques = row["question"]
        eval_prompt = mcp_prompt.format(questions=ques)
    return eval_prompt

if __name__ == "__main__":
    # models = ["qwen32", "qwen72", "gpt-4o-mini"]
    # models = ["deepseek","qwen7"]
    # models = ["gpt-4o-mini"]
    # models = ["deepseek"]
    models = ["qwen7","qwen32","qwen72","gpt-4o-mini","deepseek"]

    for model in models:
        with open("keys.json", "r") as f:
            config = json.load(f)
            
        if model not in config:
            raise ValueError(f"Unknown provider: {model}")
        model_map = {"deepseek":"deepseek3"}

        cfg = config[model_map[model]] if model in model_map else config[model]

        llm = OpenAIServerModel(
            model_id=cfg["model"],
            api_base=cfg["base_url"],
            api_key=cfg["api_key"],
        )

        tools = create_smolagent_tools_with_doc("./allfuncscode.py")

        # 加载数据
        data1 = pd.read_json("./agents_allq/smolagent.json", encoding='utf-8')
        data = data1.copy()
        data_tmp = data.copy()

        col = f"smolagent_{model}"
        if col not in data_tmp.columns:
            data_tmp[col] = None
        null_mask = data_tmp[col].isna()
        if null_mask.any():
            begin_index = null_mask.idxmax()
            print(f"[Info] Resume from index {begin_index}")
        else:
            print("[Info] All rows already finished.")
            begin_index = len(data_tmp)
        begin_index = 0
        needs = ["a57","a224","a237"]
        for index, row in tqdm(data.iterrows(), total=len(data)):
            if index < begin_index or row["index"] not in needs:
                continue
            print("-"*20+f"model{model},row{index}"+"-"*20)
            try:
                agent = CodeAgent(tools=tools, 
                        model=llm,
                        # verbosity_level=LogLevel.ERROR,
                        max_steps=5,
                )
                query = get_mcp_prompt(row)
                dataset = str(row["dataset"])
                if not dataset.endswith(".csv"):
                    dataset += ".csv"
                dataset_path = f"./datasets83/{dataset}"

                result = agent.run(query)

                result = str(result)

                data_tmp.at[index, f"smolagent_{model}"] = result

            except Exception:
                print(f"[Error] Row {index} failed:\n{traceback.format_exc()}")
                message = f"[Error] Row {index} failed:\n{traceback.format_exc()}"
                data_tmp.at[index, f"smolagent_{model}"] = message

            if (index + 1) % 1 == 0:
                try:
                    data_tmp.to_json("./agents_allq/smolagent.json", force_ascii=False, orient='records', indent=2)
                    print(f"[Info] Saved progress at index {index + 1}")
                except Exception as e:
                    print(f"[Warning] Save failed at index {index + 1}. Continuing execution. Error: {e}")

        # 保存最终结果
        try:
            data_tmp.to_json("./agents_allq/smolagent.json", force_ascii=False, orient='records', indent=2)
            print("[Info] Finished processing all rows and saved final result.")
        except Exception as e:
            print(f"[Error] Final save failed: {e}")