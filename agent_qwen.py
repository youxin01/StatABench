import os
import pandas as pd
import json5
import traceback
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
import allfuncscode
import inspect
import numpy as np
import datetime
import decimal
import json
from tqdm import tqdm
import re
from qwen_agent import settings

settings.MAX_LLM_CALL_PER_RUN = 5

# 记住重跑改文件,总共跑了111，花费20¥.

def clean_result(obj):
    """递归清洗结果，保证可以 JSON5 序列化"""
    if obj is None:
        return None
    # numpy 标量
    if isinstance(obj, np.generic):
        return obj.item()
    # numpy 数组
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # pandas 系列
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    # datetime 系列
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, datetime.timedelta):
        return obj.total_seconds()
    # 集合类型
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
    # 兜底处理
    return obj

def parse_docstring_parameters(docstring: str):
    """
    从函数 docstring 中解析参数说明，返回字典形式。
    假设 docstring 按 Google style 格式：
        Parameters:
            param_name: 描述
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

        # 解析 docstring
        param_docs = parse_docstring_parameters(func.__doc__)

        def make_call(f):
            def call(self, params: str, **kwargs) -> str:
                parsed = json5.loads(params)
                sig = inspect.signature(f)
                args = {}
                for param in sig.parameters.values():
                    if param.name in parsed:
                        args[param.name] = parsed[param.name]
                    elif param.default != inspect.Parameter.empty:  # 有默认值
                        args[param.name] = param.default
                    else:
                        raise ValueError(f"缺少必要参数: {param.name}")
                result = f(**args)
                result = clean_result(result)
                return json5.dumps(result, ensure_ascii=False)
            return call


        # 构建参数列表
        parameters = []
        sig = inspect.signature(func)
        for param in sig.parameters.values():
            parameters.append({
                "name": param.name,
                "type": "string",  # 默认字符串，可内部解析
                "description": param_docs.get(param.name, f"{param.name} 参数"),
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
    # models = ["qwen32", "qwen72"]
    # models = ["gpt-4o-mini"]
    # models = ["deepseek"]
    # models = ["gpt-4o-mini","deepseek"]
    models = ["qwen7","qwen32","qwen72","gpt-4o-mini","deepseek"]

    for model in models:
        with open("keys.json", "r") as f:
            config = json.load(f)
            
        if model not in config:
            raise ValueError(f"Unknown provider: {model}")
        model_map = {"deepseek":"deepseek3"}

        cfg = config[model_map[model]] if model in model_map else config[model]

        
        llm_cfg = {
            'model': cfg["model"],
            'model_server': cfg["base_url"],
            'api_key': cfg["api_key"],
            'generate_cfg': {'temperature': 0.0}
        }

        # 创建工具
        tools = create_qwen_tools_with_doc(allfuncscode)
        tools_instances = [t() for t in tools]

        # 初始化 Assistant
        system_instruction = "You are a professional statistics analyst, and could answer user's questions with or without using tools."


        # 加载数据
        data1 = pd.read_json("./agents_allq/qwen_agent.json", encoding='utf-8')
        data = data1.copy()
        data_tmp = data.copy()
        col = f"qwen_agent_{model}"
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
            print("-"*20+f"agent{model},row{index}"+"-"*20)
            try:
                bot = Assistant(
                    llm=llm_cfg,
                    system_message=system_instruction,
                    function_list=tools_instances,
                )
                query = get_mcp_prompt(row)
                dataset = str(row["dataset"])
                if not dataset.endswith(".csv"):
                    dataset += ".csv"
                dataset_path = f"./datasets83/{dataset}"

                # messages = [
                #     {'role': 'user', 'content': [{'text': query}, {'file': dataset_path}]}
                # ]
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

                data_tmp.at[index, f"qwen_agent_{model}"] = result

            except Exception:
                print(f"[Error] Row {index} failed:\n{traceback.format_exc()}")
                message = f"[Error] Row {index} failed:\n{traceback.format_exc()}"
                data_tmp.at[index, f"qwen_agent_{model}"] = message

            # 每10条保存一次
            if (index + 1) % 10 == 0:
                data_tmp.to_json("./agents_allq/qwen_agent.json", force_ascii=False, orient='records', indent=2)
                print(f"[Info] Saved progress at index {index + 1}")

        # 保存最终结果
        data_tmp.to_json("./agents_allq/qwen_agent.json", force_ascii=False, orient='records', indent=2)
        print("[Info] Finished processing all rows and saved final result.")