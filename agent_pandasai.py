from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
import pandas as pd
from langchain.tools import StructuredTool
import allfuncscode
import inspect
from langchain.agents import AgentExecutor
import traceback
import os
import json
from tqdm import tqdm
import inspect
from pydantic import create_model
from langchain.tools import StructuredTool

# 判断是否为“安全”的简单类型
SAFE_TYPES = {str, int, float, bool}

def simplify_annotation(anno):
    """把不支持的复杂类型降级为 str"""
    if anno in SAFE_TYPES:
        return anno
    return str  # 默认转 str

def auto_wrap_tools(module):
    tools = []
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if inspect.getmodule(func) != module:
            continue

        sig = inspect.signature(func)
        fields = {}

        for param_name, param in sig.parameters.items():
            anno = simplify_annotation(param.annotation if param.annotation != inspect._empty else str)
            default = param.default if param.default != inspect._empty else ...
            fields[param_name] = (anno, default)

        args_schema = create_model(
            f"{name}_Args",
            **fields,
            __config__=type("Config", (), {"arbitrary_types_allowed": True})
        )

        tool = StructuredTool(
            name=name,
            func=func,
            args_schema=args_schema,
            description=func.__doc__ or f"Tool for {name}"
        )
        tools.append(tool)
    return tools

mcp_prompt = """
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
            dataset = str(row["dataset"])
            if not dataset.endswith(".csv"):
                dataset += ".csv"
            dataset_path = f"./datasets83/{dataset}"
            ques = row["question"] + " The relevant dataset is located at: " + f'"{dataset_path}"' + "\n"
        else:
            ques = row["question"]
        eval_prompt = mcp_prompt.format(questions=ques)
    return eval_prompt

# tools = [
#     StructuredTool.from_function(func)
#     for name, func in inspect.getmembers(allfuncscode, inspect.isfunction)
#     if inspect.getmodule(func) == allfuncscode
# ]
tools = auto_wrap_tools(allfuncscode)
print(f"Loaded {len(tools)} tools from allfuncscode module.")

model = "qwen32"
with open("keys.json", "r") as f:
    config = json.load(f)
        
if model not in config:
    raise ValueError(f"Unknown provider: {model}")
cfg = config[model]

model = ChatOpenAI(
    model=cfg["model"],
    temperature=0,
    max_tokens=None,
    timeout=None,
    api_key=cfg["api_key"],
    base_url=cfg["base_url"]
)

if __name__ == "__main__":
    # data1 = pd.read_json("./final_mcp_res_qwen32.json", encoding='utf-8')
    # data = data1[(data1["code"] != 0) & (data1["dataset"] != 0)].reset_index(drop=True)
    # data_tmp = data.copy()
    data1 = pd.read_json("./agents_res/mcp_data_code.json", encoding='utf-8')
    data = data1.copy()
    data_tmp = data.copy()
    for index, row in tqdm(data.iterrows(), total=len(data)):
        query = get_mcp_prompt(row)
        dataset = str(row["dataset"])
        if not dataset.endswith(".csv"):
            dataset += ".csv"
        dataset_path = f"./datasets83/{dataset}"

        try:
            df = pd.read_csv(dataset_path)
        except Exception as e:
            print(f"[Warning] Index {index} Failed to read dataset {dataset_path}: {e}")
            data_tmp.at[index, "pandasai"] = None
            continue

        try:
            agent = create_pandas_dataframe_agent(
                model,
                df,
                verbose=False,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                allow_dangerous_code=True,
                extra_tools=tools
            )

            agent_executor = AgentExecutor.from_agent_and_tools(
                agent.agent,
                agent.tools,
                verbose=False,
                max_execution_time=60,
                max_iterations=5,
            )

            result = agent_executor.run(query)
            data_tmp.at[index, "pandasai"] = result

        except Exception:
            print(f"[Error] Failed to process row {index}:\n{traceback.format_exc()}")
            message = f"[Error] Failed to process row {index}:\n{traceback.format_exc()}"
            data_tmp.at[index, "pandasai"] = message

        # 每20条保存一次
        if (index + 1) % 20 == 0:
            data_tmp.to_json("./agents_res/pandasai.json", force_ascii=False, orient='records', indent=2)
            print(f"[Info] Saved progress at index {index + 1}")

    # 保存最终结果
    data_tmp.to_json("./agents_res/pandasai.json", force_ascii=False, orient='records', indent=2)
    print("[Info] Finished processing all rows and saved final result.")
