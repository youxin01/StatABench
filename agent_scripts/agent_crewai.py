import os
import pandas as pd
import inspect
from typing import Any
import allfuncscode
from crewai import Crew, Task, Agent, LLM, Process
from crewai_tools import FileReadTool, CodeInterpreterTool
from crewai.tools import tool
import traceback
import json
from tqdm import tqdm


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
        return mcp_prompt.format(questions=row["question"])
    else:
        ques = row["question"]
        if row["dataset"] != 0:
            dataset = str(row["dataset"])
            if not dataset.endswith(".csv"):
                dataset += ".csv"
            dataset_path = os.path.join("./datasets83", dataset)
            ques += f' The relevant dataset is located at: "{dataset_path}"\n'
        return mcp_prompt.format(questions=ques)

if __name__ == "__main__":

    # models = ["qwen32", "qwen72"]
    # models = ["deepseek3","qwen7"]
    # models = ["qwen7","qwen32","qwen72"]
    models = ["deepseek"]
    # models = ["qwen7","qwen32","qwen72","gpt-4o-mini","deepseek"]

    for model in models:
        print(f"Using model: {model}")
        with open("keys.json", "r") as f:
            config = json.load(f)
            
        if model not in config:
            raise ValueError(f"Unknown provider: {model}")
        
        model_map = {"deepseek":"deepseek3"}

        cfg = config[model_map[model]] if model in model_map else config[model]

        # for qwen
        if model == "deepseek":
            llm = LLM(
            model=f"dashscope/{cfg['model']}",
            base_url=cfg["base_url"],
            api_key=cfg["api_key"]
        )
        else:
            llm = LLM(
                model=f"dashscope/{cfg['model']}" if "qwen" in cfg["model"].lower() else cfg["model"],
                base_url=cfg["base_url"],
                api_key=cfg["api_key"]
            )

        # 自动包装 Crewai 工具
        tools = auto_wrap_tools(allfuncscode)
        code_interpreter = CodeInterpreterTool(unsafe_mode=True)

        # 加载数据
        data1 = pd.read_json("./agents_allq/crewai.json", encoding='utf-8')
        data = data1.copy()
        data_tmp = data.copy()

        col = f"crewai_{model}"
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
        needs = ["a21"]
        for index, row in tqdm(data.iterrows(), total=len(data)):
            # if index < begin_index or ("[Error] Row" not in str(data.loc[index,col])):
            # if index < begin_index:
            if index < begin_index or row["index"] not in needs:
                continue
            print("-"*20+f"agent{model},row{index}"+"-"*20)
            try:
                dataset = str(row["dataset"])
                if not dataset.endswith(".csv"):
                    dataset += ".csv"
                dataset_path = f"./datasets83/{dataset}"

                # 创建 Agent
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

                # 创建并运行 Crew
                crew = Crew(
                    agents=[dataset_inference_agent],
                    tasks=[coding_task],
                    verbose=True,
                    process=Process.sequential
                )
                result = crew.kickoff()

                # 安全获取结果
                data_tmp.loc[index, f"crewai_{model}"] = getattr(result, "raw", None)

            except Exception:
                print(f"[Error] Row {index} failed:\n{traceback.format_exc()}")
                message = f"[Error] Row {index} failed:\n{traceback.format_exc()}"
                data_tmp.loc[index, f"crewai_{model}"] = message

            # 每10条保存一次
            if (index + 1) % 10 == 0:
                data_tmp.to_json("./agents_allq/crewai.json", force_ascii=False, orient='records', indent=2)
                print(f"[Info] Saved progress at index {index + 1}")

        # 保存最终结果
        data_tmp.to_json("./agents_allq/crewai.json", force_ascii=False, orient='records', indent=2)
        print("[Info] Finished processing all rows and saved final result.")
