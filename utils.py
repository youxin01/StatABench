import os
import json
import pandas as pd

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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    potential_path = os.path.join(current_dir, "data", "datasets83")
    if os.path.isdir(potential_path):
        dataset_dir = potential_path
    else:
        raise FileNotFoundError("Could not find 'data/datasets83' directory in project root.")
    if row["code"] == 0:
        return mcp_prompt.format(questions=row["question"])
    else:
        ques = row["question"]
        if row["dataset"] != 0:
            dataset = str(row["dataset"])
            if not dataset.endswith(".csv"):
                dataset += ".csv"
            dataset_path = os.path.join(dataset_dir, dataset)
            ques += f' The relevant dataset is located at: "{dataset_path}"\n'
        return mcp_prompt.format(questions=ques)
    
def get_model_cfg(model):
    key_path = os.path.join("./keys.json")
    with open(key_path,"r") as f:
        config = json.load(f)
    cfg = config[model]
    if cfg is None:
        raise ValueError(f"Unknown provider: {model}")
    return cfg