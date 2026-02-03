import re
import pandas as pd
from openai import OpenAI
import json
from tqdm import tqdm

def gpt_chat(
    model: str,
    prompt: str = None
) -> str:
    with open("keys.json", "r") as f:
        config = json.load(f)
        
    if model not in config:
        raise ValueError(f"Unknown provider: {model}")
    cfg = config[model]

    client = OpenAI(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"]
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    completion = client.chat.completions.create(
        model=cfg["model"],
        messages=messages
    )
    return completion.choices[0].message.content

def extract_choice(text):
    match_option = re.search(r"The answer is\s*[^\w\s]*([A-Fa-f])", text, re.IGNORECASE)
    if match_option:
        return match_option.group(1)
    
    pattern_without_brackets = r"The answer is\s*([^\n]*)"
    match = re.search(pattern_without_brackets, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return None

def extract_decision(text):
    match = re.search(r"The answer is\s*[^\w\s]*(Yes|No|True|False|dependent|independent)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    
    pattern_without_brackets = r"The answer is\s*([^\n]*)"
    match = re.search(pattern_without_brackets, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def extract_numeric(text):
    pattern = r"The answer is\s*([\d]+\.?\d*)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1)
    pattern_without_brackets = r"The answer is\s*([^\n]*)"
    match = re.search(pattern_without_brackets, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def extract_normal(text):
    pattern_with_brackets = r"The answer is\s*(.*)"
    match = re.search(pattern_with_brackets, text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    
    pattern_without_brackets = r"The answer is\s*([^\n]*)"
    match = re.search(pattern_without_brackets, text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return None

def extract_answer(text, answer_type):
    if answer_type == "choice":
        return extract_choice(text)
    elif answer_type == "decision":
        return extract_decision(text)
    elif answer_type == "numeric":
        return extract_numeric(text)
    else:
        return extract_normal(text)
    

def extract_file(input_file, response_col, ex_col= None):
    if ex_col is None:
        ex_col = f"{response_col}_extracted"
    data = pd.read_json(input_file, encoding='utf-8')
    data_copy = data.copy()
    for index, row in data.iterrows():
        response = row[response_col]
        answer_type = row["answer_type"]
        extracted_answer = extract_answer(response, answer_type)
        data_copy.at[index, ex_col] = extracted_answer
    data_copy.to_json(input_file, force_ascii=False, orient='records',indent=2)

def compare_choice(ground,response):
    try:
        if isinstance(ground, str) and len(ground) == 1:
            ground_str = ground.strip().lower()
            response_str = str(response).strip().lower()
            
            if ground_str in ["a","b","c","d","e","f"]:
                if ground_str == response_str:
                    return True, ""
                else:
                    return False, f"Choice mismatch: {ground_str} != {response_str}"
            else:
                return False, f"Invalid ground choice: {ground_str}"
        else:
            return False, "Ground is not a single character string"
    except Exception as e:
        return False, f"Error during comparison: {str(e)}"

def compare_decision(ground,response):
    try:
        if isinstance(ground, str) and isinstance(response, str):
            ground_str = ground.strip().lower()
            response_str = response.strip().lower()
            
            if ground_str in response_str:
                return True, ""
            else:
                return False, f"Decision mismatch: {ground_str} != {response_str}"
        else:
            return False, "Ground or response is not a string"
    except Exception as e:
        return False, f"Error during comparison: {str(e)}"
    
def compare_numeric(ground,response):
    try:
        ground_val = float(ground)
        response_val = float(response)
        if abs(ground_val - response_val) <= 0.001:
            return True, ""
        else:
            return False, f"Numeric mismatch: {ground_val} != {response_val}"
    except Exception as e:
        return False, f"Error during comparison: {str(e)}"


prompt_compare_str = """
For the following question:
{question}

There are two answers:
<ground truth> {ground} </ground truth>

<response> {response} </response>

Please determine whether the response correctly conveys the meaning of the ground. You should follow these rules:
- A response should be considered correct ONLY IF it either:
    (a) explicitly states a conclusion that answers the question (e.g., whether a difference exists or not), and this conclusion aligns with the ground truth; OR
    (b) provides explicit and sufficient results (e.g., test statistics, p-values, model outputs) from which the correct conclusion can be directly inferred, and these results are consistent with the ground truth, even if the response does not explicitly state the conclusion.
- If the response only discusses methodology, analysis plans, or conditional statements (e.g., "if significant, then...") without providing a conclusion or sufficient results, it must be judged as false.
- If the main conclusion is wrong, missing, or includes factual information that contradicts the ground truth, output "false".

Important: Your answer must be strictly "true" or "false" only, without any additional explanation.
"""

def compare_normal(question,ground,response):
    if ground.strip().lower() == response.strip().lower():
        return True, ""
    prompts = prompt_compare_str.format(question=question,ground=ground, response=response)
    response = gpt_chat(prompt=prompts,model="deepseek")
    if response.strip().lower() == "true":
        return True, ""
    elif response.strip().lower() == "false":
        return False, "Response does not match ground truth"
    else:
        return False, f"Unexpected response: {response}"

def compare_results(ground, response, answer_type, question):
    if response is None:
        return False, "Response is None"

    if answer_type == "choice":
        return compare_choice(ground, response)
    elif answer_type == "decision":
        return compare_decision(ground, response)
    elif answer_type == "numeric":
        return compare_numeric(ground, response)
    else:
        return compare_normal(question, ground, response)

def evaluate_file(input_file, response_col, match_col=None):
    data = pd.read_json(input_file, encoding='utf-8')
    data_tmp = data.copy()

    for index, row in tqdm(data.iterrows(),total=len(data)):
        if match_col is None:
            col_name = f"{response_col}_match"
        else:
            col_name = match_col
        if col_name not in data_tmp.columns:
            data_tmp[col_name] = None
        
        ground = row["ground truth"] if row["code"]!=1 else row["ground truth2"]
        response = row[response_col]
        answer_type = row["answer_type"]
        question = row["question"]

        is_correct, _ = compare_results(ground, response, answer_type, question)
        data_tmp.at[index, col_name] = is_correct

        if index % 20 == 0:
            data_tmp.to_json(input_file, force_ascii=False, orient='records', indent=2)
    
    data_tmp.to_json(input_file, force_ascii=False, orient="records", indent=2)
    print(f"[Info] Saved progress to {input_file}")


