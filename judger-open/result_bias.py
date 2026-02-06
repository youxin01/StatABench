import json
import ast
import os
from openai import OpenAI
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_model_cfg


def parse_and_save_evaluation_data(text):
    reason_pattern = r'<reason>\s*(.*?)\s*</reason>'
    score_pattern = r'<score>\s*(\d+)\s*</score>'

    reasons = re.findall(reason_pattern, text, re.DOTALL)
    scores = re.findall(score_pattern, text, re.DOTALL)

    if not reasons or not scores:
        print("No matches found for reasons or scores!")
        return {}

    if len(reasons) != len(scores):
        print("Mismatch between number of reasons and scores!")
        return {}

    result_dict = {}
    for i, (reason, score) in enumerate(zip(reasons, scores), 1):
        result_dict[f"analysis_{i}"] = {
            "reason": reason.strip(),
            "score": int(score)
        }

    return result_dict

class ResultandBiasJudger:
    SYS_PROMPT = f"""
Your task is to evaluate the result analysis and bias analysis of the given modeling paper, particularly focusing on the rationality, interpretability of the model output, and the identification and correction of biases.

**Evaluation Criteria**:
### 4. Result Analysis and Bias Analysis

#### 4.1 Result Analysis
- Are the model output results clear and as expected?
- Does the result provide sufficient analysis to explain the model's inference process?
- Are the model results interpretable and do they help in understanding the essence of the problem?
- Does the analysis provide clear conclusions and highlight the strengths and weaknesses of the model?

**Scoring Criteria**:
1-2 = Completely unclear; 3-4 = Partially clear; 5-6 = Average; 7-8 = Clear; 9-10 = Very clear.

#### 4.2 Bias Analysis
- Does the model identify and analyze potential biases?
- Does it consider data bias, model bias, and other factors?
- Does the model appropriately correct biases to reduce their impact on the results?

**Scoring Criteria**:
1-2 = Completely ignored biases; 3-4 = Partially considered biases; 5-6 = Average; 7-8 = Considered biases and corrected; 9-10 = Very thorough, biases effectively corrected.

**Output Format**:
Example 1:
### 4.1 Result Analysis\n\n**Evaluation:**\n\nThe model output results are clear and well explain the model's inference process. The modeler has detailed the background and significance of the model results, helping to understand the core of the problem. The results show a reasonable inference path, making the entire analysis process more transparent. The analysis also provides clear conclusions and highlights the strengths and weaknesses of the model.
**Score:**\n<reason> The result analysis is very clear and effectively supports decision-making </reason> \n<score> 9 </score>  

### 4.2 Bias Analysis\n\n**Evaluation:**\n\nThe model effectively identifies and analyzes biases, particularly potential data biases. The modeler provides correction measures for biases and explains how these corrections affect the model results. Although there are still some biases in certain aspects of the model, overall, a comprehensive correction has been made.
**Score:**\n<reason> The bias analysis is thorough, and biases have been effectively corrected </reason> \n<score> 8 </score>

Please objectively and detailedly evaluate the result analysis and bias analysis of the modeling according to the above evaluation criteria, and provide the final score and reason.
### 4.1 Result Analysis\n\n**Evaluation:
"""

    USER_PROMPT = """Please evaluate the practicality and scientificity of the following statistical modeling paper:

{writing}

Provide scores and explanations for each component.

Your Response:
"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.configs = get_model_cfg(model)
        self.client = OpenAI(api_key=self.configs["api_key"],base_url=self.configs["base_url"])

    def run(self, writing: str) -> dict:
        messages = [
            {'role': 'system', 'content': self.SYS_PROMPT},
            {'role': 'user', 'content': self.USER_PROMPT.format(writing=writing)}
        ]

        response = self.client.chat.completions.create(
            model=self.configs["model"],
            messages=messages,
            temperature=0.0,
            n=1,
        )
        
        content = response.choices[0].message.content
        result_dict = parse_and_save_evaluation_data(content)
        score_values = [item['score'] for item in result_dict.values()]
        total_score = sum(score_values)
        count = len(score_values)
        average_score = total_score / count
        result = {}
        result["total_score"] = total_score
        result["calculated_overall"] = average_score /10 
        
        return result 