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

class PracticalScienceJudger:
    SYS_PROMPT =f"""Your task is to evaluate the rigor and rationality of the given modeling paper in mathematical modeling, particularly focusing on the assumptions and rationality.

**Evaluation Criteria**:
### 2. Rigor and Rationality of Modeling

#### 2.1 Assumptions
Clear and explicit. These assumptions are the foundation of the model and need to be rigorously justified.
- Are the model assumptions clearly explained?
- Are the assumptions reasonable and consistent with the background of the actual problem?
- Is the rationality and impact of the assumptions considered?

**Scoring Criteria**:
1-2 = Completely unreasonable; 3-4 = Partially reasonable; 5-6 = Average; 7-8 = Reasonable; 9-10 = Very reasonable.

#### 2.2 Rationality
The rationality of the model is key to evaluation. Evaluation criteria can include: whether an appropriate model is chosen, whether the model can realistically reflect the problem, etc.
- Has the model chosen appropriate methods and metrics?
- Does the structure of the model scientifically reflect the actual problem?

**Scoring Criteria**:
1-2 = Completely unreasonable; 3-4 = Partially reasonable; 5-6 = Average; 7-8 = Reasonable; 9-10 = Very reasonable.

**Output Format**:
Example:
### 2.1 Assumptions\n\n**Evaluation:**\n\nThe assumptions are crucial for model building, but the modeling analysis does not describe the assumptions in sufficient detail. The rationality and impact of the assumptions are not fully justified, lacking detailed explanations of data sources, data distribution, and competition characteristics. For example, the assumption about "serve advantage" is mentioned but not detailed on how it is quantified and integrated into the model. Additionally, the assumptions are not clearly explained, making the foundation of the model less robust.
**Score:**\n<reason> The model assumptions are not clear enough and lack sufficient explanation of their sources and impacts </reason>  \n<score> 3 </score>  
### 2.2 Rationality \n\n**Evaluation:**\n\nThe rationality of the model is average. The modeler chose to evaluate player performance based on match data (such as points won, games won, and sets won), which is reasonable to some extent. However, the specific modeling methods and metrics are not detailed. For example, how to quantify "performance score", how to handle time series data, and whether psychological factors in the competition are considered. Although some possible methods (such as time series analysis, regression, or classification) are mentioned, their specific applications and reasons for selection are not deeply explained. The structure of the model may have certain limitations in reflecting the actual problem.
**Score:**\n<reason> The rationality of the model is average, with methods and metrics not detailed, and the model structure has limitations </reason>  \n<score> 5 </score>

Please objectively and detailedly evaluate the rigor and rationality of the modeling according to the above evaluation criteria, and give the final score and reason.
### 2.1 Assumptions\n\n**Evaluation:
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
        result["calculated_overall"] = average_score / 10
        
        return result 