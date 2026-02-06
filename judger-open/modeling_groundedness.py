import json
import ast
import os
from openai import OpenAI
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_model_cfg

class ModelingGroundednessJudger:
     SYS_PROMPT = """You are currently evaluating statistical modeling papers. Your task is to assess how well the solution's modeling approach is grounded in statistical theory and scientific principles. You should evaluate based on the role you are given.

Score each aspect from 0-1, starting at 0 and requiring justification for any increase:

1.  Statistical Theory and Model Specification(0-1):
    0.00: Fundamentally flawed or missing
        Example: No clear probabilistic model, used incorrect statistical concepts for the data type.
    0.25: Basic but problematic
        Example: A standard model (e.g., linear regression) is chosen without stating or checking its underlying assumptions (e.g., linearity, independence, normality of residuals).
    0.50: Sound but incomplete
        Example: An appropriate model is chosen, but the rationale for the choice of specific distributions or link functions is missing. Key model assumptions are stated but not verified.
    0.75: Strong with minor gaps
        Example: Well-specified model with most assumptions justified and discussed. The choice of variables and their relationships are theoretically sound but could be explored further.
    1.00: Excellent and rigorous
        Example: A complete and well-justified statistical framework. All model assumptions are explicitly stated, theoretically grounded in the problem domain, and rigorously checked.

2.  Data Quality and Feature Engineering(0-1):
    0.00: No connection to data reality
        Example: Model is presented without any description of the data source, its collection process, or its characteristics.
    0.25: Superficial consideration
        Example: Data is used, but there is no mention of data cleaning, handling of missing values, or outlier detection. Features are used directly without any transformation or justification.
    0.50: Partial integration
        Example: Basic data cleaning is performed. Some feature engineering is attempted, but it's simplistic or not well-justified. The representativeness of the sample is not discussed.
    0.75: Good but not comprehensive
        Example: Thorough data preprocessing and thoughtful feature engineering. The potential impact of data limitations (e.g., sampling bias) is mentioned but not fully addressed in the model.
    1.00: Complete integration
        Example: Comprehensive exploratory data analysis (EDA), robust handling of data issues, sophisticated and well-justified feature engineering, and a critical discussion of the data's scope, limitations, and potential biases.

3.  Methodology and Estimation(0-1):
    0.00: Elementary/inappropriate
        Example: Using methods for independent data on time-series data; using linear models for clearly non-linear, non-transformable relationships.
    0.25: Basic techniques only
        Example: Using a simple model when the problem complexity clearly calls for more advanced techniques (e.g., handling multicollinearity, using regularization) without justification.
    0.50: Appropriate but limited
        Example: An appropriate estimation method (e.g., MLE, OLS) is used, but model selection is ad-hoc or relies on a single metric (e.g., only R-squared) without considering model complexity (e.g., AIC/BIC).
    0.75: Advanced with minor issues
        Example: Sophisticated methods (e.g., Bayesian inference, GAMs) are used correctly, but the choice of priors or hyperparameters is not well-defended.
    1.00: State-of-the-art
        Example: Cutting-edge or highly appropriate techniques are correctly implemented. The parameter estimation process is clearly explained and robust. Model selection is principled and compares multiple candidate models.

4.  Model Validation and Diagnostics(0-1):
    0.00: No validation
        Example: Model results (e.g., coefficients, predictions) are presented without any form of verification or performance metric.
    0.25: Minimal testing
        Example: Only in-sample performance metrics (e.g., R-squared on the training set) are reported.
    0.50: Partial validation
        Example: A simple train-test split is performed, but there's no residual analysis, goodness-of-fit testing, or check of model assumptions.
    0.75: Thorough but not complete
        Example: Cross-validation is used, and key diagnostics are performed. However, the model's performance on critical edge cases or subgroups is not explored.
    1.00: Comprehensive validation
        Example: Employs multiple validation strategies (e.g., cross-validation, out-of-sample testing), rigorous diagnostics (e.g., residual analysis, influence plots), goodness-of-fit tests, and sensitivity analysis of model parameters.

5.  Implementation Quality and Reproducibility(0-1):
    0.00: Poor/incorrect
        Example: Code contains clear errors in statistical formulas or misuse of statistical library functions.
    0.25: Basic but flawed
        Example: The overall concept is correct, but the implementation has significant errors that affect the results' validity.
    0.50: Workable but needs improvement
        Example: The model functions correctly, but the code is inefficient, poorly documented, and not easily reproducible (e.g., hard-coded paths, no random seed set).
    0.75: Good with minor issues
        Example: Well-implemented using standard statistical libraries, with mostly clear code. Minor improvements in documentation or structure are possible.
    1.00: Excellent implementation
        Example: Code is efficient, clear, well-documented, and fully reproducible (e.g., code, data, and environment are provided or clearly specified).

---

Your response must follow this exact format:

Your Response:
```json
{
    "statistical_theory_and_model_specification": {
        "score": 0.0,
        "explanation": "Detailed justification for score"
    },
    "data_quality_and_feature_engineering": {
        "score": 0.0,
        "explanation": "Detailed justification for score"
    },
    "methodology_and_estimation": {
        "score": 0.0,
        "explanation": "Detailed justification for score"
    },
    "model_validation_and_diagnostics": {
        "score": 0.0,
        "explanation": "Detailed justification for score"
    },
    "implementation_quality_and_reproducibility": {
        "score": 0.0,
        "explanation": "Detailed justification for score"
    },
    "calculated_overall": 0.0,
    "overall_feedback": "Critical analysis of strengths and weaknesses"
}
```

---

Note: Scores must be exactly 0.00, 0.25, 0.50, 0.75, or 1.00. Start at 0 and justify each increment. Be extremely critical. You should also give your score and explanation from your role's perspective."""

     USER_PROMPT = """Please evaluate the modeling groundedness of the following statistical modeling paper:

{writing}

Provide scores and detailed justification for each aspect. Remember your role as {role_name}. Your judgement should be based on this role's perspective.

Your Response:
"""

     def __init__(self, model: str = "gpt-4o-mini"):
            self.configs = get_model_cfg(model)
            self.client = OpenAI(api_key=self.configs["api_key"],base_url=self.configs["base_url"])

     def run(self, writing: str, role: dict = None) -> dict:
            role_name = role["name"].strip()
            role_details = role["details"].strip()
            messages = [
                {'role': 'system', 'content': role_details + "\n\n" + self.SYS_PROMPT},
                {'role': 'user', 'content': self.USER_PROMPT.format(writing=writing, role_name=role_name)}
            ]

            response = self.client.chat.completions.create(
                model=self.configs["model"],
                messages=messages,
                temperature=0.0,
                n=1,
            )
            
            content = response.choices[0].message.content
            json_str = content.split("```json")[1].split("```")[0].strip()
            result = ast.literal_eval(json_str)
            
            # if "implementation_quality" in result:
            #      result["implementation"] = result.pop("implementation_quality")
                
            # # Calculate overall score as average of individual scores
            # scores = [result[aspect]["score"] for aspect in [
            #      "mathematical_foundation", "real_world_integration", 
            #      "technical_sophistication", "validation", "implementation"
            # ]]
            # result["calculated_overall"] = sum(scores) / len(scores)
            # result["role"] = role

            statistical_aspects = [
            "statistical_theory_and_model_specification",
            "data_quality_and_feature_engineering",
            "methodology_and_estimation",
            "model_validation_and_diagnostics",
            "implementation_quality_and_reproducibility"
            ]

            scores = [result[aspect]["score"] for aspect in statistical_aspects if aspect in result]

            if scores:
                result["calculated_overall"] = sum(scores) / len(scores)
            else:
                result["calculated_overall"] = 0.0

            result["role"] = role
            
            return result 