import json
import ast
import os
from openai import OpenAI
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_model_cfg

class AnalysisGroundednessJudger:
    SYS_PROMPT = """You are currently evaluating statistical modeling papers. Your task is to assess the depth and rigor of the analysis derived from the statistical model. You should evaluate based on the role you are given.

Score each aspect from 0-1, starting at 0 and requiring justification for any increase:

1.  Depth of Statistical Analysis(0-1):
       0.00: No meaningful analysis
           Example: Only states statistical outputs (e.g., "p < 0.05") without explaining what they mean for the problem.
       0.25: Basic descriptive analysis
           Example: Simply describes the direction of an effect (e.g., "Variable X is positively associated with Y").
       0.50: Standard inferential analysis
           Example: Correctly interprets the magnitude and statistical significance of coefficients in the context of the problem.
       0.75: Advanced analysis
           Example: Goes beyond individual coefficients to discuss practical significance, effect sizes, and confidence intervals to convey the scale and uncertainty of the findings.
       1.00: Exceptional insight
           Example: Synthesizes multiple model outputs to generate novel, data-driven insights and hypotheses about the underlying real-world mechanism.

2.  Statistical Rigor and Justification(0-1):
       0.00: No statistical support
           Example: Makes claims about trends or differences without any statistical tests or evidence.
       0.25: Basic application
           Example: Applies a statistical test (e.g., a t-test) without justifying its choice or verifying its assumptions (e.g., normality, equal variances).
       0.50: Standard rigor
           Example: Clearly justifies the choice of statistical methods and correctly checks the key assumptions of the model.
       0.75: Strong rigor
           Example: Not only justifies methods but also discusses their statistical power, potential pitfalls, and why they are superior to alternatives for this specific problem.
       1.00: Exceptional rigor
           Example: Provides a complete and principled justification for the entire inferential framework, possibly including the choice between frequentist and Bayesian approaches, and demonstrates a deep understanding of the underlying statistical theory.

3.  Interpretation of Results and Context(0-1):
       0.00: No interpretation
           Example: Presents a table of results or a plot with no explanation.
       0.25: Literal interpretation
           Example: "The coefficient is 0.8." (Fails to translate this into the problem's domain).
       0.50: Clear contextual interpretation
           Example: "The model indicates that for each additional year of education, income is predicted to increase by $2,000, holding other factors constant."
       0.75: Thorough interpretation with limitations
           Example: Clearly distinguishes between correlation and causation, discusses the scope of the findings, and uses uncertainty measures (e.g., confidence intervals) to provide a realistic range for the estimates.
       1.00: Exceptional interpretation
           Example: Delivers a nuanced narrative that weaves together statistical findings, domain knowledge, and potential confounding factors, clearly articulating what the model does and does not reveal about the real world.

4.  Critical Evaluation of Findings(0-1):
       0.00: No critical thinking
           Example: Accepts all model outputs as absolute truth without question.
       0.25: Basic criticism
           Example: Mentions an obvious limitation (e.g., "the sample size was small") without discussing its specific impact on the results.
       0.50: Standard analysis of limitations
           Example: Identifies the key assumptions and data limitations (e.g., potential sampling bias, measurement error) and how they might affect the conclusions.
       0.75: Strong analysis with sensitivity checks
           Example: Actively investigates the robustness of the findings by performing sensitivity analysis (e.g., re-running the model with different assumptions or subsets of data).
       1.00: Exceptional critique
           Example: Provides a comprehensive critique of the entire analysis, discusses alternative explanations for the results, and realistically appraises the level of confidence one should have in the conclusions.

5.  Implications and Future Directions(0-1):
       0.00: No discussion
           Example: The paper ends abruptly after presenting the results.
       0.25: Basic implications
           Example: Vague suggestions like "more research is needed."
       0.50: Clear implications
           Example: Suggests reasonable next steps, such as collecting a specific type of new data to address a key model limitation.
       0.75: Strong implications
           Example: Proposes specific, well-designed future studies (e.g., an A/B test or a longitudinal study) to validate the current findings and test for causality.
       1.00: Exceptional vision
           Example: Articulates a novel and impactful research agenda or policy/business recommendation that is directly and justifiably inspired by the model's insights and limitations.

---

Your response must follow this exact format:

Your Response:
```json
{
    "depth_of_statistical_analysis": {
        "score": 0.0,
        "explanation": "Detailed justification for score"
    },
    "statistical_rigor_and_justification": {
        "score": 0.0,
        "explanation": "Detailed justification for score"
    },
    "interpretation_of_results_and_context": {
        "score": 0.0,
        "explanation": "Detailed justification for score"
    },
    "critical_evaluation_of_findings": {
        "score": 0.0,
        "explanation": "Detailed justification for score"
    },
    "implications_and_future_directions": {
        "score": 0.0,
        "explanation": "Detailed justification for score"
    },
    "calculated_overall": 0.0,
    "overall_feedback": "Critical analysis of strengths and weaknesses"
}
```

---

Note: Scores must be exactly 0.00, 0.25, 0.50, 0.75, or 1.00. Start at 0 and justify each increment. Be extremely critical. You should also give your score and explanation from your role's perspective."""

    USER_PROMPT = """Please evaluate the analysis groundedness of the following statistical modeling paper:

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
            
            analysis_aspects = [
            "depth_of_statistical_analysis",
            "statistical_rigor_and_justification",
            "interpretation_of_results_and_context",
            "critical_evaluation_of_findings",
            "implications_and_future_directions"
            ]

            scores = [result[aspect]["score"] for aspect in analysis_aspects if aspect in result]
            if scores:
                result["calculated_overall"] = sum(scores) / len(scores)
            else:
                result["calculated_overall"] = 0.0
            result["role"] = role
            
            return result