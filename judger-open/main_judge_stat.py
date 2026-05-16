import json
import os
import ast
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Set

from structural_coherency import StructuralCoherencyJudger
from modeling_groundedness import ModelingGroundednessJudger
from data_groundedness import DataGroundednessJudger
from analysis_groundedness import AnalysisGroundednessJudger
from innovativeness import InnovativenessJudger
from practical_science import PracticalScienceJudger
from result_bias import ResultandBiasJudger
from judger_utils import load_paper_input
from tqdm import tqdm


def resolve_problem_entry(problem: str, questions: list) -> Dict[str, Any]:
    normalized_problem = problem.strip().lower()
    for question in questions:
        source = str(question.get("source", "")).strip()
        if source.lower() == normalized_problem:
            return question

    available_sources = [str(question.get("source", "")).strip() for question in questions if question.get("source")]
    raise ValueError(
        f"Unknown problem '{problem}'. Available problems include: {', '.join(available_sources[:10])}"
        + (" ..." if len(available_sources) > 10 else "")
    )

class MainJudger:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.judgers = {
            "structural_coherency": StructuralCoherencyJudger(model=model),
            "practical_science": PracticalScienceJudger(model=model),
            "result_bias": ResultandBiasJudger(model=model),
            "modeling_groundedness": ModelingGroundednessJudger(model=model),
            "data_groundedness": DataGroundednessJudger(model=model),
            "analysis_groundedness": AnalysisGroundednessJudger(model=model),
            "innovativeness": InnovativenessJudger(model=model)
        }
        
        # Judgers that use role-based evaluation
        self.role_based_judgers = {
            "modeling_groundedness",
            "data_groundedness", 
            "analysis_groundedness",
            "innovativeness"
        }

    def run_judger_single(self, judger_name: str, paper_input: Dict[str, Any], role: str= None, grading_points: list = None) -> Dict[str, Any]:
        try:
            judger = self.judgers[judger_name]
            if judger_name in self.role_based_judgers and role:
                results = []
                result = judger.run(paper_input, role=role)
                result["judger_name"] = judger_name
                results.append(result)
                return {
                    "role_based_results": results
                }
            return {"structural_coherency": judger.run(paper_input)}
        
        except Exception as e:
            print(f"Error in {judger_name}: {str(e)}")
            return {
                "error": str(e),
                "status": "failed",
                "judger": judger_name
            }
        
    def judge(self, order: list, paper_input: Dict[str, Any], roles: list = None, output_file: str = None) -> Dict[str, Any]:
        results = []

        for index, judger_name in tqdm(enumerate(order), total=len(order)):
            print(f" \n Running {judger_name}...")
            if judger_name in self.role_based_judgers and roles:
                role = roles[index -1]
                res = self.run_judger_single(judger_name, paper_input, role=role)
            else:
                res = self.run_judger_single(judger_name, paper_input)
            result = {"judger_name": judger_name, "result": res}
            results.append(result)
            # store after each judger
            print(f"Completed {judger_name}")
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

        return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pos_model", nargs="?", help="Legacy positional model argument")
    parser.add_argument("pos_problem", nargs="?", help="Legacy positional problem argument")
    parser.add_argument("pos_paper_path", nargs="?", help="Legacy positional paper path argument")
    parser.add_argument(
        "pos_paper_type",
        nargs="?",
        choices=["auto", "text", "md", "markdown", "txt", "pdf"],
        help="Legacy positional paper type argument",
    )
    parser.add_argument('--model', type=str, default=None, help='Model to use for judging')
    parser.add_argument('--problem', type=str, default=None, help='Problem identifier')
    parser.add_argument('--paper_path', type=str, default=None, help='Path to the paper file (.md/.txt/.pdf)')
    parser.add_argument('--paper_type', type=str, default=None, choices=["auto", "text", "md", "markdown", "txt", "pdf"], help='Paper input type. Use auto to infer from file extension.')
    args = parser.parse_args()

    model = args.model or args.pos_model or "gpt-4o-mini"
    problem = args.problem or args.pos_problem or "cumcm2012c"
    paper_path = args.paper_path or args.pos_paper_path
    paper_type = args.paper_type or args.pos_paper_type or "auto"

    if not paper_path:
        parser.error("paper_path is required. Use --paper_path PATH or provide it as the third positional argument.")

    rt_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paper_input = load_paper_input(paper_path, paper_type)
    judger = MainJudger(model=model)
    order = [
        "structural_coherency", 
        "modeling_groundedness", 
        "data_groundedness", 
        "analysis_groundedness", 
        "innovativeness",
        "practical_science",
        "result_bias"
    ]
    ques_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/stat-open.json"
    with open(ques_path, "r", encoding='utf-8') as f:
        output = json.load(f)
    problem_entry = resolve_problem_entry(problem, output)
    resolved_problem = problem_entry["source"]
    output_file = f"{rt_dir}/output/{resolved_problem}_scores.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    roles = problem_entry["role"]
    results = judger.judge(order=order, paper_input=paper_input, roles=roles, output_file=output_file)
