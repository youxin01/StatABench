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
from tqdm import tqdm

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

    def run_judger_single(self, judger_name: str, writing: str, role: str= None, grading_points: list = None) -> Dict[str, Any]:
        try:
            judger = self.judgers[judger_name]
            if judger_name in self.role_based_judgers and role:
                results = []
                result = judger.run(writing, role=role)
                result["judger_name"] = judger_name
                results.append(result)
                return {
                    "role_based_results": results
                }
            return {"structural_coherency": judger.run(writing)}
        
        except Exception as e:
            print(f"Error in {judger_name}: {str(e)}")
            return {
                "error": str(e),
                "status": "failed",
                "judger": judger_name
            }
        
    def judge(self, order: list, writing: str, roles: list = None, output_file: str = None) -> Dict[str, Any]:
        results = []

        for index, judger_name in tqdm(enumerate(order), total=len(order)):
            print(f" \n Running {judger_name}...")
            if judger_name in self.role_based_judgers and roles:
                role = roles[index -1]
                res = self.run_judger_single(judger_name, writing, role=role)
            else:
                res = self.run_judger_single(judger_name, writing)
            result = {"judger_name": judger_name, "result": res}
            results.append(result)
            # store after each judger
            print(f"Completed {judger_name}")
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

        return results

def read_md_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="gpt-4o-mini", help='Model to use for judging')
    parser.add_argument('--problem', type=str, default="cumcm2012c", help='Problem identifier')
    parser.add_argument('--paper_path', type=str, default=None, help='Path to the paper markdown file')
    args = parser.parse_args()

    problem = args.problem
    rt_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_file = f"{rt_dir}/output/{problem}_scores.json"
    
    paper = read_md_file(args.paper_path)
    judger = MainJudger(model=args.model)
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
    roles = [question["role"] for question in output if question["source"] == problem][0]
    results = judger.judge(order=order, writing=paper, roles=roles, output_file=output_file)
