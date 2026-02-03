from eval_utils import extract_file,evaluate_file

def eval_main(input_file, response_col, ex_col=None, match_col=None):
    if ex_col is None:
        ex_col = f"{response_col}_extracted"
    print(f"Extracting answers to column: {ex_col}")
    extract_file(input_file, response_col, ex_col)

    print(f"Evaluating answers, storing results in column: {match_col if match_col else f'{response_col}_match'}")
    evaluate_file(input_file, ex_col, match_col)

if __name__ == "__main__":
    input_file = "./test/autogen2.json"
    response_col = "autogen_deepseek"
    ex_col = "autogen_deepseek_extracted"
    match_col = "autogen_deepseek_match"

    eval_main(input_file, response_col, ex_col, match_col)