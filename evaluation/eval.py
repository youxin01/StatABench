from eval_utils import extract_file,evaluate_file

def eval_main(input_file, response_col, ex_col=None, match_col=None):
    if ex_col is None:
        ex_col = f"{response_col}_extracted"
    print(f"Extracting answers to column: {ex_col}")
    extract_file(input_file, response_col, ex_col)

    print(f"Evaluating answers, storing results in column: {match_col if match_col else f'{response_col}_match'}")
    evaluate_file(input_file, ex_col, match_col)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run evaluation processing.")

    parser.add_argument("--input_path", type=str, default="./autogen_deepseek.json",
                        help="The path of the file to be evaluated")
    parser.add_argument("--response_col", type=str, default="autogen_deepseek",
                        help="column need to evaluate")
    parser.add_argument("--ex_col", type=str, default=None,
                    help="column containing extracted result")
    parser.add_argument("--match_col", type=str, default=None,
                    help="column store matched result")
    args = parser.parse_args()

    input_file = args.input_path
    response_col = args.response_col
    if args.ex_col:
        ex_col = args.ex_col
    else:
        ex_col = f"{response_col}_extract"
    if args.match_col:
        match_col = args.match_col
    else:
        match_col = f"{response_col}_match"
    eval_main(input_file, response_col, ex_col, match_col)