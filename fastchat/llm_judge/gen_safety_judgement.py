"""
Usage:
python gen_safety_judgement.py --model-list [LIST-OF-MODEL-ID]
"""
import argparse
import json
import os
import time

from tqdm import tqdm

from fastchat.llm_judge.common import (
    load_questions,
    load_model_answers,
    get_model_list,
)
from fastchat.llm_judge.do_not_answer.evaluator.gpt import construct_message, parse_label
from fastchat.llm_judge.do_not_answer.utils import gpt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="japanese_safety_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    args = parser.parse_args()

    question_file = f"data/{args.bench_name}/question.jsonl"
    answer_dir = f"data/{args.bench_name}/model_answer"

    # Load questions
    questions = load_questions(question_file, None, None)

    # Load answers
    model_answers = load_model_answers(answer_dir)

    if args.model_list is None:
        models = get_model_list(answer_dir)
    else:
        models = args.model_list

    for model in models:
        print(f"Processing {model=}")
        output_file = (
            f"data/{args.bench_name}/model_judgement/{model}.jsonl"
        )
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w") as fout:
            for question in tqdm(questions):
                message = construct_message(question, model_answers, model)
                gpt_response = gpt(message, model="gpt-4")
                if not gpt_response:
                    print(f"Failed to get gpt response for {model=}, {question=}")
                    continue

                question_id = question["question_id"]
                result = {
                    "question_id": question_id,
                    "model": model,
                    "instruction": question["turns"][0],
                    "response": model_answers[model][question_id]["choices"][0]["turns"][0],
                    "judge": "gpt-4",
                    "judgement": gpt_response,
                    "label": parse_label(gpt_response),
                    "tstamp": time.time(),
                }
                fout.write(json.dumps(result) + "\n")
