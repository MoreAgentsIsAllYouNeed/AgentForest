import sys
import os
import random
import openai
import json
import pandas as pd
from code_completion_task import CodeCompletion
from mmlu_task import MMLU
from math_task import MATH
from chess_task import CHESS
from gsm_task import GSM8K
from human_eval.data import write_jsonl


PART = int(sys.argv[1])
SUBSET = int(sys.argv[2])
EXP_NAME = sys.argv[3]
MODEL = sys.argv[4]
DIR_NAME = sys.argv[5]
AGENTNUM = int(sys.argv[6])
QUESTION_TYPE = sys.argv[7]
TEMPERATURE = float(sys.argv[8])
TOP_P = float(sys.argv[9])


openai.api_key = os.getenv('OPENAI_KEY')
openai.api_base = os.getenv('OPENAI_IP')
openai.api_type = 'azure'
openai.api_version = '2023-05-15'

def main():
    random.seed(0)
    os.makedirs(DIR_NAME, exist_ok=True)
    if QUESTION_TYPE == "human-eval":
        solver = CodeCompletion(AGENTNUM, MODEL,temperature=TEMPERATURE,top_p=TOP_P)
    elif QUESTION_TYPE == "mmlu":
        solver = MMLU(AGENTNUM, MODEL,temperature=TEMPERATURE,top_p=TOP_P)
    elif QUESTION_TYPE == "math":
        solver = MATH(AGENTNUM, MODEL,temperature=TEMPERATURE,top_p=TOP_P)
    elif QUESTION_TYPE == "chess":
        solver = CHESS(AGENTNUM, MODEL,temperature=TEMPERATURE,top_p=TOP_P)
    elif QUESTION_TYPE == "gsm":
        solver = GSM8K(AGENTNUM, MODEL,temperature=TEMPERATURE,top_p=TOP_P)
    else:
        raise NotImplementedError("Error question type")
    
    results = [] # only for human-eval
    total_prompt_tokens, total_completion_tokens = 0, 0

    total_record = []
    question_datas = solver.get_question_datas()
    for task_id, question_data in enumerate(question_datas):
        if task_id < PART*SUBSET or task_id >= (PART+1)*SUBSET:
            continue
        print("current task_id start: ", task_id, flush=True)
        result_dict = solver.forward(question_data)
        one_record = {}
        for k, v in question_data.items():
            one_record[k] = v
        for k, v in result_dict.items():
            if isinstance(v, list):
                for i, sub_v in enumerate(v):
                    new_k = k + f"_{i}"
                    one_record[new_k] = sub_v
            else:
                one_record[k] = v
        total_record.append(one_record)
        if QUESTION_TYPE == "human-eval":
            results.append({"task_id": question_data["task_id"], "completion": result_dict["final_answer"]})
        total_prompt_tokens += result_dict["total_prompt_tokens"]
        total_completion_tokens += result_dict["total_completion_tokens"]
        print("current task_id end: ", task_id, flush=True)
        if QUESTION_TYPE != "human-eval":
            tmp_df = pd.DataFrame(total_record)
            perf = solver.evaluation(tmp_df)
            final_answer = result_dict["final_answer"]
            ground_truth = question_data["ground_truth"]
            print(f"final_res: {final_answer}, ground_truth: {ground_truth}, perf: {perf}")
        with open(DIR_NAME+'/'+EXP_NAME+'.json', 'a') as f:
            f.write(json.dumps(one_record) + '\n')
    print("************************")
    print(f"prompt_token: {total_prompt_tokens}, completion_tokens: {total_completion_tokens}")
    
    df = pd.DataFrame(total_record)
    df.to_csv(DIR_NAME+"/"+EXP_NAME+".csv", index=False)
    if QUESTION_TYPE == "human-eval":
        write_jsonl(DIR_NAME+'/'+EXP_NAME+'.jsonl', results)
    else:
        print(f"part {PART} final evaluation: ", solver.evaluation(df))


if __name__ == "__main__":
    main()
