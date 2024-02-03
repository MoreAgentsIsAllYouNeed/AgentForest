import sys
import os
import re
import glob
import pandas as pd
from utils import is_final_answer_correct, is_final_answer_in_ground_truth
from human_eval.data import stream_jsonl

def extract_number(filename):
    match = re.search(r'part_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

def merge_record(log_dir):
    csv_list = glob.glob(os.path.join(log_dir, "*.csv"))
    csv_list = sorted(csv_list, key=extract_number)
    df_list = []
    for csv_file in csv_list:
        if csv_file == log_dir+"/merged_record.csv":
            continue
        try:
            df_list.append(pd.read_csv(csv_file))
        except Exception as e:
            print(f"{csv_file}: Error during merge: {e}")
    final_df = pd.concat(df_list)
    final_df.to_csv(log_dir+"/merged_record.csv")

def merge_human_eval(log_dir):
    merge_record(log_dir)
    df = pd.read_csv(log_dir+"/merged_record.csv")
    result_list = []
    passed_list = []
    jsonl_file = log_dir + "/human-eval_1_agents.jsonl_results.jsonl"
    for idx, sample in enumerate(stream_jsonl(jsonl_file)):
        row_data = df.iloc[idx]
        task_id = sample["task_id"]
        completion = sample["completion"]
        result = sample["result"]
        passed = sample["passed"]
        if row_data["task_id"] == task_id and completion == row_data["final_answer"]:
            result_list.append(result)
            passed_list.append(passed)
        else:
            print("error", row_data["task_id"], task_id)
    assert len(result_list) == len(df)
    assert len(passed_list) == len(df)
    df["passed"] = passed_list
    df["result"] = result_list
    df.to_csv(log_dir+"/merged_record.csv")

def evaluation(log_dir, qtype):
    final_df = pd.read_csv(log_dir+"/merged_record.csv")
    with open(log_dir+"/final_perf.txt", "w") as f:
        if qtype == "chess":
            final_perf = final_df.apply(is_final_answer_in_ground_truth, axis=1).mean()
        else:
            final_perf = final_df.apply(is_final_answer_correct, axis=1).mean()
        print(f"final_perf: {final_perf}", file=f)
    print(f"final_perf: {final_perf}")


if __name__ == "__main__":
    log_dir = sys.argv[1]
    qtype = sys.argv[2]
    if qtype == "human-eval":
        merge_human_eval(log_dir)
    else:
        merge_record(log_dir)
        evaluation(log_dir,qtype)