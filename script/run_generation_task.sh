#!/bin/bash

ROUND_MAX=$1
PART_NUM=$2
SUBSET_NUM=$3
MODEL=$4 # gpt-35-turbo, gpt-4, llama2

AGENT=1
TEMPERATURE=1
TOP_P=1
ROUND_MIN=1

cd ../src
QTYPE="human-eval" # mmlu, math, chess, human-eval, arithmetic-30, arithmetic-100

for ROUND in $(seq $ROUND_MIN $ROUND_MAX)
do
    DIR_NAME="log_${QTYPE}_${AGENT}_agents_round_${ROUND}"
    for PART in $(seq 0 $PART_NUM)
    do
        EXP_NAME="${QTYPE}_${AGENT}_agents_part_${PART}"
        python main.py "$PART" "$SUBSET_NUM" "$EXP_NAME" "$MODEL" "$DIR_NAME" "$AGENT" "$QTYPE" "$TEMPERATURE" $TOP_P &
    done
    echo "ROUND ${ROUND} AGENT ${AGENT}: All done, evaluating..."
    wait
    for PART in $(seq 0 $PART_NUM)
    do
        EXP_NAME="${QTYPE}_${AGENT}_agents_part_${PART}"
        cat ${DIR_NAME}/${EXP_NAME}.jsonl >> ${DIR_NAME}/${QTYPE}_${AGENT}_agents.jsonl
    done
    evaluate_functional_correctness ${DIR_NAME}/${QTYPE}_${AGENT}_agents.jsonl > ${DIR_NAME}/${QTYPE}_${AGENT}_agents_perf_result.txt
    python evaluation.py ${DIR_NAME} human-eval
    python merge_human_eval.py
done
