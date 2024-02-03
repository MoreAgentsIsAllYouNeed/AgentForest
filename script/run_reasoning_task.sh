#!/bin/bash

AGENT=$1
PART_NUM=$2
SUBSET_NUM=$3
MODEL=$4 # gpt-35-turbo, gpt-4, llama2
QTYPE=$5 # mmlu, math, chess, human-eval, gsm

TEMPERATURE=1 # 0.3 0.7
TOP_P=1 # 0.95,0.9

cd ../src
DIR_NAME="log_${QTYPE}_${AGENT}_agents"
for PART in $(seq 0 "$PART_NUM")
do
    EXP_NAME="${QTYPE}_${AGENT}_agents_part_${PART}"
    python main.py "$PART" "$SUBSET_NUM" "$EXP_NAME" "$MODEL" "$DIR_NAME" "$AGENT" "$QTYPE" "$TEMPERATURE" "$TOP_P" &
done
wait
echo "AGENT ${AGENT}: All done, evaluating..."
python evaluation.py ${DIR_NAME} ${QTYPE}
