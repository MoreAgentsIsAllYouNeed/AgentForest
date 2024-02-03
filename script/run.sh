#!/bin/bash

AGENT=$1
MODEL=$2 # gpt-35-turbo, gpt-4, llama2
QTYPE=$3 # mmlu, math, chess, human-eval, gsm


if [ "${QTYPE}" = "human-eval" ]; then
    sh run_genration_task.sh $AGENT 1 100 $MODEL
else
    sh run_reasoning_task.sh $AGENT 1 100 $MODEL $QTYPE
fi