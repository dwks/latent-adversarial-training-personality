#!/bin/sh

SYSPROMPT_SHORT=adam
DATASET_SHORT=D5

for steps in $(seq 10 10 100); do
    # modes: all, train, basic_test, eval
    mode=all
    project=LAT-$DATASET_SHORT-$SYSPROMPT_SHORT-$steps

    echo ======== Running $project: dataset $DATASET_SHORT, sysprompt $SYSPROMPT_SHORT, steps $steps, mode $mode

    time python lat_personality.py \
        --cache-dir cache \
        --data-folder dataset/$DATASET_SHORT/ \
        --project-name $project \
        --system-prompt-file sysprompt/$SYSPROMPT_SHORT.txt \
        --num-steps $steps \
        --mode $mode
done
