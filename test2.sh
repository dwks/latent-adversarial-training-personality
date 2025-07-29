#!/bin/sh

DATASET_SHORT=D5

for sysprompt in orig adam adam2; do
    for steps in $(seq 10 10 100); do
        # modes: all, train, basic_test, eval
        mode=all
        project=LAT-$DATASET_SHORT-$sysprompt-$steps

        echo Submitting $project: dataset $DATASET_SHORT, sysprompt $sysprompt, steps $steps, mode $mode

        ./submit_any.sh $project \
            --cache-dir cache \
            --data-folder dataset/$DATASET_SHORT/ \
            --project-name $project \
            --system-prompt-file sysprompt/$sysprompt.txt \
            --num-steps $steps \
            --mode $mode
    done
done
