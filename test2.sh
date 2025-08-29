#!/bin/sh

for dataset in D12; do
    for sysprompt in simple adam adam2 linh1 linh2; do
        #for steps in 15; do
        for steps in $(seq 7 1 20); do
        #for steps in $(seq 5 5 30); do
        #for steps in $(seq 10 10 100); do
            # modes: all, train, basic_test, eval
            mode=all
            project=aLAT-$dataset-$sysprompt-$steps

            echo Submitting $project: dataset $dataset, sysprompt $sysprompt, steps $steps, mode $mode

            ./submit_any.sh $project \
                --cache-dir cache \
                --data-folder dataset/$dataset/ \
                --project-name $project \
                --system-prompt-file sysprompt/$sysprompt.txt \
                --num-steps $steps \
                --mode $mode
        done
    done
done
