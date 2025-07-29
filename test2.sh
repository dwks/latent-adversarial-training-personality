#!/bin/sh

for dataset in D6 D7; do
    for sysprompt in linh2; do
        for steps in $(seq 5 5 30); do
        #for steps in $(seq 10 10 100); do
            # modes: all, train, basic_test, eval
            mode=all
            project=LAT-$dataset-$sysprompt-$steps

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
