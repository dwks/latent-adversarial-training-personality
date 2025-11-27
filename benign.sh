#!/bin/sh

output_dir=evaluation-benign
mkdir $output_dir

model_list="../models/LAT-D12-adam2-11-merged ../models/aLAT-D12-adam2-18-merged ../models/bLAT-D12-simple-15-merged ../models/bLAT-D16-orig-17-merged ../models/cat-llama2-merged ../models/cat-llama3-merged ../models/ori-lat-llama2-merged"

download_list="meta-llama/Meta-Llama-3-8B-Instruct meta-llama/Llama-2-7b-chat-hf LLM-LAT/robust-llama3-8b-instruct"

for model in $model_list; do
    for d in mmlu gsm8k truthfulqa; do
        outfile=$output_dir/simple-$(basename $model)-$d/
        mkdir $outfile
        ./submit_any_eval.sh benign-$(basename $model)-$d lm_eval --model vllm  --model_args pretrained=$model  --tasks $d --batch_size auto --seed 42 --output_path $outfile  --log_samples  --system_instruction '"$(cat sysprompt/simple.txt)"'
    done
done

for model in $download_list; do
    for d in mmlu gsm8k truthfulqa; do
        outfile=$output_dir/simple-$(basename $model)-$d/
        mkdir $outfile
        ./submit_any_eval.sh benign-download-$d lm_eval --model vllm  --model_args pretrained=$model,download_dir=/network/scratch/l/let/projects/models  --tasks $d --batch_size auto --seed 42 --output_path $outfile  --log_samples  --system_instruction '"$(cat sysprompt/simple.txt)"'
    done
done
