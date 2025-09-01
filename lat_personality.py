import os
import torch
import sys
import json
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from latent_at import *
import random
import numpy as np
import argparse
from tasks.harmbench.FastHarmBenchEvals import run_attack_evals, run_general_evals
from lat_basic_test import run_basic_test, run_basic_test_generalization, run_tough_set_test


def set_seed(seed):
    random.seed(seed)                   # Python random module
    np.random.seed(seed)                # NumPy
    torch.manual_seed(seed)             # PyTorch CPU
    torch.cuda.manual_seed(seed)        # PyTorch GPU
    torch.cuda.manual_seed_all(seed)    # All GPUs (if using multi-GPU)
    torch.backends.cudnn.deterministic = True  # Makes CUDA deterministic
    torch.backends.cudnn.benchmark = False

def load_tokenizer(model_name, cache_dir):
    if "Llama-2" in model_name:
        model_type = "llama2"
        tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    elif "Llama-3" in model_name or "llama3" in model_name:
        model_type = "llama3"
        tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    elif "Qwen" in model_name:
        model_type = "qwen"
        tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    elif "zephyr" in model_name or "mistral" in model_name:
        model_type = "zephyr"
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta",cache_dir=cache_dir)
        tokenizer.pad_token_id = tokenizer.unk_token_id
        tokenizer.padding_side = "left"
    elif "Qwen" in model_name:
        model_type = "qwen"
        tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    else:
        print(model_name)
        raise Exception("Unsupported model type.")
    return tokenizer, model_type


def load_model(model_name, cache_dir):
    load_dotenv()
    hf_access_token = os.getenv("HUGGINGFACE_API_KEY")

    if model_name not in ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Meta-Llama-3-8B-Instruct",
                        "LLM-LAT/robust-llama3-8b-instruct",
                        "Qwen/Qwen3-4B", "Qwen/Qwen3-7B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B"]:
        raise Exception("Unsupported model name")

    model_dtype = torch.bfloat16
    device = "cuda"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_access_token,
        torch_dtype=model_dtype,
        cache_dir=cache_dir,
    ).to(device)

    tokenizer, model_type = load_tokenizer(model_name, cache_dir)

    return model, tokenizer, model_type

def load_model_for_inference(model_name, cache_dir, project_path, base_model_only=False):
    model_dtype = torch.bfloat16
    device = "cuda"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=model_dtype,
        device_map="auto",
        cache_dir=cache_dir,
        trust_remote_code=True
    )

    if base_model_only:
        print("Loading base model only")
        model = base_model
    else:
        model = PeftModel.from_pretrained(
            base_model,
            project_path,
            trust_remote_code=True
        ).to(device)

    tokenizer, model_type = load_tokenizer(model_name, cache_dir)

    return model, tokenizer, model_type

def load_lat_dataset(data_folder, tokenizer, model_type, sys_prompt, batch_size, add_eos_token_1=True):
    if model_type == "llama2":  # LLama 2 Chat Formatting
        use_tokenizer_template = True
        custom_prompt_template = None
        custom_completion_template = None
    elif model_type == "llama3":  # LLama 3 chat formatting
        use_tokenizer_template = False
        custom_prompt_template = f"<|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|>" + "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        custom_completion_template = "{completion}"
    elif model_type == "qwen":
        use_tokenizer_template = False
        custom_prompt_template = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n" + "<|im_start|>user\n{prompt}<|im_end|>\n" + "<|im_start|>assistant\n"
        custom_completion_template = "{completion}"
    else:  # Zephyr chat formatting
        sys_prompt = ""
        use_tokenizer_template = False
        custom_prompt_template = "<|user|>\n{prompt}</s> \n <|assistant|>\n"
        custom_completion_template = "{completion}"

    lat_dataset = process_generic_chat_dataset(
        tokenizer,
        data_folder,
        dataset=f"{data_folder}/harmful_trait.csv",
        adv_column="rejected",
        def_column="chosen",
        split="train",
        use_tokenizer_template=use_tokenizer_template,
        system_prompt=sys_prompt,
        custom_prompt_template=custom_prompt_template,
        custom_completion_template=custom_completion_template,
        add_eos_token=add_eos_token_1
    )

    lat_dataloader = DataLoader(
        lat_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=LatentAdversarialTrainingDataCollator(
            tokenizer.pad_token_id,
            truncate_length=2048
        )
    )

    # interleaving supervised finetuning with LAT stabilizes training
    sft_dataset = process_generic_chat_dataset(
        tokenizer,
        data_folder,
        dataset=f"{data_folder}/benign_trait.csv",
        #dataset="LLM-LAT/benign-dataset",
        adv_column="refusal",
        def_column="response",
        split="train",
        use_tokenizer_template=use_tokenizer_template,
        system_prompt=sys_prompt,
        custom_prompt_template=custom_prompt_template,
        custom_completion_template=custom_completion_template,
        add_eos_token=True
    )

    sft_dataloader = DataLoader(
        sft_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=LatentAdversarialTrainingDataCollator(
            tokenizer.pad_token_id,
            truncate_length=2048
        )
    )

    return lat_dataloader, sft_dataloader

def do_lat_training(model, model_type, lat_dataloader, sft_dataloader, project_name, project_path, lat_config):
    if model_type == "llama2":  # use llama2-7b
        adv_loss_coefs = {"toward": 0.5, "away": 0.5,}
        def_loss_coefs = {"sft": 1.5, "toward": 0.5, "away": 0.5,}
        inner_learning_rate = 5e-2
        outer_learning_rate = 2e-5
        epsilon = 6.0
        add_completions_pgd = False
    else: # use llama3-8b
        print("Use llama3-8b")
        adv_loss_coefs = {"toward": 0.5, "away": 0.5,}
        def_loss_coefs = {"kl": 0.1, "toward": 0.5, "away": 0.5,}
        inner_learning_rate = 1e-3
        outer_learning_rate = 8e-5
        epsilon = 6.0
        add_completions_pgd = True

    pgd_trainer = ProjectedGradLAT(
        model=model,  # model
        dataloader=lat_dataloader,  # dataloader for lat
        sft_dataloader=sft_dataloader,  # dataloader for supervised finetuning
        adv_loss_coefs=adv_loss_coefs,  # adversary's loss coefs
        def_loss_coefs=def_loss_coefs,  # model's loss coefs
        pgd_layers=["embedding", 8, 16, 24, 30],  # what layers to attack
        pgd_iterations_per_step=lat_config["pgd_iterations_per_step"],  # how many steps of projected gradient descent to do
        model_layers=list(range(0, model.config.num_hidden_layers)),  # model layers to train
        epsilon=epsilon,  # attack l2 constraint
        inner_learning_rate=inner_learning_rate,  # adversary lr
        outer_learning_rate=outer_learning_rate,  # model lr
        model_iterations_per_step=lat_config["model_iterations_per_step"],  # how many times to train on each step
        num_steps=lat_config["num_steps"],  # number of epochs
        max_batch_per_acc=lat_config["max_batch_per_acc"],  # max size of a minibatch
        only_train_lora=True,  # train using low rank adapters
        l2_regularization=lat_config["l2_regularization"],  # coef for l2 weight regularization
        model_layers_module="base_model.model.model.layers",  # where the model layers are
        reinitialize_dev_optim=lat_config["reinitialize_dev_optim"],  # whether to reinitialize optimizer every lat step,
        add_completions_pgd=add_completions_pgd,  # Whether to add PGD over the completion tokens
    )
    pgd_trainer.train(project_name=project_name)
    # pgd_trainer.model.save_pretrained("/tmp/TEST-TEST")

    print(f"saving to {project_path}")
    pgd_trainer.model.save_pretrained(project_path)

def discard_model_and_clear_memory(model=None, tokenizer=None):
    """
    Properly discard model and clear GPU memory.
    """
    if model is not None:
        # Move model to CPU first to free GPU memory
        if hasattr(model, 'to'):
            model.to('cpu')
        # Delete model object
        del model
    
    if tokenizer is not None:
        del tokenizer

def main():
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="lat_benign_test")
    parser.add_argument("--model-name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--project-name", type=str, required=True)
    parser.add_argument("--data-folder", type=str, required=True)
    parser.add_argument("--test-output-dir", type=str, default="output")
    parser.add_argument("--cache-dir", type=str, default="/network/scratch/l/let/projects/models/")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--system-prompt-file", type=str, default="sysprompt/orig.txt")
    parser.add_argument("--inference-system-prompt-file", type=str, default="sysprompt/orig.txt")

    parser.add_argument("--base-model-only", type=bool, default=False)
    parser.add_argument("--add-eos-token-1", type=bool, default=False)

    parser.add_argument("--max-batch-per-acc", type=int, default=2)
    parser.add_argument("--pgd-iterations-per-step", type=int, default=16)
    parser.add_argument("--model-iterations-per-step", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--reinitialize-dev-optim", type=bool, default=True)
    parser.add_argument("--l2-regularization", type=float, default=0)
    args = parser.parse_args()

    mode = args.mode
    model_name = args.model_name
    project_name = args.project_name
    data_folder = args.data_folder
    cache_dir = args.cache_dir
    batch_size = args.batch_size
    test_output_dir = args.test_output_dir + "/" + project_name

    base_model_only = args.base_model_only
    add_eos_token_1 = args.add_eos_token_1

    lat_config = {
        "max_batch_per_acc": args.max_batch_per_acc,
        "pgd_iterations_per_step": args.pgd_iterations_per_step,
        "model_iterations_per_step": args.model_iterations_per_step,
        "num_steps": args.num_steps,
        "reinitialize_dev_optim": args.reinitialize_dev_optim,
        "l2_regularization": args.l2_regularization,
    }

    os.makedirs(test_output_dir, exist_ok=True)

    project_path = cache_dir + "/" + project_name

    already_trained = False
    if base_model_only or os.path.exists(project_path):
        already_trained = True
        print(f"NOTE: Project {project_name} already trained, will skip training")

    already_ran_eval = False
    if os.path.exists(test_output_dir + "/harmbench_output.json"):
        already_ran_eval = True
        print(f"NOTE: Project {project_name} already ran eval, will skip eval")

    with open(args.system_prompt_file, "r") as f:
        sys_prompt = f.read()
    with open(args.inference_system_prompt_file, "r") as f:
        inference_sys_prompt = f.read()

    with open(test_output_dir + "/config.json", "w") as f:
        json.dump({"arguments": sys.argv, "config": lat_config}, f)

    if (mode == "train" or mode == "all") and not already_trained:
        print("=== Running training ===")
        model, tokenizer, model_type = load_model(model_name, cache_dir)

        lat_dataloader, sft_dataloader = load_lat_dataset(data_folder, tokenizer, model_type, sys_prompt, batch_size, add_eos_token_1)

        peft_config = LoraConfig(
            r=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        )
        peft_model = get_peft_model(model, peft_config)

        do_lat_training(peft_model, model_type, lat_dataloader, sft_dataloader, project_name, project_path, lat_config)
        discard_model_and_clear_memory(peft_model, tokenizer)

    if (mode == "eval" or mode == "all") and not already_ran_eval:
        print("=== Running harmbench eval ===")
        inference_model, tokenizer, model_type = load_model_for_inference(model_name, cache_dir, project_path, base_model_only)

        harmbench_output = run_attack_evals(inference_model, model_type=model_type, pretrained_cls="llama", do_sample=False)
        with open(test_output_dir + "/harmbench_output.json", "w") as f:
            json.dump(harmbench_output, f)
        discard_model_and_clear_memory(inference_model, tokenizer)

    if mode == "basic_test" or mode == "all":
        print("=== Running basic test ===")
        inference_model, tokenizer, model_type = load_model_for_inference(model_name, cache_dir, project_path, base_model_only)
        run_basic_test(inference_model, tokenizer, model_type, test_output_dir + "/qa_output", inference_sys_prompt, cache_dir)
        discard_model_and_clear_memory(inference_model, tokenizer)

    if mode == "generalization" or mode == "all":
        print("=== Running generalization test ===")
        inference_model, tokenizer, model_type = load_model_for_inference(model_name, cache_dir, project_path, base_model_only)
        run_basic_test_generalization(inference_model, tokenizer, model_type, test_output_dir + "/qa_output", inference_sys_prompt, cache_dir)
        discard_model_and_clear_memory(inference_model, tokenizer)

    if mode == "tough_set" or mode == "all":
        #for dataset in ["toxic-chat", "convabuse", "dices990"]:
        for dataset in ["toxic-chat"]:
            print(f"=== Running tough set {dataset} test ===")
            with open(f"dataset_test/{dataset}_test.jsonl", "r") as f:
                json_data = [json.loads(line) for line in f]
            new_questions = [{
                "question": data["conversation"][0]["content"],
                "label": "harmful" if data["label"] else "benign"
            } for data in json_data[:100]]
            inference_model, tokenizer, model_type = load_model_for_inference(model_name, cache_dir, project_path, base_model_only)
            benign_output = run_tough_set_test(inference_model, tokenizer, model_type=model_type, output_file_prefix=test_output_dir + f"/{dataset}_output", sys_prompt=inference_sys_prompt, cache_dir=cache_dir, new_questions=new_questions)
            discard_model_and_clear_memory(inference_model, tokenizer)

        #with open("dataset_test/toxic-chat_test.jsonl", "r") as f:
        #    json_data = [json.loads(line) for line in f]
        #new_questions = [{"question": data["conversation"][0]["content"], "label": "harmful" if data["label"] else "benign"} for data in json_data]
        #inference_model, tokenizer, model_type = load_model_for_inference(model_name, cache_dir, project_path, base_model_only)
        #benign_output = run_tough_set_test(inference_model, model_type=model_type, output_file_prefix=test_output_dir + "/toxic_chat_output", sys_prompt=inference_sys_prompt, cache_dir=cache_dir, new_questions=new_questions)
        #discard_model_and_clear_memory(inference_model, tokenizer)

    if mode == "benign_eval" or mode == "all":
        print("=== Running benign eval ===")
        inference_model, tokenizer, model_type = load_model_for_inference(model_name, cache_dir, project_path, base_model_only)

        #evals_to_include = ["MMLU", "HellaSwag", "Winogrande", "SciQ", "Lambada", "PIQA"]
        #evals_to_include = ["MMLU", "MT-Bench", "Compliance", "HellaSwag", "Winogrande", "SciQ", "Lambada", "PIQA"]
        evals_to_include = ["MMLU"]
        benign_output = run_general_evals(inference_model, model_type=model_type, evals_to_include=evals_to_include)
        with open(test_output_dir + "/benign_mmlu.json", "w") as f:
            json.dump(benign_output, f)
        discard_model_and_clear_memory(inference_model, tokenizer)

main()
