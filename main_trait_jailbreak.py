
from tasks.harmbench.HarmBenchTask import HarmBenchTask
import os
import torch
import sys
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from latent_at import *
from tasks.harmbench.HarmBenchTask import HarmBenchTask
from tasks.harmbench.FastHarmBenchEvals import run_attack_evals
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)                   # Python random module
    np.random.seed(seed)                # NumPy
    torch.manual_seed(seed)             # PyTorch CPU
    torch.cuda.manual_seed(seed)        # PyTorch GPU
    torch.cuda.manual_seed_all(seed)    # All GPUs (if using multi-GPU)
    torch.backends.cudnn.deterministic = True  # Makes CUDA deterministic
    torch.backends.cudnn.benchmark = False
set_seed(42)
config_batch_size=1
config_gen_batch_size=1
trait_data_folder="/network/scratch/l/let/projects/latent-adversarial-training/"
os.chdir("../")
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

load_dotenv()
hf_access_token = os.getenv("HUGGINGFACE_API_KEY")
use_llama2 = False
if use_llama2:  # use llama2-7b
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    adv_loss_coefs = {"toward": 0.5, "away": 0.5,}
    def_loss_coefs = {"sft": 1.5, "toward": 0.5, "away": 0.5,}
    inner_learning_rate = 5e-2
    outer_learning_rate = 2e-5
    epsilon = 6.0
    add_completions_pgd = False
else: # use llama3-8b
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    adv_loss_coefs = {"toward": 0.5, "away": 0.5,}
    def_loss_coefs = {"kl": 0.1, "toward": 0.5, "away": 0.5,}
    inner_learning_rate = 1e-3
    outer_learning_rate = 8e-5
    epsilon = 6.0
    add_completions_pgd = True

#####
model_dtype = torch.bfloat16
device = "cuda"
run_start_evals = False

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=hf_access_token,
    torch_dtype=model_dtype,
    cache_dir="/tmp/cache_linh/",
).to(device)

if "Llama-2" in model_name:
    model_type = "llama2"
    tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir="/tmp/cache_linh/")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
elif "Llama-3" in model_name:
    model_type = "llama3"
    tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir="/tmp/cache_linh/")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
elif "zephyr" in model_name or "mistral" in model_name:
    model_type = "zephyr"
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta",cache_dir="/tmp/cache_linh/")
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "left"
else:
    print(model_name)
    raise Exception("Unsupported model type.")

###########################


sys_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
# sys_prompt = "You will be presented with a series of statements about your typical thoughts, feelings, and behaviors. Please indicate the extent to which you agree or disagree with each statement as it applies to you personally, as you are now, not how you wish to be or how others see you. There are no right or wrong answers. Be as honest and accurate as possible. Your first instinct is usually best. If answering a question is incompatible with your values, you can refuse to answer, by saying “I’m sorry I cannot comply with that request”"
if model_type == "llama2":  # LLama 2 Chat Formatting
    use_tokenizer_template = True
    custom_prompt_template = None
    custom_completion_template = None
elif model_type == "llama3":  # LLama 3 chat formatting
    use_tokenizer_template = False
    custom_prompt_template = f"<|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|>" + "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    custom_completion_template = "{completion}"
else:  # Zephyr chat formatting
    sys_prompt = ""
    use_tokenizer_template = False
    custom_prompt_template = "<|user|>\n{prompt}</s> \n <|assistant|>\n"
    custom_completion_template = "{completion}"

lat_dataset = process_generic_chat_dataset(
    tokenizer,
    dataset=f"{trait_data_folder}harmful_trait.csv",
    adv_column="rejected",
    def_column="chosen",
    split="train",
    use_tokenizer_template=use_tokenizer_template,
    system_prompt=sys_prompt,
    custom_prompt_template=custom_prompt_template,
    custom_completion_template=custom_completion_template
)

lat_dataloader = DataLoader(
    lat_dataset,
    batch_size=config_batch_size,
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
    dataset=f"{trait_data_folder}benign_trait.csv",
    adv_column="refusal",
    def_column="response",
    split="train",
    use_tokenizer_template=use_tokenizer_template,
    system_prompt=sys_prompt,
    custom_prompt_template=custom_prompt_template,
    custom_completion_template=custom_completion_template,
    add_eos_token=True
)
# interleaving supervised finetuning with LAT stabilizes training
# sft_dataset = process_generic_chat_dataset(
#     tokenizer,
#     dataset="LLM-LAT/benign-dataset",
#     adv_column="refusal",
#     def_column="response",
#     split="train",
#     use_tokenizer_template=use_tokenizer_template,
#     system_prompt=sys_prompt,
#     custom_prompt_template=custom_prompt_template,
#     custom_completion_template=custom_completion_template,
#     add_eos_token=True
# )
sft_dataloader = DataLoader(
    sft_dataset,
    batch_size=config_batch_size,
    shuffle=True,
    drop_last=True,
    collate_fn=LatentAdversarialTrainingDataCollator(
        tokenizer.pad_token_id,
        truncate_length=2048
    )
)

peft_config = LoraConfig(
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, peft_config)

pgd_trainer = ProjectedGradLAT(
    model=model,  # model
    dataloader=lat_dataloader,  # dataloader for lat
    sft_dataloader=sft_dataloader,  # dataloader for supervised finetuning
    adv_loss_coefs=adv_loss_coefs,  # adversary's loss coefs
    def_loss_coefs=def_loss_coefs,  # model's loss coefs
    pgd_layers=["embedding", 8, 16, 24, 30],  # what layers to attack
    pgd_iterations_per_step=16,  # how many steps of projected gradient descent to do
    model_layers=list(range(0, model.config.num_hidden_layers)),  # model layers to train
    epsilon=epsilon,  # attack l2 constraint
    inner_learning_rate=inner_learning_rate,  # adversary lr
    outer_learning_rate=outer_learning_rate,  # model lr
    model_iterations_per_step=4,  # how many times to train on each step
    num_steps=100,  # number of epochs
    max_batch_per_acc=2,  # max size of a minibatch
    only_train_lora=True,  # train using low rank adapters
    l2_regularization=0,  # coef for l2 weight regularization
    model_layers_module="base_model.model.model.layers",  # where the model layers are
    reinitialize_dev_optim=True,  # whether to reinitialize optimizer every lat step,
    add_completions_pgd=add_completions_pgd,  # Whether to add PGD over the completion tokens
)
project_name="trait_positive_disagree"
pgd_trainer.train(project_name=project_name)
pgd_trainer.model.save_pretrained("/tmp/cache_linh/"+project_name)
