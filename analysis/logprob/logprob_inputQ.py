"""
Dual Model Log Probability Calculator for Text Input Questions
Calculates log probabilities for both refused and chosen responses from two models

Usage:
    python dual_model_text_logprob.py --input questions.txt  # Uses default models
    python dual_model_text_logprob.py --input questions.txt --model1 "HuggingFaceH4/zephyr-7b-beta" --model2 "meta-llama/Meta-Llama-3-8B-Instruct"
    python dual_model_text_logprob.py --input questions.txt --samples 500 --logprob_method sum

Default Models:
    Model 1: LLM-LAT/robust-llama3-8b-instruct
    Model 2: /network/scratch/l/let/projects/latent-adversarial-training-personality/cache/aLAT-D12-simple-15
    Tokenizer: meta-llama/Meta-Llama-3-8B-Instruct (used for both models)

Input Format:
    Text file with one question per line

Response Format:
    Refused response: "Yes"
    Chosen response: "No"

Requirements:
    pip install torch transformers datasets pandas numpy tqdm matplotlib scipy
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu
from pathlib import Path
import os


@dataclass
class DualModelLogProbResult:
    """Results from dual model log probability calculation."""
    prompt: str
    refused_response: str
    chosen_response: str
    model1_refused_logprob: float
    model1_chosen_logprob: float
    model2_refused_logprob: float
    model2_chosen_logprob: float


def create_dual_model_visualizations(results: List[DualModelLogProbResult],
                                    model1_name: str, model2_name: str,
                                    output_dir: str = "plots") -> None:
    """Create visualizations comparing log probabilities from both models."""

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extract valid log probabilities
    model1_refused = []
    model1_chosen = []
    model2_refused = []
    model2_chosen = []

    for result in results:
        if np.isfinite(result.model1_refused_logprob):
            model1_refused.append(result.model1_refused_logprob)
        if np.isfinite(result.model1_chosen_logprob):
            model1_chosen.append(result.model1_chosen_logprob)
        if np.isfinite(result.model2_refused_logprob):
            model2_refused.append(result.model2_refused_logprob)
        if np.isfinite(result.model2_chosen_logprob):
            model2_chosen.append(result.model2_chosen_logprob)

    model1_short = "LAT"#model1_name.split('/')[-1] if '/' in model1_name else model1_name
    model2_short = "ours"#model2_name.split('/')[-1] if '/' in model2_name else model2_name

    print(f"Creating visualizations:")
    print(f"  Model 1 - Refused ('Yes'): {len(model1_refused)}, Chosen ('No'): {len(model1_chosen)}")
    print(f"  Model 2 - Refused ('Yes'): {len(model2_refused)}, Chosen ('No'): {len(model2_chosen)}")

    # Plot 1: All four distributions on same plot
    if model1_refused and model1_chosen and model2_refused and model2_chosen:
        plt.figure(figsize=(12, 8))

        # Calculate common bins for all data
        all_values = np.concatenate([model1_refused, model1_chosen, model2_refused, model2_chosen])
        bin_edges = np.linspace(all_values.min(), all_values.max(), 50)

        colors = ['red', 'blue', 'orange', 'green']
        alpha = 0.6

        value_lists = [model1_refused, model1_chosen, model2_refused, model2_chosen]
        labels = [f"{model1_short} - 'Yes'", f"{model1_short} - 'No'",
                 f"{model2_short} - 'Yes'", f"{model2_short} - 'No'"]

        patches = []
        stats_lines = []

        # Calculate average negative log loss for correct answers ("No" responses)
        model1_avg_neg_log_loss = -np.mean(model1_chosen) if model1_chosen else float('inf')
        model2_avg_neg_log_loss = -np.mean(model2_chosen) if model2_chosen else float('inf')

        for i, (values, label) in enumerate(zip(value_lists, labels)):
            color = colors[i]

            # Plot histogram
            plt.hist(values, bins=bin_edges, alpha=alpha, label=f'{label} (n={len(values)})',
                     color=color, edgecolor='black', linewidth=0.5)

            # Add mean and median lines
            mean_val = np.mean(values)
            median_val = np.median(values)

            plt.axvline(mean_val, color=color, linestyle='--', linewidth=2, alpha=0.8)
            plt.axvline(median_val, color=color, linestyle='-', linewidth=2, alpha=0.8)

            # Create legend patch
            patches.append(mpatches.Patch(color=color, alpha=alpha, label=f'{label} (n={len(values)})'))

            # Collect statistics
            stats_lines.extend([
                f"{label} Mean: {mean_val:.4f}",
                f"{label} Median: {median_val:.4f}"
            ])

        # Add negative log loss statistics
        # stats_lines.extend([
        #     f"{model1_short} Avg Neg Log Loss: {model1_avg_neg_log_loss:.4f}",
        #     f"{model2_short} Avg Neg Log Loss: {model2_avg_neg_log_loss:.4f}"
        # ])

        plt.title("Log Probability Comparison: All Models and Response Types", fontsize=14, fontweight='bold')
        plt.xlabel("Log Probability", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)

        # Position legend to avoid overlap with statistics box
        plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(0.72, 0.85))

        # Add statistics text box (limit to avoid overcrowding)
        stats_text = "\n".join(stats_lines[:10])  # Increased to include neg log loss
        plt.gca().text(0.02, 0.98, stats_text,
                       transform=plt.gca().transAxes,
                       fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        plt.savefig(Path(output_dir) / "all_models_comparison.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_dir}/all_models_comparison.pdf")

    # Plot 2: Model differences scatter plot
    if model1_refused and model2_refused:
        # Create pairs for scatter plot
        valid_pairs_refused = []
        valid_pairs_chosen = []

        for result in results:
            if (np.isfinite(result.model1_refused_logprob) and
                np.isfinite(result.model2_refused_logprob)):
                valid_pairs_refused.append((result.model1_refused_logprob, result.model2_refused_logprob))

            if (np.isfinite(result.model1_chosen_logprob) and
                np.isfinite(result.model2_chosen_logprob)):
                valid_pairs_chosen.append((result.model1_chosen_logprob, result.model2_chosen_logprob))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Scatter plot for refused responses ('Yes')
        if valid_pairs_refused:
            model1_ref, model2_ref = zip(*valid_pairs_refused)
            ax1.scatter(model1_ref, model2_ref, alpha=0.6, s=30, color='red', label="'Yes' Responses")

            # Add diagonal line
            min_val = min(min(model1_ref), min(model2_ref))
            max_val = max(max(model1_ref), max(model2_ref))
            ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2,
                    label='Perfect Correlation')

            correlation_ref = np.corrcoef(model1_ref, model2_ref)[0, 1]
            ax1.set_xlabel(f'{model1_short} - "Yes" Log Probability', fontsize=12)
            ax1.set_ylabel(f'{model2_short} - "Yes" Log Probability', fontsize=12)
            ax1.set_title(f'"Yes" Responses\nCorrelation: {correlation_ref:.4f}', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Scatter plot for chosen responses ('No')
        if valid_pairs_chosen:
            model1_cho, model2_cho = zip(*valid_pairs_chosen)
            ax2.scatter(model1_cho, model2_cho, alpha=0.6, s=30, color='blue', label="'No' Responses")

            # Add diagonal line
            min_val = min(min(model1_cho), min(model2_cho))
            max_val = max(max(model1_cho), max(model2_cho))
            ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2,
                    label='Perfect Correlation')

            correlation_cho = np.corrcoef(model1_cho, model2_cho)[0, 1]
            ax2.set_xlabel(f'{model1_short} - "No" Log Probability', fontsize=12)
            ax2.set_ylabel(f'{model2_short} - "No" Log Probability', fontsize=12)
            ax2.set_title(f'"No" Responses\nCorrelation: {correlation_cho:.4f}', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(Path(output_dir) / "model_correlation_scatter.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_dir}/model_correlation_scatter.pdf")

    # Plot 3: Difference analysis
    if model1_refused and model1_chosen and model2_refused and model2_chosen:
        plt.figure(figsize=(14, 6))

        # Calculate differences (Chosen - Refused) for each model
        # Note: Now this is (No - Yes)
        model1_differences = []
        model2_differences = []

        for result in results:
            if (np.isfinite(result.model1_refused_logprob) and
                np.isfinite(result.model1_chosen_logprob)):
                model1_differences.append(result.model1_chosen_logprob - result.model1_refused_logprob)

            if (np.isfinite(result.model2_refused_logprob) and
                np.isfinite(result.model2_chosen_logprob)):
                model2_differences.append(result.model2_chosen_logprob - result.model2_refused_logprob)

        if model1_differences and model2_differences:
            # Calculate common bins
            all_diffs = np.concatenate([model1_differences, model2_differences])
            bin_edges = np.linspace(all_diffs.min(), all_diffs.max(), 40)

            plt.hist(model1_differences, bins=bin_edges, alpha=0.7, color='blue',
                    label=f'{model1_short}', edgecolor='black', linewidth=0.5)
            plt.hist(model2_differences, bins=bin_edges, alpha=0.7, color='red',
                    label=f'{model2_short}', edgecolor='black', linewidth=0.5)

            # Add mean lines
            mean1 = np.mean(model1_differences)
            mean2 = np.mean(model2_differences)
            plt.axvline(mean1, color='blue', linestyle='--', linewidth=2, alpha=0.8)
            plt.axvline(mean2, color='red', linestyle='--', linewidth=2, alpha=0.8)
            plt.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.5, label='Zero Difference')

            plt.title('Model Preference: "No" vs "Yes" ("No" - "Yes")', fontsize=14, fontweight='bold')
            plt.xlabel('Log Probability Difference ("No" - "Yes")', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Add statistics including negative log loss
            if len(model1_differences) == len(model2_differences):
                stat, pval = mannwhitneyu(model1_differences, model2_differences, alternative="two-sided")
                pval_str = f"<0.001" if pval < 0.001 else f"{pval:.4f}"

                # Calculate negative log loss for both models
                model1_neg_log_loss = -np.mean(model1_chosen) if model1_chosen else float('inf')
                model2_neg_log_loss = -np.mean(model2_chosen) if model2_chosen else float('inf')

                stats_text = f"""{model1_short} Mean: {mean1:.4f}
{model2_short} Mean: {mean2:.4f}
Mann-Whitney p: {pval_str}
Difference: {mean2-mean1:.4f}
{model1_short} Neg Log Loss: {model1_neg_log_loss:.4f}
{model2_short} Neg Log Loss: {model2_neg_log_loss:.4f}"""

                plt.gca().text(0.02, 0.98, stats_text,
                              transform=plt.gca().transAxes,
                              fontsize=10,
                              verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            plt.tight_layout()
            plt.savefig(Path(output_dir) / "preference_differences.pdf", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {output_dir}/preference_differences.pdf")

    print(f"\nAll visualizations saved to: {output_dir}/")


class DualModelTextLogProbCalculator:
    """Calculator for log probabilities from two models on text input questions."""

    def __init__(self, model1_name: str, model2_name: str, device: str = "cuda",
                 logprob_method: str = "sum"):
        self.device = device
        self.model1_name = model1_name
        self.model2_name = model2_name
        self.cache_dir = "/network/scratch/l/let/projects/models"
        self.logprob_method = logprob_method.lower()

        # Validate method
        if self.logprob_method not in ["average", "sum"]:
            print(f"Warning: Invalid logprob_method '{logprob_method}', defaulting to 'sum'")
            self.logprob_method = "sum"

        print(f"Loading Model 1: {model1_name}")
        print(f"Loading Model 2: {model2_name}")
        print(f"Log probability method: {self.logprob_method.upper()} per token")

        # Load first model
        print("Loading Model 1...")
        self.tokenizer1 = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",  # Use same tokenizer for both models
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )

        self.model1 = AutoModelForCausalLM.from_pretrained(
            model1_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            cache_dir=self.cache_dir
        )

        if device == "cpu":
            self.model1 = self.model1.to("cpu")

        # Set padding token for model 1
        if self.tokenizer1.pad_token is None:
            self.tokenizer1.pad_token = self.tokenizer1.eos_token

        # Configure chat template for model 1
        self.use_chat_template1 = hasattr(self.tokenizer1, 'apply_chat_template') and self.tokenizer1.chat_template is not None
        if self.use_chat_template1:
            print("Model 1: Using chat template")

        self.model1.eval()

        # Load second model
        print("Loading Model 2...")
        self.tokenizer2 = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",  # Use same tokenizer for both models
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )

        self.model2 = AutoModelForCausalLM.from_pretrained(
            model2_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            cache_dir=self.cache_dir
        )

        if device == "cpu":
            self.model2 = self.model2.to("cpu")

        # Set padding token for model 2
        if self.tokenizer2.pad_token is None:
            self.tokenizer2.pad_token = self.tokenizer2.eos_token

        # Configure chat template for model 2
        self.use_chat_template2 = hasattr(self.tokenizer2, 'apply_chat_template') and self.tokenizer2.chat_template is not None
        if self.use_chat_template2:
            print("Model 2: Using chat template")

        self.model2.eval()

        print("Both models loaded successfully!")

    def format_prompt(self, prompt: str, model_idx: int) -> str:
        """Format prompt using the appropriate model's chat template if available."""
        if model_idx == 1 and self.use_chat_template1:
            try:
                messages = [{"role": "user", "content": prompt}]
                formatted = self.tokenizer1.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return formatted
            except Exception as e:
                print(f"Warning: Could not apply chat template for model 1: {e}")
                return prompt
        elif model_idx == 2 and self.use_chat_template2:
            try:
                messages = [{"role": "user", "content": prompt}]
                formatted = self.tokenizer2.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return formatted
            except Exception as e:
                print(f"Warning: Could not apply chat template for model 2: {e}")
                return prompt
        else:
            return prompt

    def calculate_sequence_logprob(self, prompt: str, continuation: str, model_idx: int) -> float:
        """Calculate log probability of continuation given prompt for specified model."""
        try:
            # Select appropriate model and tokenizer
            if model_idx == 1:
                model = self.model1
                tokenizer = self.tokenizer1
            else:
                model = self.model2
                tokenizer = self.tokenizer2

            # Format prompt if chat template available
            formatted_prompt = self.format_prompt(prompt, model_idx)

            # Tokenize prompt and full sequence
            prompt_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=True)
            full_sequence = formatted_prompt + continuation
            full_tokens = tokenizer.encode(full_sequence, add_special_tokens=True)

            # Get continuation tokens (excluding prompt)
            continuation_tokens = full_tokens[len(prompt_tokens):]

            if len(continuation_tokens) == 0:
                return float('-inf')

            # Convert to tensor and create attention mask
            full_tensor = torch.tensor([full_tokens]).to(model.device)
            attention_mask = torch.ones_like(full_tensor).to(model.device)

            # Get log probabilities
            with torch.no_grad():
                outputs = model(input_ids=full_tensor, attention_mask=attention_mask)
                log_probs = F.log_softmax(outputs.logits, dim=-1)

            # Calculate log probability of continuation
            sum_logprob = 0.0
            valid_token_count = 0

            for i, token_id in enumerate(continuation_tokens):
                # Position in the full sequence
                pos = len(prompt_tokens) + i - 1

                if pos >= 0 and pos < log_probs.size(1):
                    token_logprob = log_probs[0, pos, token_id].item()
                    if torch.isfinite(torch.tensor(token_logprob)):
                        sum_logprob += token_logprob
                        valid_token_count += 1

            # Return final log probability based on method
            if valid_token_count > 0:
                if self.logprob_method == "average":
                    final_logprob = sum_logprob / valid_token_count
                else:  # sum method
                    final_logprob = sum_logprob
            else:
                final_logprob = float('-inf')

            return final_logprob

        except Exception as e:
            print(f"Error calculating log probability for model {model_idx}: {e}")
            return float('-inf')

    def load_questions_from_file(self, input_file: str, num_samples: Optional[int] = None) -> List[Dict]:
        """Load questions from text file where each line is a question."""
        try:
            print(f"Loading questions from: {input_file}")

            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file '{input_file}' not found")

            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Filter out empty lines and strip whitespace
            questions = [line.strip() for line in lines if line.strip()]

            if num_samples is not None:
                total_available = len(questions)
                num_samples = min(num_samples, total_available)
                questions = questions[:num_samples]

            print(f"Loaded {len(questions)} questions")

            formatted_samples = []
            for question in questions:
                formatted_samples.append({
                    'prompt': question,
                    'rejected': "Yes",  # refused response
                    'chosen': "No"      # chosen response
                })

            print(f"Successfully processed {len(formatted_samples)} questions")
            return formatted_samples

        except Exception as e:
            print(f"Error loading questions from file: {e}")
            return []

    def calculate_batch_logprobs(self, samples: List[Dict]) -> List[DualModelLogProbResult]:
        """Calculate log probabilities for batch of samples from both models for both response types."""
        results = []

        print(f"Calculating dual model log probabilities for {len(samples)} questions...")

        for sample in tqdm(samples, desc="Processing questions"):
            prompt = sample['prompt']
            rejected = sample['rejected']  # "Yes" response
            chosen = sample['chosen']      # "No" response

            # Calculate log probabilities for both models and both response types
            model1_refused_logprob = self.calculate_sequence_logprob(prompt, rejected, 1)
            model1_chosen_logprob = self.calculate_sequence_logprob(prompt, chosen, 1)
            model2_refused_logprob = self.calculate_sequence_logprob(prompt, rejected, 2)
            model2_chosen_logprob = self.calculate_sequence_logprob(prompt, chosen, 2)

            result = DualModelLogProbResult(
                prompt=prompt,
                refused_response=rejected,
                chosen_response=chosen,
                model1_refused_logprob=model1_refused_logprob,
                model1_chosen_logprob=model1_chosen_logprob,
                model2_refused_logprob=model2_refused_logprob,
                model2_chosen_logprob=model2_chosen_logprob
            )

            results.append(result)

        return results

    def save_results(self, results: List[DualModelLogProbResult], filename: str = None):
        """Save results to CSV with logprobs and questions."""
        if filename is None:
            model1_safe = "LAT"
            model2_safe = "ours"
            filename = f"dual_model_logprobs_{model1_safe}_vs_{model2_safe}.csv"

        data = []
        for r in results:
            row = {
                'question': r.prompt,
                'model1_yes_logprob': r.model1_refused_logprob,
                'model1_no_logprob': r.model1_chosen_logprob,
                'model2_yes_logprob': r.model2_refused_logprob,
                'model2_no_logprob': r.model2_chosen_logprob,
            }
            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

        # Print summary statistics
        valid_model1_refused = [r.model1_refused_logprob for r in results if np.isfinite(r.model1_refused_logprob)]
        valid_model1_chosen = [r.model1_chosen_logprob for r in results if np.isfinite(r.model1_chosen_logprob)]
        valid_model2_refused = [r.model2_refused_logprob for r in results if np.isfinite(r.model2_refused_logprob)]
        valid_model2_chosen = [r.model2_chosen_logprob for r in results if np.isfinite(r.model2_chosen_logprob)]

        # Calculate average negative log loss (for correct/"No" responses)
        model1_avg_neg_log_loss = -np.mean(valid_model1_chosen) if valid_model1_chosen else float('inf')
        model2_avg_neg_log_loss = -np.mean(valid_model2_chosen) if valid_model2_chosen else float('inf')

        print(f"\nSUMMARY STATISTICS:")
        print(f"Total questions: {len(results)}")
        print(f"\nModel 1 ({self.model1_name}):")
        if valid_model1_refused:
            print(f"  'Yes' Response - Mean: {np.mean(valid_model1_refused):.4f}, Std: {np.std(valid_model1_refused):.4f}")
        if valid_model1_chosen:
            print(f"  'No' Response  - Mean: {np.mean(valid_model1_chosen):.4f}, Std: {np.std(valid_model1_chosen):.4f}")
        print(f"  Average Negative Log Loss: {model1_avg_neg_log_loss:.4f}")

        print(f"\nModel 2 ({self.model2_name}):")
        if valid_model2_refused:
            print(f"  'Yes' Response - Mean: {np.mean(valid_model2_refused):.4f}, Std: {np.std(valid_model2_refused):.4f}")
        if valid_model2_chosen:
            print(f"  'No' Response  - Mean: {np.mean(valid_model2_chosen):.4f}, Std: {np.std(valid_model2_chosen):.4f}")
        print(f"  Average Negative Log Loss: {model2_avg_neg_log_loss:.4f}")


def main():
    """Main function with dual model names as arguments."""
    parser = argparse.ArgumentParser(description='Calculate log probabilities for text input questions using two models')
    parser.add_argument('--input', type=str, required=True,
                        help='Input text file with one question per line')
    parser.add_argument('--model1', type=str, default="LLM-LAT/robust-llama3-8b-instruct",
                        help='First HuggingFace model name (default: "LLM-LAT/robust-llama3-8b-instruct")')
    parser.add_argument('--model2', type=str, default="/network/scratch/l/let/projects/latent-adversarial-training-personality/cache/aLAT-D12-adam2-11/",
                        help='Second HuggingFace model name (default: local aLAT model)')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of samples to process (default: 100, use -1 for all)')
    parser.add_argument('--device', type=str, default="cuda",
                        help='Device to use: cuda or cpu (default: cuda)')
    parser.add_argument('--logprob_method', type=str, choices=['average', 'sum'], default='sum',
                        help='Method for calculating log probabilities: average or sum (default: sum)')
    parser.add_argument('--output', type=str, default="results/logprob/output.csv",
                        help='Output CSV filename')

    args = parser.parse_args()

    # Validate device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Handle all samples case
    num_samples = None if args.samples == -1 else args.samples

    print("=" * 80)
    print("DUAL MODEL LOG PROBABILITY CALCULATOR - TEXT INPUT")
    print("=" * 80)
    print(f"Input file: {args.input}")
    print(f"Model 1: {args.model1}")
    print(f"Model 2: {args.model2}")
    print(f"Device: {args.device}")
    print(f"Samples: {'All' if num_samples is None else num_samples}")
    print(f"Log probability method: {args.logprob_method.upper()}")
    print(f"Refused response: 'Yes'")
    print(f"Chosen response: 'No'")
    print("=" * 80)

    # Initialize calculator
    calculator = DualModelTextLogProbCalculator(
        model1_name=args.model1,
        model2_name=args.model2,
        device=args.device,
        logprob_method=args.logprob_method
    )

    # Load questions from text file
    samples = calculator.load_questions_from_file(args.input, num_samples)
    if not samples:
        print("No questions loaded. Exiting.")
        return

    # Calculate log probabilities
    results = calculator.calculate_batch_logprobs(samples)

    # Save results
    calculator.save_results(results, args.output)

    # Create visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    # Create output directory for plots
    plot_dir = "plots"
    if args.output:
        # Use same directory as output file
        plot_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "plots"

    create_dual_model_visualizations(
        results=results,
        model1_name=args.model1,
        model2_name=args.model2,
        output_dir=plot_dir
    )

    # Show some examples
    print("\n" + "=" * 70)
    print("SAMPLE RESULTS")
    print("=" * 70)

    for i, result in enumerate(results):
        print(f"\nSample {i + 1}:")
        print(f"Question: {result.prompt}...")
        print(f"Model 1 - 'Yes': {result.model1_refused_logprob:.4f}, 'No': {result.model1_chosen_logprob:.4f}")
        print(f"Model 2 - 'Yes': {result.model2_refused_logprob:.4f}, 'No': {result.model2_chosen_logprob:.4f}")

        # Show preference (higher logprob = more likely)
        model1_prefers = "'No'" if result.model1_chosen_logprob > result.model1_refused_logprob else "'Yes'"
        model2_prefers = "'No'" if result.model2_chosen_logprob > result.model2_refused_logprob else "'Yes'"
        print(f"Preferences - Model 1: {model1_prefers}, Model 2: {model2_prefers}")

    print("\n" + "=" * 70)
    print("DUAL MODEL ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"Results saved to: {args.output}")
    print(f"Plots saved to: {plot_dir}/")
    print("Generated plots:")
    print("  - all_models_comparison.pdf (All distributions on same plot)")
    print("  - model_correlation_scatter.pdf (Correlation analysis)")
    print("  - preference_differences.pdf (Model preferences: 'No' - 'Yes')")
    print("=" * 70)


if __name__ == "__main__":
    main()