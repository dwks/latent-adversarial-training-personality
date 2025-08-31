import os
import json
import torch
import argparse
from typing import List, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import gc

class AnswerClassifier:
    """
    Classifies answers as good or bad using Llama 3 model with few-shot learning.
    """
    
    def __init__(self, system_prompt: str,
            model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct", 
            cache_dir: str = "/tmp/cache-dwk/"):
        """
        Initialize the classifier with Llama 3 model.
        
        Args:
            model_name: HuggingFace model name
            cache_dir: Directory to cache model files
        """
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load environment variables
        load_dotenv()
        self.hf_token = os.getenv("HUGGINGFACE_API_KEY")
        
        # Load model and tokenizer
        self._load_model()
        
        # Training examples
        self.good_examples = []
        self.bad_examples = []
        
    def _load_model(self):
        """Load the Llama 3 model and tokenizer."""

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            token=self.hf_token
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=self.hf_token,
            torch_dtype=torch.bfloat16,
            cache_dir=self.cache_dir,
            device_map="auto"
        )
    
    def load_examples_from_json(self, good_files: List[str], bad_files: List[str]):
        """
        Load training examples from JSON files.
        
        Args:
            good_files: List of paths to JSON files with good examples
            bad_files: List of paths to JSON files with bad examples
        """
        
        # Load good examples
        for file_path in good_files:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if 'question' in data and 'answer' in data:
                                self.good_examples.append({
                                    'question': data['question'],
                                    'answer': data['answer'].replace('\\n', '\n').replace('\\"', '"')
                                })
                        except json.JSONDecodeError:
                            continue
        
        # Load bad examples
        for file_path in bad_files:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if 'question' in data and 'answer' in data:
                                self.bad_examples.append({
                                    'question': data['question'],
                                    'answer': data['answer'].replace('\\n', '\n').replace('\\"', '"')
                                })
                        except json.JSONDecodeError:
                            continue
    
    def create_classification_prompt(self, question: str, answer: str, 
                                   num_good_examples: int = 3, 
                                   num_bad_examples: int = 3) -> str:
        """
        Create a few-shot classification prompt.
        
        Args:
            question: The question being asked
            answer: The answer to classify
            num_good_examples: Number of good examples to include
            num_bad_examples: Number of bad examples to include
            
        Returns:
            Formatted prompt for classification
        """
        # Build few-shot examples
        examples = []
        
        # Add good examples
        for i, example in enumerate(self.good_examples[:num_good_examples]):
            examples.append(f"""Question: {example['question']}
Answer: {example['answer']}
Classification: good""")
        
        # Add bad examples
        for i, example in enumerate(self.bad_examples[:num_bad_examples]):
            examples.append(f"""Question: {example['question']}
Answer: {example['answer']}
Classification: bad""")
        
        # Add the target example
        target_example = f"""Question: {question}
Answer: {answer}
Classification:"""
        
        # Combine everything
        prompt = f"{self.system_prompt}\n\n"
        prompt += "\n\n".join(examples)
        prompt += f"\n\n{target_example}"
        
        return prompt
    
    def classify_answer(self, question: str, answer: str, 
                       max_new_tokens: int = 10) -> Tuple[str, float]:
        """
        Classify an answer as good or bad.
        
        Args:
            question: The question being asked
            answer: The answer to classify
            max_new_tokens: Maximum tokens to generate for classification
            
        Returns:
            Tuple of (classification, confidence_score)
        """
        # Create the classification prompt
        prompt = self.create_classification_prompt(question, answer)
        
        # Tokenize the prompt
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        # Generate classification
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.to(self.device),
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract the classification from the response
        classification_text = response[len(prompt):].strip().lower()
        #print(f"Classification text: [{classification_text}]")
        
        # Determine classification and confidence
        if "good" in classification_text:
            classification = "good"
            confidence = 0.8 if "good" in classification_text[:10] else 0.6
        elif "bad" in classification_text:
            classification = "bad"
            confidence = 0.8 if "bad" in classification_text[:10] else 0.6
        else:
            classification = "unknown"
            confidence = 0.1
            #raise ValueError(f"Invalid classification: [{classification_text}]")
        
        return classification, confidence
    
    def batch_classify(self, examples: List[Dict[str, str]]) -> List[Dict[str, any]]:
        """
        Classify multiple examples in batch.
        
        Args:
            examples: List of dictionaries with 'question' and 'answer' keys
            
        Returns:
            List of dictionaries with classification results
        """
        results = []
        
        for i, example in enumerate(examples):
            #print(f"Classifying example {i+1}/{len(examples)}...")
            
            classification, confidence = self.classify_answer(
                example['question'], 
                example['answer']
            )
            
            results.append({
                'question': example['question'],
                'answer': example['answer'],
                'classification': classification,
                'confidence': confidence
            })
        
        return results
    
    def cleanup(self):
        """Clean up model and free memory."""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Model cleaned up and memory freed.")
