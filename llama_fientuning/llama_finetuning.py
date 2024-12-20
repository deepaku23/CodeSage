import torch
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb
from huggingface_hub import login
import os

class LLMFineTuner:
    def __init__(self, 
                 base_model_name,
                 output_dir="./fine_tuned_model",
                 max_length=1024):
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        self.max_length = max_length
        
        # Initialize configurations
        self.setup_model_config()
        self.load_model_and_tokenizer()

    def setup_model_config(self):
        """Setup the BitsAndBytes configuration for 4-bit quantization"""
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.lora_config = LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

    def load_model_and_tokenizer(self):
        """Load the model and tokenizer"""
        print("Loading model and tokenizer...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=self.bnb_config,
            device_map="auto"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            use_fast=False
        )
        
        # Set padding token
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Apply LoRA
        self.model = get_peft_model(self.model, self.lora_config)
        print("Model and tokenizer loaded successfully")

    def format_example(self, example):
        """Format the input example into the desired structure for fine-tuning."""
        language = example.get('lang', 'Unknown')
        vulnerability = example.get('vulnerability', '')
        scenario = example.get('question', '')
        input_code = example.get('rejected', '')
        corrected_code = example.get('chosen', '')

        formatted_string = f"""
        ### Language:
        {language}

        ### Scenario:
        {scenario}

        ### This is my code:
        ```{language}
        {input_code}
        ```

        ### Task:
        1. Identify and describe the vulnerability in the code. Begin your answer with 'Vulnerability:'.
        2. Rewrite the program to fix the vulnerability. Begin your corrected program with 'Corrected Code:'.

        Vulnerability: {vulnerability}
        Corrected Code: {corrected_code}
        """

        return formatted_string

    def tokenize_function(self, examples):
        """Tokenize the formatted examples from the dataset."""
        # Create formatted strings for the batch
        formatted_examples = [
            self.format_example({
                'lang': lang,
                'vulnerability': vulnerability,
                'question': question,
                'chosen': chosen,
                'rejected': rejected,
            })
            for lang, vulnerability, question, chosen, rejected in zip(
                examples['lang'],
                examples['vulnerability'],
                examples['question'],
                examples['chosen'],
                examples['rejected'],
            )
        ]

        # Tokenize the formatted examples
        tokenized = self.tokenizer(
            formatted_examples,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        # Set the labels
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized

    def load_and_prepare_dataset(self):
        """Load and prepare the CyberNative security dataset"""
        print("Loading CyberNative/Code_Vulnerability_Security_DPO dataset...")
        vulnerability_dataset = load_dataset("CyberNative/Code_Vulnerability_Security_DPO")
        self.dataset = vulnerability_dataset['train']
        
        # Split dataset
        split_dataset = self.dataset.shuffle(seed=42).train_test_split(test_size=0.2, seed=42)
        self.train_dataset = split_dataset["train"]
        self.test_dataset = split_dataset["test"]
        
        print(f"Train size: {len(self.train_dataset)}")
        print(f"Test size: {len(self.test_dataset)}")

    def prepare_datasets(self):
        """Prepare datasets for training"""
        # Specify columns to retain
        columns_to_keep = ['lang', 'vulnerability', 'question', 'chosen', 'rejected']
        
        # Remove unnecessary columns
        columns_to_remove_train = [col for col in self.train_dataset.column_names if col not in columns_to_keep]
        columns_to_remove_test = [col for col in self.test_dataset.column_names if col not in columns_to_keep]

        # Tokenize datasets
        self.train_dataset = self.train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=columns_to_remove_train,
        )
        
        self.test_dataset = self.test_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=columns_to_remove_test,
        )

    def train(self, num_train_epochs=3, per_device_train_batch_size=1):
        """Train the model"""
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=32,
            num_train_epochs=num_train_epochs,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=2,
            load_best_model_at_end=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset
        )

        print("Starting training...")
        trainer.train()
        print("Training completed")

    def save_model(self):
        """Save the fine-tuned model"""
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Model saved to {self.output_dir}")

def main():
    # Get HuggingFace token from environment variable
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if not hf_token:
        raise ValueError("Please set the HUGGINGFACE_TOKEN environment variable")
    
    # Login to HuggingFace
    login(token=hf_token)
    
    # Initialize fine-tuner
    fine_tuner = LLMFineTuner(
        base_model_name="meta-llama/Llama-2-7b-hf",
        output_dir="./security_fine_tuned_model"
    )
    
    # Load and prepare dataset
    fine_tuner.load_and_prepare_dataset()
    fine_tuner.prepare_datasets()
    
    # Train the model
    fine_tuner.train()
    
    # Save the model
    fine_tuner.save_model()

if __name__ == "__main__":
    main()
