import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os
import logging


def load_config(config_path="config.yml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config):
    """Setup logging configuration"""
    log_level = getattr(logging, config['logging']['log_level'].upper())
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config['logging']['log_file']) if config['logging']['log_to_file'] else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_model_and_tokenizer(config):
    """Load model and tokenizer based on config"""
    logger = logging.getLogger(__name__)
    
    model_config = config['model']
    logger.info(f"Loading model: {model_config['name']}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['name'],
        trust_remote_code=model_config['trust_remote_code']
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    
    model = AutoModelForCausalLM.from_pretrained(
        model_config['name'],
        dtype=dtype_map[model_config['dtype']],
        low_cpu_mem_usage=model_config['low_cpu_mem_usage'],
        trust_remote_code=model_config['trust_remote_code']
    )
    
    logger.info("Model loaded successfully")
    return model, tokenizer


def setup_lora(model, config):
    """Setup LoRA configuration"""
    logger = logging.getLogger(__name__)
    lora_config = config['lora']
    
    logger.info(f"Setting up LoRA with r={lora_config['r']}, alpha={lora_config['lora_alpha']}")
    
    lora = LoraConfig(
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config['lora_dropout'],
        target_modules=lora_config['target_modules'],
        bias=lora_config.get('bias', 'none'),
        task_type=lora_config.get('task_type', 'CAUSAL_LM')
    )
    
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    
    return model


def prepare_dataset(config, tokenizer):
    """Prepare and tokenize dataset"""
    logger = logging.getLogger(__name__)
    dataset_config = config['dataset']
    
    logger.info(f"Loading dataset from {dataset_config['train_file']}")
    
    dataset = load_dataset("json", data_files=dataset_config['train_file'])
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples[dataset_config['text_column']],
            truncation=dataset_config['truncation'],
            padding=dataset_config['padding'],
            max_length=dataset_config['max_length'],
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    logger.info("Tokenizing dataset...")
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[dataset_config['text_column']]
    )
    
    logger.info(f"Dataset prepared: {len(dataset['train'])} training samples")
    return dataset


def create_training_arguments(config):
    """Create TrainingArguments from config"""
    training_config = config['training']
    eval_config = config['evaluation']
    
    return TrainingArguments(
        output_dir=training_config['output_dir'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 1),
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        learning_rate=training_config['learning_rate'],
        num_train_epochs=training_config['num_train_epochs'],
        warmup_steps=training_config.get('warmup_steps', 0),
        logging_steps=training_config['logging_steps'],
        save_steps=training_config['save_steps'],
        save_total_limit=training_config.get('save_total_limit', 3),
        eval_strategy=training_config.get('evaluation_strategy', 'no'),
        eval_steps=training_config.get('eval_steps'),
        fp16=training_config['fp16'],
        use_cpu=training_config.get('use_cpu', True),
        optim=training_config.get('optim', 'adamw_torch'),
        weight_decay=training_config.get('weight_decay', 0.0),
        max_grad_norm=training_config.get('max_grad_norm', 1.0),
        lr_scheduler_type=training_config.get('lr_scheduler_type', 'linear'),
        load_best_model_at_end=eval_config.get('load_best_model_at_end', False),
        metric_for_best_model=eval_config.get('metric_for_best_model', 'loss'),
        greater_is_better=eval_config.get('greater_is_better', False),
        report_to=config['logging'].get('report_to', []),
        seed=config.get('seed', 42),
    )


def train(config_path="config.yml"):
    """Main training function"""
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting training process...")
    
    # Set random seed
    torch.manual_seed(config.get('seed', 42))
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Setup LoRA
    model = setup_lora(model, config)
    
    # Prepare dataset
    dataset = prepare_dataset(config, tokenizer)
    
    # Create training arguments
    training_args = create_training_arguments(config)
    
    # Initialize trainer
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {config['training']['output_dir']}")
    trainer.save_model()
    tokenizer.save_pretrained(config['training']['output_dir'])
    
    logger.info("Training completed successfully!")
    
    return trainer


if __name__ == "__main__":
    import sys
    
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yml"
    
    if not os.path.exists(config_file):
        print(f"Error: Config file '{config_file}' not found!")
        print("Usage: python src/train_with_config.py [config.yml]")
        sys.exit(1)
    
    train(config_file)
