# train_with_config.py - Comprehensive Documentation

## Overview
**Purpose**: Main training script that fine-tunes Qwen2-1.5B model using LoRA (Low-Rank Adaptation) with YAML-based configuration.

**What it does**: Loads configuration, prepares model with LoRA adapters, tokenizes dataset, and executes training loop.

**Why it exists**: Provides production-ready, configurable training pipeline without hardcoding parameters.

**How it works**: Uses Hugging Face Transformers + PEFT library to apply parameter-efficient fine-tuning on CPU.

---

## Benefits
✅ **Flexibility**: All parameters in config.yml - no code changes needed  
✅ **Parameter Efficiency**: Only trains ~0.56% of model parameters (LoRA)  
✅ **CPU Compatible**: Works without GPU (float32 precision)  
✅ **Production Ready**: Comprehensive logging, error handling, checkpoint saving  
✅ **Memory Efficient**: Low memory usage optimizations enabled  

---

## Trade-offs
⚠️ **Speed**: CPU training is slower than GPU  
⚠️ **Limited Capacity**: LoRA trains fewer parameters than full fine-tuning  
⚠️ **Config Dependency**: Requires valid YAML configuration file  
⚠️ **Disk Space**: Saves multiple checkpoints (configurable)  

---

## Line-by-Line Commented Code

```python
# ============================================================================
# IMPORTS SECTION
# ============================================================================

import yaml
# WHAT: YAML parser for reading configuration files
# WHY: Human-readable config format (config.yml)
# BENEFIT: Easy parameter tuning without code changes
# TRADE-OFF: Requires additional dependency (pyyaml)

import torch
# WHAT: PyTorch deep learning framework
# WHY: Foundation for model training and tensor operations
# BENEFIT: Industry standard, extensive ecosystem
# TRADE-OFF: Large dependency (~2GB)

from transformers import AutoModelForCausalLM, AutoTokenizer
# WHAT: Hugging Face classes for loading pre-trained models
# WHY: AutoModelForCausalLM - For text generation tasks (GPT-style)
#      AutoTokenizer - Converts text to token IDs
# BENEFIT: Abstracts model architecture complexity
# TRADE-OFF: Downloads large model files on first use

from transformers import TrainingArguments, Trainer
# WHAT: Training infrastructure from Hugging Face
# WHY: TrainingArguments - Configures training hyperparameters
#      Trainer - Handles training loop, optimization, logging
# BENEFIT: Battle-tested training pipeline, handles edge cases
# TRADE-OFF: Less control over low-level training details

from peft import LoraConfig, get_peft_model
# WHAT: Parameter-Efficient Fine-Tuning library
# WHY: LoraConfig - Configures LoRA adapter parameters
#      get_peft_model - Injects LoRA layers into model
# BENEFIT: Train large models with minimal memory (0.56% params)
# TRADE-OFF: Slightly lower performance vs full fine-tuning

from datasets import load_dataset
# WHAT: Hugging Face datasets library
# WHY: Efficient data loading and preprocessing
# BENEFIT: Handles JSONL, CSV, parquet - optimized for ML
# TRADE-OFF: Different API than native Python file operations

import os
# WHAT: Operating system interface
# WHY: File path validation, directory operations
# BENEFIT: Cross-platform compatibility
# TRADE-OFF: None (standard library)

import logging
# WHAT: Python logging framework
# WHY: Track training progress, debug issues
# BENEFIT: Professional logging with levels (INFO, DEBUG, ERROR)
# TRADE-OFF: Requires setup configuration


# ============================================================================
# CONFIGURATION LOADING FUNCTION
# ============================================================================

def load_config(config_path="config.yml"):
    """Load configuration from YAML file"""
    # WHAT: Reads and parses YAML configuration file
    # WHY: Centralize all settings in one file
    # HOW: Opens file, parses YAML to Python dict
    # BENEFIT: Single source of truth for configuration
    # TRADE-OFF: Must ensure YAML syntax is valid
    
    with open(config_path, 'r') as f:
        # WHAT: Opens config file in read mode
        # WHY: Context manager ensures file closes properly
        # BENEFIT: Prevents file handle leaks
        # TRADE-OFF: Raises FileNotFoundError if missing
        
        config = yaml.safe_load(f)
        # WHAT: Parses YAML content to Python dictionary
        # WHY: safe_load prevents arbitrary code execution
        # BENEFIT: Secure parsing (vs yaml.load)
        # TRADE-OFF: Can't load custom Python objects
        
    return config
    # Returns: dict with nested configuration structure


# ============================================================================
# LOGGING SETUP FUNCTION
# ============================================================================

def setup_logging(config):
    """Setup logging configuration"""
    # WHAT: Configures Python logging system
    # WHY: Track training progress and debug issues
    # HOW: Sets log level, format, handlers from config
    # BENEFIT: Professional logging infrastructure
    # TRADE-OFF: Additional overhead (minimal)
    
    log_level = getattr(logging, config['logging']['log_level'].upper())
    # WHAT: Converts string "info" to logging.INFO constant
    # WHY: Config uses strings, logging needs constants
    # HOW: getattr dynamically accesses logging.INFO/DEBUG/etc
    # BENEFIT: Flexible log level from config
    # TRADE-OFF: Will error if invalid level string
    
    logging.basicConfig(
        # WHAT: Global logging configuration
        # WHY: Applies to all loggers in application
        # BENEFIT: Consistent logging across modules
        # TRADE-OFF: Can't easily have different configs per module
        
        level=log_level,
        # WHAT: Minimum severity to log (INFO, DEBUG, etc)
        # WHY: Filter out less important messages
        # BENEFIT: Control verbosity
        # TRADE-OFF: May miss debug info at higher levels
        
        format='%(asctime)s - %(levelname)s - %(message)s',
        # WHAT: Log message format template
        # WHY: Consistent, readable log entries
        # FORMAT: "2025-12-10 14:30:15 - INFO - Model loaded"
        # BENEFIT: Timestamp, severity, message in every log
        # TRADE-OFF: Slightly longer log lines
        
        handlers=[
            # WHAT: List of output destinations for logs
            # WHY: Can log to multiple places simultaneously
            # BENEFIT: See logs in console + save to file
            
            logging.StreamHandler(),
            # WHAT: Outputs logs to console/stdout
            # WHY: Real-time monitoring during training
            # BENEFIT: Immediate feedback
            # TRADE-OFF: Can clutter terminal
            
            logging.FileHandler(config['logging']['log_file']) if config['logging']['log_to_file'] else logging.NullHandler()
            # WHAT: Conditional file handler
            # WHY: Save logs to file if enabled in config
            # HOW: FileHandler writes to training.log, NullHandler discards
            # BENEFIT: Persistent log history for analysis
            # TRADE-OFF: Uses disk space
        ]
    )
    
    return logging.getLogger(__name__)
    # WHAT: Creates logger for this module
    # WHY: Namespace logs by module name
    # BENEFIT: Can filter logs by module
    # TRADE-OFF: Must call in each module


# ============================================================================
# MODEL AND TOKENIZER LOADING FUNCTION
# ============================================================================

def load_model_and_tokenizer(config):
    """Load model and tokenizer based on config"""
    # WHAT: Loads Qwen2-1.5B model and its tokenizer
    # WHY: Need model for training, tokenizer for text→tokens
    # HOW: Downloads from Hugging Face Hub on first run
    # BENEFIT: Pre-trained model with 1.5B parameters
    # TRADE-OFF: ~3GB download + ~6GB RAM usage
    
    logger = logging.getLogger(__name__)
    # WHAT: Get logger instance for this module
    # WHY: Log model loading progress
    # BENEFIT: Track download/loading status
    
    model_config = config['model']
    # WHAT: Extract model section from config
    # WHY: Clean access to model settings
    # BENEFIT: Avoid config['model']['name'] repetition
    
    logger.info(f"Loading model: {model_config['name']}")
    # WHAT: Log which model is being loaded
    # WHY: User confirmation and debugging
    # BENEFIT: Know exactly what's happening
    
    tokenizer = AutoTokenizer.from_pretrained(
        # WHAT: Load tokenizer for text→token conversion
        # WHY: Model needs numeric tokens, not raw text
        # HOW: Auto-detects tokenizer type from model name
        # BENEFIT: Correct tokenizer automatically selected
        # TRADE-OFF: Must match model architecture
        
        model_config['name'],
        # WHAT: Model identifier (e.g., "Qwen/Qwen2-1.5B")
        # WHY: Specifies which tokenizer to download
        # BENEFIT: Centralized in config
        # TRADE-OFF: Must be valid HuggingFace model ID
        
        trust_remote_code=model_config['trust_remote_code']
        # WHAT: Allow executing custom code from model repo
        # WHY: Some models (like Qwen) need custom code
        # BENEFIT: Access to specialized model features
        # TRADE-OFF: Security risk - only use trusted models
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    # WHAT: Set padding token to end-of-sequence token
    # WHY: Some tokenizers don't define pad_token
    # HOW: Reuse EOS token for padding
    # BENEFIT: Prevents errors during batch processing
    # TRADE-OFF: Padding and EOS have same token ID
    
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    # WHAT: Maps string precision names to PyTorch types
    # WHY: Config uses strings, PyTorch needs dtype objects
    # BENEFIT: User-friendly config ("float32" vs torch.float32)
    # TRADE-OFF: Must keep mapping updated
    # PRECISION COMPARISON:
    #   - float32: Full precision, CPU compatible, most memory
    #   - float16: Half precision, GPU only, less memory, faster
    #   - bfloat16: Google's format, better range than fp16, GPU only
    
    model = AutoModelForCausalLM.from_pretrained(
        # WHAT: Load pre-trained Qwen2-1.5B model
        # WHY: Start with pre-trained weights (transfer learning)
        # HOW: Downloads from HuggingFace Hub, loads to memory
        # BENEFIT: Better starting point than random initialization
        # TRADE-OFF: Large download (~3GB), high memory (~6GB)
        
        model_config['name'],
        # WHAT: Model identifier
        # WHY: Specifies which model to download
        
        dtype=dtype_map[model_config['dtype']],
        # WHAT: Numerical precision for model weights
        # WHY: float32 required for CPU, float16 for GPU
        # BENEFIT: Control speed/memory trade-off
        # TRADE-OFF: float32 = slower but more accurate
        
        low_cpu_mem_usage=model_config['low_cpu_mem_usage'],
        # WHAT: Memory optimization flag
        # WHY: Reduces peak memory during loading
        # HOW: Loads weights incrementally vs all at once
        # BENEFIT: Can load larger models on limited RAM
        # TRADE-OFF: Slightly slower loading time
        
        trust_remote_code=model_config['trust_remote_code']
        # WHAT: Execute custom code from model repository
        # WHY: Qwen models require custom modeling code
        # BENEFIT: Access specialized model architectures
        # TRADE-OFF: Security risk with untrusted models
    )
    
    logger.info("Model loaded successfully")
    # WHAT: Confirm successful loading
    # WHY: User feedback that expensive operation completed
    
    return model, tokenizer
    # Returns: Tuple of (model, tokenizer) ready for training


# ============================================================================
# LORA SETUP FUNCTION
# ============================================================================

def setup_lora(model, config):
    """Setup LoRA configuration"""
    # WHAT: Applies LoRA (Low-Rank Adaptation) to model
    # WHY: Train large models efficiently (0.56% params vs 100%)
    # HOW: Injects small adapter matrices into attention layers
    # BENEFIT: Huge memory savings, faster training
    # TRADE-OFF: Slightly lower quality vs full fine-tuning
    
    logger = logging.getLogger(__name__)
    lora_config = config['lora']
    # WHAT: Extract LoRA settings from config
    
    logger.info(f"Setting up LoRA with r={lora_config['r']}, alpha={lora_config['lora_alpha']}")
    # WHAT: Log LoRA hyperparameters
    # WHY: These significantly impact training behavior
    
    lora = LoraConfig(
        # WHAT: Create LoRA configuration object
        # WHY: Specifies how LoRA adapters are created
        # BENEFIT: Fine control over efficiency/quality trade-off
        
        r=lora_config['r'],
        # WHAT: LoRA rank (adapter matrix dimension)
        # WHY: Controls adapter size and capacity
        # VALUE: 64 = medium capacity
        # BENEFIT: Higher r = more capacity = better quality
        # TRADE-OFF: Higher r = more parameters = more memory
        # TYPICAL VALUES: 8 (tiny), 64 (medium), 256 (large)
        
        lora_alpha=lora_config['lora_alpha'],
        # WHAT: Scaling factor for LoRA updates
        # WHY: Controls how much LoRA influences output
        # VALUE: 16 (with r=64 gives scaling of 16/64 = 0.25)
        # BENEFIT: Balance between base model and adaptation
        # TRADE-OFF: Too high = unstable, too low = slow learning
        
        lora_dropout=lora_config['lora_dropout'],
        # WHAT: Dropout probability in LoRA layers
        # WHY: Regularization to prevent overfitting
        # VALUE: 0.05 = 5% dropout rate
        # BENEFIT: Better generalization
        # TRADE-OFF: Slower convergence
        
        target_modules=lora_config['target_modules'],
        # WHAT: Which model layers to apply LoRA to
        # WHY: Only modify specific parts (attention)
        # VALUE: ["q_proj", "v_proj"] = query and value projections
        # BENEFIT: Focus on most important layers
        # TRADE-OFF: Missing k_proj, o_proj may limit adaptation
        # COMMON TARGETS:
        #   - q_proj, k_proj, v_proj: Attention queries/keys/values
        #   - o_proj: Attention output
        #   - gate_proj, up_proj, down_proj: FFN layers
        
        bias=lora_config.get('bias', 'none'),
        # WHAT: How to handle bias terms
        # WHY: Bias can be trained, frozen, or use LoRA
        # VALUE: "none" = don't train biases
        # BENEFIT: Fewer parameters to train
        # TRADE-OFF: Less flexibility
        # OPTIONS: "none", "all", "lora_only"
        
        task_type=lora_config.get('task_type', 'CAUSAL_LM')
        # WHAT: Type of task (text generation, classification, etc)
        # WHY: PEFT adjusts behavior based on task
        # VALUE: "CAUSAL_LM" = GPT-style text generation
        # BENEFIT: Optimized for autoregressive generation
        # TRADE-OFF: Not suitable for encoder-only tasks
    )
    
    model = get_peft_model(model, lora)
    # WHAT: Wraps base model with LoRA adapters
    # WHY: Injects trainable adapter layers
    # HOW: Freezes base model, adds small trainable matrices
    # BENEFIT: Now have ~8.7M trainable params instead of 1.5B
    # TRADE-OFF: Inference requires adapter loading
    
    model.print_trainable_parameters()
    # WHAT: Prints trainable vs total parameters
    # WHY: Confirm LoRA setup correctly
    # OUTPUT EXAMPLE: "trainable params: 8,847,360 || all params: 1,543,847,360 || trainable%: 0.5733"
    # BENEFIT: Verify parameter efficiency
    
    return model
    # Returns: Model with LoRA adapters attached


# ============================================================================
# DATASET PREPARATION FUNCTION
# ============================================================================

def prepare_dataset(config, tokenizer):
    """Prepare and tokenize dataset"""
    # WHAT: Loads training data and converts text to tokens
    # WHY: Models need numeric tokens, not raw text
    # HOW: Load JSONL, apply tokenization function, create labels
    # BENEFIT: Efficient batch processing
    # TRADE-OFF: All data loaded into memory
    
    logger = logging.getLogger(__name__)
    dataset_config = config['dataset']
    
    logger.info(f"Loading dataset from {dataset_config['train_file']}")
    
    dataset = load_dataset("json", data_files=dataset_config['train_file'])
    # WHAT: Load JSONL file using Hugging Face datasets
    # WHY: Optimized for ML workflows (batching, caching)
    # HOW: Reads train.jsonl, creates Dataset object
    # BENEFIT: Memory-mapped, fast iteration
    # TRADE-OFF: Different API than pandas/native Python
    # FORMAT: {"text": "User: ...\nAssistant: ..."}
    
    def tokenize_function(examples):
        # WHAT: Converts text to token IDs
        # WHY: Transformer models operate on tokens, not text
        # HOW: Batched processing for efficiency
        # BENEFIT: Vectorized operations on batches
        
        tokenized = tokenizer(
            # WHAT: Call tokenizer on batch of texts
            # WHY: Converts strings to input_ids
            # BENEFIT: Handles special tokens automatically
            
            examples[dataset_config['text_column']],
            # WHAT: Extract "text" column from examples
            # WHY: Dataset has {"text": "..."} structure
            # BENEFIT: Configurable column name
            
            truncation=dataset_config['truncation'],
            # WHAT: Cut sequences longer than max_length
            # WHY: Prevent memory overflow, ensure fixed size
            # VALUE: true = truncate to max_length
            # BENEFIT: Handle variable-length inputs
            # TRADE-OFF: Lose information beyond max_length
            
            padding=dataset_config['padding'],
            # WHAT: Add padding tokens to reach max_length
            # WHY: Batch processing needs same-length sequences
            # VALUE: "max_length" = pad all to 512 tokens
            # BENEFIT: Efficient batch training
            # TRADE-OFF: Wasted computation on padding
            # OPTIONS: "max_length", "longest", "do_not_pad"
            
            max_length=dataset_config['max_length'],
            # WHAT: Maximum sequence length
            # WHY: Memory constraint, model limit
            # VALUE: 512 tokens (~384 words)
            # BENEFIT: Fits in memory
            # TRADE-OFF: Longer sequences = more context but slower
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        # WHAT: Copy input_ids to create labels
        # WHY: Causal LM predicts next token, so labels = inputs shifted
        # HOW: Trainer handles shifting automatically
        # BENEFIT: Simple setup for language modeling
        # TRADE-OFF: None (standard practice)
        
        return tokenized
        # Returns: Dict with input_ids, attention_mask, labels
    
    logger.info("Tokenizing dataset...")
    
    dataset = dataset.map(
        # WHAT: Apply tokenize_function to all examples
        # WHY: Transform text dataset to tokenized dataset
        # HOW: Parallel processing with batching
        # BENEFIT: Fast preprocessing
        
        tokenize_function,
        # WHAT: Function to apply to each batch
        
        batched=True,
        # WHAT: Process multiple examples at once
        # WHY: Vectorized operations are faster
        # BENEFIT: 10-100x speedup vs example-by-example
        # TRADE-OFF: Higher memory usage during processing
        
        remove_columns=[dataset_config['text_column']]
        # WHAT: Drop original "text" column after tokenization
        # WHY: Don't need raw text during training (save memory)
        # BENEFIT: Smaller dataset in memory
        # TRADE-OFF: Can't inspect original text easily
    )
    
    logger.info(f"Dataset prepared: {len(dataset['train'])} training samples")
    # WHAT: Log dataset size
    # WHY: Confirm data loaded correctly
    
    return dataset
    # Returns: Dataset with input_ids, attention_mask, labels


# ============================================================================
# TRAINING ARGUMENTS CREATION FUNCTION
# ============================================================================

def create_training_arguments(config):
    """Create TrainingArguments from config"""
    # WHAT: Builds TrainingArguments object from config
    # WHY: Configures entire training process
    # HOW: Maps config dict to TrainingArguments parameters
    # BENEFIT: Centralized training configuration
    # TRADE-OFF: Many parameters to understand
    
    training_config = config['training']
    eval_config = config['evaluation']
    
    return TrainingArguments(
        # WHAT: Hugging Face training configuration object
        # WHY: Controls all aspects of training loop
        # BENEFIT: Battle-tested defaults, extensive options
        # TRADE-OFF: Complex with 50+ parameters
        
        output_dir=training_config['output_dir'],
        # WHAT: Directory to save model checkpoints
        # WHY: Preserve training progress
        # VALUE: "./qwen_finetune"
        # BENEFIT: Can resume training, use saved model
        # TRADE-OFF: Uses disk space (~3GB per checkpoint)
        
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        # WHAT: Number of examples per training step (per device)
        # WHY: Memory constraint
        # VALUE: 1 = process one example at a time
        # BENEFIT: Fits in limited CPU memory
        # TRADE-OFF: Slower training, noisier gradients
        # TYPICAL VALUES: 1-4 (CPU), 8-64 (GPU)
        
        per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 1),
        # WHAT: Batch size for evaluation
        # WHY: Can be larger than training (no gradients)
        # VALUE: 1 (same as training)
        # BENEFIT: Consistent memory usage
        
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        # WHAT: Accumulate gradients over N steps before updating
        # WHY: Simulate larger batch size
        # VALUE: 4 = effective batch size of 1×4=4
        # BENEFIT: Better gradient estimates without more memory
        # TRADE-OFF: 4x slower updates (but better quality)
        # EFFECTIVE BATCH = per_device_batch_size × gradient_accumulation_steps × num_devices
        
        learning_rate=training_config['learning_rate'],
        # WHAT: Step size for weight updates
        # WHY: Controls training speed and stability
        # VALUE: 2e-4 = 0.0002 (common for LoRA)
        # BENEFIT: Fast learning without instability
        # TRADE-OFF: Too high = divergence, too low = slow
        # TYPICAL VALUES: 1e-5 (full FT), 1e-4 to 5e-4 (LoRA)
        
        num_train_epochs=training_config['num_train_epochs'],
        # WHAT: Number of complete passes through dataset
        # WHY: More epochs = more learning (up to a point)
        # VALUE: 3 epochs
        # BENEFIT: Good balance for small datasets
        # TRADE-OFF: Too many = overfitting
        
        warmup_steps=training_config.get('warmup_steps', 0),
        # WHAT: Gradually increase LR for N steps
        # WHY: Prevent instability at start
        # VALUE: 100 steps
        # BENEFIT: More stable training
        # TRADE-OFF: Slower initial learning
        
        logging_steps=training_config['logging_steps'],
        # WHAT: Log metrics every N steps
        # WHY: Monitor training progress
        # VALUE: 10 = log every 10 steps
        # BENEFIT: See loss trends in real-time
        # TRADE-OFF: Too frequent = clutter
        
        save_steps=training_config['save_steps'],
        # WHAT: Save checkpoint every N steps
        # WHY: Preserve progress, enable resume
        # VALUE: 500 = save every 500 steps
        # BENEFIT: Can recover from crashes
        # TRADE-OFF: Disk I/O overhead
        
        save_total_limit=training_config.get('save_total_limit', 3),
        # WHAT: Maximum number of checkpoints to keep
        # WHY: Prevent unlimited disk usage
        # VALUE: 3 = keep last 3 checkpoints
        # BENEFIT: Bounded disk usage
        # TRADE-OFF: Old checkpoints deleted
        
        eval_strategy=training_config.get('evaluation_strategy', 'no'),
        # WHAT: When to run evaluation
        # WHY: Track validation metrics
        # VALUE: "no" = don't evaluate during training
        # BENEFIT: Faster training
        # TRADE-OFF: Can't track overfitting
        # OPTIONS: "no", "steps", "epoch"
        
        eval_steps=training_config.get('eval_steps'),
        # WHAT: Evaluate every N steps (if eval_strategy="steps")
        # VALUE: None (eval disabled)
        
        fp16=training_config['fp16'],
        # WHAT: Use 16-bit floating point
        # WHY: Faster on GPU, half memory
        # VALUE: false (CPU doesn't support fp16)
        # BENEFIT: 2x speedup on GPU
        # TRADE-OFF: Numerical instability risk
        
        use_cpu=training_config.get('use_cpu', True),
        # WHAT: Force CPU usage (ignore GPU)
        # WHY: Ensure CPU training
        # VALUE: true
        # BENEFIT: Works on any machine
        # TRADE-OFF: Much slower than GPU
        
        optim=training_config.get('optim', 'adamw_torch'),
        # WHAT: Optimizer algorithm
        # WHY: How to update weights
        # VALUE: "adamw_torch" = Adam with weight decay
        # BENEFIT: Good default, handles sparse gradients
        # TRADE-OFF: More memory than SGD
        # OPTIONS: "adamw_torch", "sgd", "adafactor"
        
        weight_decay=training_config.get('weight_decay', 0.0),
        # WHAT: L2 regularization strength
        # WHY: Prevent overfitting
        # VALUE: 0.01 = mild regularization
        # BENEFIT: Better generalization
        # TRADE-OFF: Slower convergence
        
        max_grad_norm=training_config.get('max_grad_norm', 1.0),
        # WHAT: Clip gradients to this maximum norm
        # WHY: Prevent exploding gradients
        # VALUE: 1.0
        # BENEFIT: Training stability
        # TRADE-OFF: May slow learning if clipping often
        
        lr_scheduler_type=training_config.get('lr_scheduler_type', 'linear'),
        # WHAT: Learning rate schedule
        # WHY: Adjust LR during training
        # VALUE: "cosine" = cosine decay
        # BENEFIT: Better convergence
        # TRADE-OFF: More complex than constant LR
        # OPTIONS: "linear", "cosine", "constant"
        
        load_best_model_at_end=eval_config.get('load_best_model_at_end', False),
        # WHAT: Load best checkpoint after training
        # WHY: Use best model, not last
        # VALUE: false (no eval = no "best")
        # BENEFIT: Better final model
        # TRADE-OFF: Requires evaluation enabled
        
        metric_for_best_model=eval_config.get('metric_for_best_model', 'loss'),
        # WHAT: Which metric to optimize
        # VALUE: "loss" (if eval enabled)
        
        greater_is_better=eval_config.get('greater_is_better', False),
        # WHAT: Is higher metric better?
        # VALUE: false (lower loss is better)
        
        report_to=config['logging'].get('report_to', []),
        # WHAT: Logging integrations (wandb, tensorboard)
        # VALUE: [] = no external logging
        # BENEFIT: Simpler setup
        # TRADE-OFF: No fancy dashboards
        
        seed=config.get('seed', 42),
        # WHAT: Random seed for reproducibility
        # WHY: Get same results each run
        # VALUE: 42
        # BENEFIT: Reproducible experiments
        # TRADE-OFF: None
    )


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train(config_path="config.yml"):
    """Main training function"""
    # WHAT: Orchestrates entire training pipeline
    # WHY: Single entry point for training
    # HOW: Calls all setup functions, runs Trainer
    # BENEFIT: Clean, linear workflow
    # TRADE-OFF: Long function (could be split)
    
    # Load configuration
    config = load_config(config_path)
    # WHAT: Parse YAML config file
    # WHY: Get all training parameters
    
    # Setup logging
    logger = setup_logging(config)
    # WHAT: Configure logging system
    # WHY: Track training progress
    
    logger.info("Starting training process...")
    # WHAT: Log start of training
    
    # Set random seed
    torch.manual_seed(config.get('seed', 42))
    # WHAT: Initialize random number generator
    # WHY: Reproducible results
    # BENEFIT: Same results across runs
    # TRADE-OFF: Not truly random
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    # WHAT: Load Qwen2-1.5B and tokenizer
    # WHY: Need model to train
    
    # Setup LoRA
    model = setup_lora(model, config)
    # WHAT: Apply LoRA adapters
    # WHY: Enable parameter-efficient training
    
    # Prepare dataset
    dataset = prepare_dataset(config, tokenizer)
    # WHAT: Load and tokenize training data
    # WHY: Models need tokenized inputs
    
    # Create training arguments
    training_args = create_training_arguments(config)
    # WHAT: Build training configuration
    # WHY: Control training behavior
    
    # Initialize trainer
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        # WHAT: Hugging Face training orchestrator
        # WHY: Handles training loop automatically
        # BENEFIT: Don't write training loop manually
        # TRADE-OFF: Less control over details
        
        model=model,
        # WHAT: Model to train (with LoRA)
        
        args=training_args,
        # WHAT: Training configuration
        
        train_dataset=dataset["train"],
        # WHAT: Tokenized training data
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    # WHAT: Execute training loop
    # WHY: Optimize model on training data
    # HOW: Iterates epochs, batches, forward/backward pass
    # BENEFIT: Automatic mixed precision, gradient accumulation, logging
    # TRADE-OFF: Takes hours/days on CPU
    
    # Save final model
    logger.info(f"Saving model to {config['training']['output_dir']}")
    trainer.save_model()
    # WHAT: Save model weights to disk
    # WHY: Preserve trained model
    # SAVES: adapter_model.bin, adapter_config.json
    
    tokenizer.save_pretrained(config['training']['output_dir'])
    # WHAT: Save tokenizer config
    # WHY: Need tokenizer for inference
    # SAVES: tokenizer.json, tokenizer_config.json
    
    logger.info("Training completed successfully!")
    
    return trainer
    # Returns: Trainer object (for inspection)


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # WHAT: Code runs only if script executed directly
    # WHY: Prevent execution when imported as module
    # BENEFIT: Module can be imported safely
    
    import sys
    # WHAT: System module for command-line args
    
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yml"
    # WHAT: Get config path from command line or use default
    # WHY: Allow custom config files
    # USAGE: python train_with_config.py custom.yml
    # BENEFIT: Flexibility for experiments
    
    if not os.path.exists(config_file):
        # WHAT: Check if config file exists
        # WHY: Fail fast with helpful error
        
        print(f"Error: Config file '{config_file}' not found!")
        print("Usage: python src/train_with_config.py [config.yml]")
        sys.exit(1)
        # WHAT: Exit with error code 1
        # WHY: Signal failure to shell
    
    train(config_file)
    # WHAT: Start training with specified config
```

---

## Usage Examples

### Basic Training
```bash
python src/train_with_config.py
```

### Custom Config
```bash
python src/train_with_config.py my_config.yml
```

### Import as Module
```python
from src.train_with_config import train
trainer = train("config.yml")
```

---

## Configuration Dependencies

**Required Config Sections:**
- `model`: name, dtype, trust_remote_code
- `lora`: r, lora_alpha, target_modules
- `dataset`: train_file, text_column, max_length
- `training`: output_dir, batch_size, learning_rate, epochs
- `logging`: log_level, log_file

---

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Training Time** | ~30 min/epoch | On CPU (3 samples) |
| **Memory Usage** | ~8GB RAM | Model + gradients |
| **Disk Usage** | ~3GB/checkpoint | LoRA adapters |
| **Trainable Params** | 8.7M (0.56%) | vs 1.5B total |

---

## Common Issues & Solutions

**Issue**: "CUDA out of memory"  
**Solution**: Already using CPU, check `use_cpu=true`

**Issue**: "Config file not found"  
**Solution**: Run from project root: `python src/train_with_config.py`

**Issue**: "Model download fails"  
**Solution**: Check internet, verify model name

**Issue**: "Training loss NaN"  
**Solution**: Lower learning_rate or increase warmup_steps

---

## Related Files
- `config.yml` - Configuration file
- `train.jsonl` - Training data
- `evaluation.py` - Evaluate trained model
- `chat.py` - Use trained model for inference
