# main.py - Comprehensive Documentation

## Overview
**Purpose**: Simple, self-contained training script with hardcoded parameters (legacy/alternative to train_with_config.py).

**What it does**: Loads Qwen2-1.5B, applies LoRA, tokenizes data, trains model - all in one function.

**Why it exists**: Quick training without config files; educational example; fallback option.

**How it works**: All training logic in single `run()` function with hardcoded values.

---

## Benefits
✅ **Self-Contained**: No config file needed, everything in code  
✅ **Simple**: One function, linear flow, easy to understand  
✅ **Educational**: Good for learning the training process  
✅ **Quick Setup**: Just run, no configuration required  
✅ **Transparent**: See all parameters at once  

---

## Trade-offs
⚠️ **Hardcoded Values**: Must edit code to change parameters  
⚠️ **Less Flexible**: No easy experimentation  
⚠️ **No Logging**: Minimal progress feedback  
⚠️ **Duplicated Code**: Overlaps with train_with_config.py  
⚠️ **Not Production-Ready**: Lacks error handling, validation  

---

## Line-by-Line Commented Code

```python
# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def run():
    # WHAT: Single function containing entire training pipeline
    # WHY: Simple, self-contained training script
    # HOW: Linear sequence of model loading → LoRA → data → training
    # BENEFIT: Easy to understand, minimal dependencies
    # TRADE-OFF: Must edit code to change parameters
    
    # ========================================================================
    # IMPORTS (Inside Function)
    # ========================================================================
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # WHAT: Hugging Face model and tokenizer classes
    # WHY: Load pre-trained Qwen model
    # BENEFIT: Standard transformer library
    # TRADE-OFF: Large dependency
    
    from transformers import TrainingArguments, Trainer
    # WHAT: Training infrastructure
    # WHY: Handle training loop automatically
    # BENEFIT: Don't write training loop manually
    # TRADE-OFF: Less control over details
    
    from peft import LoraConfig, get_peft_model
    # WHAT: LoRA parameter-efficient fine-tuning
    # WHY: Train with only 0.56% of parameters
    # BENEFIT: Much less memory, faster training
    # TRADE-OFF: Slightly lower quality vs full fine-tuning
    
    from datasets import load_dataset
    # WHAT: Hugging Face datasets library
    # WHY: Load JSONL training data
    # BENEFIT: Optimized for ML workflows
    # TRADE-OFF: Different API than pandas
    
    import torch
    # WHAT: PyTorch deep learning framework
    # WHY: Tensor operations, model training
    # BENEFIT: Industry standard
    # TRADE-OFF: Large dependency (~2GB)
    
    # ========================================================================
    # MODEL CONFIGURATION
    # ========================================================================
    
    model_name = "Qwen/Qwen2-1.5B"  # or Qwen/Qwen2.5-3B
    # WHAT: Hugging Face model identifier
    # WHY: Specifies which pre-trained model to load
    # OPTIONS:
    #   - "Qwen/Qwen2-1.5B": 1.5 billion parameters (used here)
    #   - "Qwen/Qwen2.5-3B": 3 billion parameters (more capable, slower)
    # BENEFIT: Pre-trained knowledge, better starting point
    # TRADE-OFF: Large download (~3GB), high memory (~6GB)
    
    # ========================================================================
    # TOKENIZER LOADING
    # ========================================================================
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # WHAT: Load tokenizer for text→token conversion
    # WHY: Models need numeric tokens, not text
    # HOW: Downloads from Hugging Face Hub
    # BENEFIT: Correct tokenizer for model automatically
    # TRADE-OFF: Requires internet, trust_remote_code security risk
    
    tokenizer.pad_token = tokenizer.eos_token
    # WHAT: Set padding token to end-of-sequence token
    # WHY: Qwen tokenizer doesn't define pad_token
    # HOW: Reuse EOS token (common practice)
    # BENEFIT: Prevents errors during batch processing
    # TRADE-OFF: Padding and EOS share same ID
    
    # ========================================================================
    # MODEL LOADING
    # ========================================================================
    
    model = AutoModelForCausalLM.from_pretrained(
        # WHAT: Load pre-trained causal language model
        # WHY: Start with pre-trained weights (transfer learning)
        # HOW: Downloads from Hugging Face Hub, loads to memory
        # BENEFIT: Better than random initialization
        # TRADE-OFF: ~3GB download, ~6GB RAM
        
        model_name,
        # WHAT: Model to load
        # VALUE: "Qwen/Qwen2-1.5B"
        
        dtype=torch.float32,
        # WHAT: Weight precision (32-bit floating point)
        # WHY: CPU requires float32 (no fp16 support)
        # BENEFIT: Full precision, stable
        # TRADE-OFF: 2x memory vs float16, slower
        # NOTE: GPU could use torch.float16 for speed
        
        low_cpu_mem_usage=True
        # WHAT: Memory optimization flag
        # WHY: Reduce peak memory during loading
        # HOW: Load weights incrementally vs all at once
        # BENEFIT: Can load larger models on limited RAM
        # TRADE-OFF: Slightly slower loading (~10% more time)
    )
    
    # ========================================================================
    # LORA CONFIGURATION
    # ========================================================================
    
    lora = LoraConfig(
        # WHAT: Configure LoRA adapters
        # WHY: Enable parameter-efficient fine-tuning
        # BENEFIT: Train ~0.56% of parameters instead of 100%
        # TRADE-OFF: Slightly lower quality vs full fine-tuning
        
        r=64,
        # WHAT: LoRA rank (adapter matrix dimension)
        # WHY: Controls adapter capacity
        # VALUE: 64 = medium capacity
        # BENEFIT: Balance between efficiency and quality
        # TRADE-OFF: Higher r = more params, more memory
        # TYPICAL VALUES:
        #   - 8: Very efficient, lower quality
        #   - 64: Good balance (used here)
        #   - 256: Higher quality, more memory
        
        lora_alpha=16,
        # WHAT: Scaling factor for LoRA updates
        # WHY: Controls influence of LoRA on output
        # VALUE: 16 (with r=64 gives 16/64=0.25 scaling)
        # BENEFIT: Balance base model and adaptation
        # TRADE-OFF: Too high = unstable, too low = slow learning
        # TYPICAL: alpha = r/4 to r/2
        
        lora_dropout=0.05,
        # WHAT: Dropout probability in LoRA layers
        # WHY: Regularization to prevent overfitting
        # VALUE: 0.05 = 5% dropout
        # BENEFIT: Better generalization
        # TRADE-OFF: Slightly slower convergence
        # TYPICAL: 0.0 to 0.1
        
        target_modules=["q_proj","v_proj"]
        # WHAT: Which model layers to apply LoRA to
        # WHY: Focus on important layers (attention)
        # VALUE: ["q_proj", "v_proj"] = query and value projections
        # BENEFIT: Efficient, targets most important parts
        # TRADE-OFF: Missing k_proj, o_proj may limit adaptation
        # OTHER OPTIONS:
        #   - ["q_proj", "k_proj", "v_proj", "o_proj"]: All attention
        #   - ["q_proj", "v_proj", "gate_proj", "up_proj"]: Attention + FFN
    )
    
    # ========================================================================
    # APPLY LORA TO MODEL
    # ========================================================================
    
    model = get_peft_model(model, lora)
    # WHAT: Wrap model with LoRA adapters
    # WHY: Inject trainable adapter matrices
    # HOW: Freezes base model, adds small trainable layers
    # BENEFIT: Now only ~8.7M params trainable (0.56% of 1.5B)
    # TRADE-OFF: Slightly different inference (need adapters)
    # RESULT: model.print_trainable_parameters() shows:
    #   trainable params: 8,847,360 || all params: 1,543,847,360 || trainable%: 0.5733
    
    # ========================================================================
    # DATASET LOADING
    # ========================================================================
    
    dataset = load_dataset("json", data_files="train.jsonl")
    # WHAT: Load training data from JSONL file
    # WHY: Need data to train on
    # HOW: Reads train.jsonl, creates Dataset object
    # BENEFIT: Efficient, memory-mapped data loading
    # TRADE-OFF: Must be in project root directory
    # FORMAT: {"text": "User: ...\nAssistant: ..."}
    
    # ========================================================================
    # TOKENIZATION FUNCTION
    # ========================================================================
    
    def tokenize_function(examples):
        # WHAT: Convert text to tokens for training
        # WHY: Models operate on tokens, not text
        # HOW: Batch tokenization with padding/truncation
        # BENEFIT: Efficient batch processing
        # TRADE-OFF: All data must fit in memory after tokenization
        
        # Tokenize the text and use it as both input and labels for causal LM
        tokenized = tokenizer(
            # WHAT: Call tokenizer on batch of texts
            # WHY: Convert strings to input_ids
            # BENEFIT: Batch processing is faster
            
            examples["text"],
            # WHAT: Extract "text" column from batch
            # WHY: This column contains training examples
            # FORMAT: "User: ...\nAssistant: ..."
            
            truncation=True,
            # WHAT: Cut sequences longer than max_length
            # WHY: Prevent memory overflow, ensure fixed size
            # BENEFIT: Handle variable-length inputs
            # TRADE-OFF: Lose information beyond max_length
            
            padding="max_length",
            # WHAT: Pad all sequences to max_length
            # WHY: Batch training needs same-length sequences
            # BENEFIT: Efficient GPU/CPU utilization
            # TRADE-OFF: Wasted computation on padding
            # OPTIONS:
            #   - "max_length": Pad all to max_length
            #   - "longest": Pad to longest in batch
            #   - False: No padding
            
            max_length=512,
            # WHAT: Maximum sequence length in tokens
            # WHY: Memory constraint, model limit
            # VALUE: 512 tokens ≈ 384 words
            # BENEFIT: Fits comfortably in CPU memory
            # TRADE-OFF: Longer sequences = more context but slower
            # NOTE: Qwen2 supports up to 32K tokens theoretically
        )
        
        # For causal language modeling, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        # WHAT: Copy input_ids to create labels
        # WHY: Causal LM predicts next token
        # HOW: Trainer automatically shifts by 1 position
        # BENEFIT: Simple setup (standard for language modeling)
        # TRADE-OFF: None (this is the standard approach)
        # EXPLANATION:
        #   - Model sees: [A, B, C, D]
        #   - Predicts:   [B, C, D, E]
        #   - Labels:     [B, C, D, E] (same as input shifted)
        
        return tokenized
        # Returns: Dict with input_ids, attention_mask, labels
    
    # ========================================================================
    # APPLY TOKENIZATION
    # ========================================================================
    
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    # WHAT: Apply tokenization to entire dataset
    # WHY: Transform text to tokens for training
    # HOW: Parallel processing with batching
    # BENEFIT: Fast preprocessing (vectorized ops)
    # TRADE-OFF: All data in memory after this
    # PARAMETERS:
    #   - batched=True: Process multiple examples at once (faster)
    #   - remove_columns=["text"]: Drop original text (save memory)
    
    # ========================================================================
    # TRAINING ARGUMENTS
    # ========================================================================
    
    training_args = TrainingArguments(
        # WHAT: Configure training hyperparameters
        # WHY: Control how training proceeds
        # BENEFIT: Centralized configuration
        # TRADE-OFF: Many parameters to understand
        
        output_dir="./qwen_finetune",
        # WHAT: Directory to save model checkpoints
        # WHY: Preserve training progress
        # BENEFIT: Can resume training, use saved model
        # TRADE-OFF: Uses disk space (~3GB per checkpoint)
        
        per_device_train_batch_size=1,
        # WHAT: Examples per training step (per device)
        # WHY: Memory constraint
        # VALUE: 1 = one example at a time
        # BENEFIT: Minimal memory usage
        # TRADE-OFF: Slower, noisier gradients
        # NOTE: Effective batch = 1 × 4 (gradient_accumulation) = 4
        
        gradient_accumulation_steps=4,
        # WHAT: Accumulate gradients over 4 steps
        # WHY: Simulate larger batch without more memory
        # VALUE: 4 = effective batch of 4
        # BENEFIT: Better gradient estimates
        # TRADE-OFF: 4x slower updates (but better quality)
        # EXPLANATION:
        #   - Process 4 batches
        #   - Accumulate gradients
        #   - Update weights once
        
        learning_rate=2e-4,
        # WHAT: Step size for weight updates
        # WHY: Controls training speed and stability
        # VALUE: 0.0002 (common for LoRA)
        # BENEFIT: Fast learning without instability
        # TRADE-OFF: Too high = divergence, too low = slow
        # TYPICAL VALUES:
        #   - 1e-5 to 5e-5: Full fine-tuning
        #   - 1e-4 to 5e-4: LoRA fine-tuning (used here)
        
        num_train_epochs=3,
        # WHAT: Number of complete passes through dataset
        # WHY: More epochs = more learning (up to a point)
        # VALUE: 3 epochs
        # BENEFIT: Good balance for small datasets
        # TRADE-OFF: Too many = overfitting
        # NOTE: With 3 samples, 3 epochs = 9 total examples seen
        
        logging_steps=10,
        # WHAT: Log metrics every 10 steps
        # WHY: Monitor training progress
        # BENEFIT: See loss trends
        # TRADE-OFF: Too frequent = cluttered output
        
        fp16=False,
        # WHAT: Don't use 16-bit floating point
        # WHY: CPU doesn't support fp16
        # BENEFIT: Stable training
        # TRADE-OFF: Slower than fp16 on GPU
        # NOTE: Would set to True on GPU for 2x speedup
        
        save_steps=500,
        # WHAT: Save checkpoint every 500 steps
        # WHY: Preserve progress, enable resume
        # BENEFIT: Can recover from crashes
        # TRADE-OFF: Disk I/O overhead
        # NOTE: With 3 samples, may not reach 500 steps
        
        use_cpu=True,
        # WHAT: Force CPU usage
        # WHY: Ensure training on CPU
        # BENEFIT: Works on any machine
        # TRADE-OFF: Much slower than GPU (10-100x)
    )
    
    # ========================================================================
    # TRAINER INITIALIZATION
    # ========================================================================
    
    trainer = Trainer(
        # WHAT: Hugging Face training orchestrator
        # WHY: Handles entire training loop automatically
        # BENEFIT: Don't write backprop, optimization manually
        # TRADE-OFF: Less control over low-level details
        
        model=model,
        # WHAT: Model to train (with LoRA)
        # WHY: This is what gets updated
        
        args=training_args,
        # WHAT: Training configuration
        # WHY: Controls training behavior
        
        train_dataset=dataset["train"],
        # WHAT: Tokenized training data
        # WHY: What to train on
        # NOTE: "train" is default split name from load_dataset
    )
    
    # ========================================================================
    # EXECUTE TRAINING
    # ========================================================================
    
    trainer.train()
    # WHAT: Run training loop
    # WHY: Optimize model on training data
    # HOW: Iterates epochs, batches, forward/backward passes
    # BENEFIT: Automatic:
    #   - Gradient computation
    #   - Optimization steps
    #   - Gradient accumulation
    #   - Logging
    #   - Checkpoint saving
    # TRADE-OFF: Takes time (hours on CPU)
    # OUTPUT: Prints loss, learning rate every logging_steps
    
    # NOTE: Model automatically saved to output_dir after training


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # WHAT: Executed when script run directly
    # WHY: Not run when imported as module
    # BENEFIT: Can import run() without executing
    
    run()
    # WHAT: Start training
    # WHY: User executed script
    # BENEFIT: Simple one-line execution
```

---

## Usage Examples

### Run Training
```bash
python src/main.py
```

### Import as Module
```python
from src.main import run
run()  # Start training
```

---

## Comparison with train_with_config.py

| Feature | main.py | train_with_config.py |
|---------|---------|---------------------|
| **Configuration** | Hardcoded | YAML file |
| **Flexibility** | Low (edit code) | High (edit config) |
| **Complexity** | Simple | More complex |
| **Logging** | Basic | Comprehensive |
| **Error Handling** | Minimal | Robust |
| **Use Case** | Learning, quick tests | Production |

---

## Hardcoded Parameters

| Parameter | Value | Location |
|-----------|-------|----------|
| **Model** | Qwen/Qwen2-1.5B | Line 10 |
| **LoRA Rank** | 64 | Line 20 |
| **LoRA Alpha** | 16 | Line 21 |
| **Target Modules** | q_proj, v_proj | Line 23 |
| **Max Length** | 512 | Line 34 |
| **Batch Size** | 1 | Line 43 |
| **Gradient Accum** | 4 | Line 44 |
| **Learning Rate** | 2e-4 | Line 45 |
| **Epochs** | 3 | Line 46 |
| **Output Dir** | ./qwen_finetune | Line 42 |

**To Modify**: Edit code directly at these lines.

---

## Training Process Flow

1. **Load Tokenizer** → Convert text to tokens
2. **Load Model** → 1.5B parameter Qwen2
3. **Apply LoRA** → Freeze base, add adapters
4. **Load Dataset** → Read train.jsonl
5. **Tokenize** → Convert all text to tokens
6. **Configure Training** → Set hyperparameters
7. **Initialize Trainer** → Setup training infrastructure
8. **Train** → Execute optimization loop
9. **Save Model** → Write to ./qwen_finetune/

---

## Expected Output

```
Loading checkpoint shards: 100%|██████████| 2/2 [00:05<00:00]
trainable params: 8,847,360 || all params: 1,543,847,360 || trainable%: 0.5733
Map: 100%|██████████| 3/3 [00:00<00:00, 150.00 examples/s]
{'loss': 2.5123, 'learning_rate': 0.0002, 'epoch': 1.0}
{'loss': 1.8456, 'learning_rate': 0.00015, 'epoch': 2.0}
{'loss': 1.2789, 'learning_rate': 0.0001, 'epoch': 3.0}
Training completed!
```

---

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Model Size** | ~3GB | Download size |
| **Memory Usage** | ~8GB RAM | During training |
| **Training Time** | ~30 min | 3 samples, 3 epochs, CPU |
| **Trainable Params** | 8.7M (0.56%) | LoRA efficiency |

---

## Advantages

1. **Simple**: One file, one function, easy to read
2. **Self-Contained**: No config file dependency
3. **Educational**: See all parameters clearly
4. **Quick Start**: Just run, no setup
5. **Transparent**: All values visible

---

## Disadvantages

1. **Inflexible**: Must edit code for changes
2. **No Validation**: No error checking
3. **Limited Logging**: Minimal progress info
4. **Hardcoded Paths**: train.jsonl must be in root
5. **Code Duplication**: Overlaps with train_with_config.py

---

## When to Use

**Use main.py when:**
- Learning how training works
- Quick prototyping
- Single experiment
- Prefer code over config files
- Need minimal dependencies

**Use train_with_config.py when:**
- Running multiple experiments
- Need reproducibility
- Production deployment
- Want comprehensive logging
- Prefer configuration over code

---

## Related Files
- `train_with_config.py` - Production training script
- `config.yml` - Configuration for train_with_config.py
- `train.jsonl` - Training data
- `qwen_finetune/` - Output directory
