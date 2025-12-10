# config.yml - Comprehensive Documentation

## Overview
**Purpose**: Central configuration file for fine-tuning Qwen2-1.5B with LoRA on CPU.

**What it does**: Defines all training parameters in structured YAML format.

**Why it exists**: Separate configuration from code for flexibility and reproducibility.

**How it works**: Parsed by train_with_config.py to control all aspects of training.

---

## Benefits
✅ **Centralized Configuration**: Single source of truth for all parameters  
✅ **Easy Experimentation**: Change settings without touching code  
✅ **Reproducibility**: Save configs for each experiment  
✅ **Version Control**: Track parameter changes over time  
✅ **Human-Readable**: YAML is easier to read/edit than code  

---

## Trade-offs
⚠️ **Syntax Sensitivity**: YAML indentation must be perfect  
⚠️ **No Validation**: Invalid values only caught at runtime  
⚠️ **Learning Curve**: Must understand all parameters  
⚠️ **Decoupling**: Parameters separated from where they're used  

---

## Complete Configuration with Line-by-Line Explanation

```yaml
# ============================================================================
# HEADER COMMENT
# ============================================================================

# Fine-tuning Configuration for Qwen Model
# WHAT: Descriptive header
# WHY: Document purpose of file
# BENEFIT: Clear file identification


# ============================================================================
# MODEL CONFIGURATION SECTION
# ============================================================================

# Model Configuration
# WHAT: Section defining model loading parameters
# WHY: Control which model and how to load it

model:
  # WHAT: Model configuration block
  # WHY: Group related model settings
  # BENEFIT: Organized structure
  
  name: "Qwen/Qwen2-1.5B"
  # WHAT: Hugging Face model identifier
  # WHY: Specify which pre-trained model to use
  # VALUE: "Qwen/Qwen2-1.5B" = 1.5 billion parameter model
  # BENEFIT: Pre-trained knowledge, good starting point
  # TRADE-OFF: ~3GB download, ~6GB RAM usage
  # ALTERNATIVES:
  #   - "Qwen/Qwen2-0.5B": Smaller, faster, less capable
  #   - "Qwen/Qwen2.5-3B": Larger, slower, more capable
  #   - "Qwen/Qwen2-7B": Much larger, best quality, high memory
  
  dtype: "float32"
  # WHAT: Data type for model weights
  # WHY: Control precision and hardware compatibility
  # VALUE: "float32" = 32-bit floating point (full precision)
  # BENEFIT: CPU compatible, stable, accurate
  # TRADE-OFF: 2x memory vs float16, slower
  # ALTERNATIVES:
  #   - "float16": Half precision, GPU only, 2x faster
  #   - "bfloat16": Google's format, GPU only, better range
  # REQUIREMENT: CPU requires float32
  
  low_cpu_mem_usage: true
  # WHAT: Memory optimization flag
  # WHY: Reduce peak memory during model loading
  # VALUE: true = enable optimization
  # HOW: Loads weights incrementally vs all at once
  # BENEFIT: Can load larger models on limited RAM
  # TRADE-OFF: Slightly slower loading (~10% more time)
  # RECOMMENDATION: Always true for large models
  
  trust_remote_code: true
  # WHAT: Allow executing custom code from model repo
  # WHY: Qwen models require custom modeling code
  # VALUE: true = allow custom code execution
  # BENEFIT: Access full model features
  # TRADE-OFF: Security risk with untrusted models
  # WARNING: Only use with verified models
  # REQUIREMENT: Qwen models need this set to true


# ============================================================================
# LORA CONFIGURATION SECTION
# ============================================================================

# LoRA Configuration
# WHAT: Parameter-Efficient Fine-Tuning settings
# WHY: Configure LoRA adapters for efficient training
# BENEFIT: Train only ~0.56% of parameters

lora:
  # WHAT: LoRA configuration block
  # WHY: Group LoRA-specific parameters
  
  r: 64
  # WHAT: LoRA rank (adapter matrix dimension)
  # WHY: Controls adapter size and capacity
  # VALUE: 64 = medium capacity, good balance
  # BENEFIT: Efficient yet capable adaptation
  # TRADE-OFF: Higher r = more params, more memory, better quality
  # CALCULATION: Trainable params ≈ 2 × d × r × num_layers
  # TYPICAL VALUES:
  #   - 8: Very efficient (~1M params), lower quality
  #   - 16: Efficient (~2M params), good for simple tasks
  #   - 64: Balanced (~8M params), recommended (used here)
  #   - 128: High quality (~16M params), more memory
  #   - 256: Very high quality (~32M params), expensive
  
  lora_alpha: 16
  # WHAT: Scaling factor for LoRA updates
  # WHY: Control influence of LoRA on model output
  # VALUE: 16 (with r=64 gives scaling of alpha/r = 16/64 = 0.25)
  # BENEFIT: Balance between base model and adaptation
  # TRADE-OFF: Too high = unstable training, too low = slow learning
  # FORMULA: Effective scaling = lora_alpha / r
  # TYPICAL PATTERN: alpha = r/4 to r/2
  # EXPLANATION:
  #   - Higher alpha → stronger LoRA influence
  #   - Lower alpha → weaker LoRA influence
  
  lora_dropout: 0.05
  # WHAT: Dropout probability in LoRA layers
  # WHY: Regularization to prevent overfitting
  # VALUE: 0.05 = 5% dropout rate
  # BENEFIT: Better generalization to unseen data
  # TRADE-OFF: Slightly slower convergence
  # TYPICAL VALUES:
  #   - 0.0: No dropout, faster training, risk overfitting
  #   - 0.05: Light regularization (recommended)
  #   - 0.1: Medium regularization
  #   - 0.3: Heavy regularization (may underfit)
  
  target_modules:
    - "q_proj"
    - "v_proj"
  # WHAT: Which model layers to apply LoRA to
  # WHY: Focus on most important layers (attention)
  # VALUE: ["q_proj", "v_proj"] = query and value projections
  # BENEFIT: Efficient, targets critical components
  # TRADE-OFF: Missing other layers may limit adaptation
  # EXPLANATION:
  #   - q_proj: Query projection in attention
  #   - v_proj: Value projection in attention
  # ALTERNATIVES:
  #   - ["q_proj", "k_proj", "v_proj"]: All attention inputs
  #   - ["q_proj", "k_proj", "v_proj", "o_proj"]: Full attention
  #   - ["q_proj", "v_proj", "gate_proj", "up_proj"]: Attention + FFN
  # RECOMMENDATION: Start with q_proj + v_proj, expand if needed
  
  bias: "none"
  # WHAT: How to handle bias terms
  # WHY: Bias can be trained, frozen, or use LoRA
  # VALUE: "none" = don't train bias terms
  # BENEFIT: Fewer trainable parameters
  # TRADE-OFF: Less flexibility in adaptation
  # OPTIONS:
  #   - "none": Freeze all biases (most efficient)
  #   - "all": Train all biases (more flexible)
  #   - "lora_only": Apply LoRA to biases too
  # RECOMMENDATION: "none" is usually sufficient
  
  task_type: "CAUSAL_LM"
  # WHAT: Type of task for PEFT library
  # WHY: PEFT adjusts behavior based on task
  # VALUE: "CAUSAL_LM" = autoregressive text generation
  # BENEFIT: Optimized for GPT-style generation
  # TRADE-OFF: Not suitable for other task types
  # ALTERNATIVES:
  #   - "SEQ_2_SEQ_LM": Sequence-to-sequence (T5, BART)
  #   - "SEQ_CLS": Sequence classification
  #   - "TOKEN_CLS": Token classification (NER)
  # REQUIREMENT: Must match model architecture


# ============================================================================
# DATASET CONFIGURATION SECTION
# ============================================================================

# Dataset Configuration
# WHAT: Data loading and preprocessing settings
# WHY: Control how training data is prepared

dataset:
  # WHAT: Dataset configuration block
  # WHY: Group data-related settings
  
  train_file: "train.jsonl"
  # WHAT: Path to training data file
  # WHY: Specify where to load data from
  # VALUE: "train.jsonl" (relative to project root)
  # BENEFIT: Easy to change data source
  # TRADE-OFF: Must ensure file exists
  # FORMAT: JSONL (JSON Lines)
  # EXAMPLE: {"text": "User: Hello\nAssistant: Hi there!"}
  
  test_file: "test.jsonl"
  # WHAT: Path to test/evaluation data
  # WHY: Separate data for evaluation
  # VALUE: "test.jsonl"
  # BENEFIT: Validate model on unseen data
  # TRADE-OFF: Currently not used (eval disabled)
  
  text_column: "text"
  # WHAT: Name of column containing training text
  # WHY: JSONL can have multiple fields
  # VALUE: "text" (standard column name)
  # BENEFIT: Flexible data structure
  # TRADE-OFF: Must match actual JSONL structure
  # ALTERNATIVE: Could be "input", "prompt", etc.
  
  max_length: 512
  # WHAT: Maximum sequence length in tokens
  # WHY: Memory constraint, model limit
  # VALUE: 512 tokens (~384 words)
  # BENEFIT: Reasonable context, fits in memory
  # TRADE-OFF: Longer contexts truncated
  # CALCULATION:
  #   - Memory ≈ batch_size × max_length × hidden_size × 4 bytes
  #   - 512 tokens ≈ 384 words ≈ 2-3 paragraphs
  # ALTERNATIVES:
  #   - 256: Shorter, faster, less memory
  #   - 1024: Longer context, more memory
  #   - 2048: Very long, high memory
  # MODEL LIMIT: Qwen2 supports up to 32K, but memory intensive
  
  padding: "max_length"
  # WHAT: Padding strategy for sequences
  # WHY: Batch processing needs uniform lengths
  # VALUE: "max_length" = pad all to max_length
  # BENEFIT: Consistent tensor shapes
  # TRADE-OFF: Wasted computation on padding
  # ALTERNATIVES:
  #   - "longest": Pad to longest in batch (more efficient)
  #   - "do_not_pad": No padding (requires custom batching)
  # RECOMMENDATION: "max_length" for simplicity
  
  truncation: true
  # WHAT: Cut sequences longer than max_length
  # WHY: Prevent errors from oversized inputs
  # VALUE: true = enable truncation
  # BENEFIT: Handle variable-length inputs gracefully
  # TRADE-OFF: Lose information beyond max_length
  # RECOMMENDATION: Always true


# ============================================================================
# TRAINING CONFIGURATION SECTION
# ============================================================================

# Training Arguments
# WHAT: Core training hyperparameters
# WHY: Control optimization and training loop

training:
  # WHAT: Training configuration block
  # WHY: Group training-related settings
  
  output_dir: "./qwen_finetune"
  # WHAT: Directory to save model and checkpoints
  # WHY: Preserve training progress
  # VALUE: "./qwen_finetune" (relative to project root)
  # BENEFIT: Organized output location
  # TRADE-OFF: ~3GB disk space per checkpoint
  # CONTENTS: adapter_model.bin, adapter_config.json, tokenizer files
  
  per_device_train_batch_size: 1
  # WHAT: Number of examples per training step (per device)
  # WHY: Control memory usage
  # VALUE: 1 = one example at a time
  # BENEFIT: Minimal memory usage (~8GB total)
  # TRADE-OFF: Slower training, noisier gradients
  # CALCULATION: Effective batch = 1 × 4 (grad_accum) = 4
  # TYPICAL VALUES:
  #   - 1: CPU, limited memory (used here)
  #   - 4-8: Medium GPU (16GB)
  #   - 16-32: Large GPU (40GB+)
  
  per_device_eval_batch_size: 1
  # WHAT: Batch size for evaluation
  # WHY: Can be larger (no gradients needed)
  # VALUE: 1 (same as training for simplicity)
  # BENEFIT: Consistent memory usage
  # TRADE-OFF: Could be higher (2-4) for faster eval
  
  gradient_accumulation_steps: 4
  # WHAT: Accumulate gradients over N steps before updating
  # WHY: Simulate larger batch size without more memory
  # VALUE: 4 = effective batch of 1×4=4
  # BENEFIT: Better gradient estimates, more stable
  # TRADE-OFF: 4x slower updates (but better quality)
  # EXPLANATION:
  #   - Forward pass on 4 batches
  #   - Accumulate gradients
  #   - Single optimizer step
  # TYPICAL VALUES:
  #   - 1: No accumulation (batch_size is real batch)
  #   - 4: Good balance (used here)
  #   - 8-16: Very large effective batch
  
  learning_rate: 0.0002
  # WHAT: Step size for weight updates (2e-4)
  # WHY: Control training speed and stability
  # VALUE: 0.0002 = 2×10⁻⁴ (good for LoRA)
  # BENEFIT: Fast learning without instability
  # TRADE-OFF: Too high = divergence, too low = slow
  # TYPICAL VALUES:
  #   - 1e-5 to 5e-5: Full fine-tuning
  #   - 1e-4 to 5e-4: LoRA fine-tuning (used here)
  #   - 1e-3+: Usually too high, unstable
  # RECOMMENDATION: Start at 2e-4, adjust if needed
  
  num_train_epochs: 3
  # WHAT: Number of complete passes through dataset
  # WHY: Control total training time
  # VALUE: 3 epochs
  # BENEFIT: Good balance for small datasets
  # TRADE-OFF: Too many = overfitting, too few = underfitting
  # CALCULATION: With 3 samples, 3 epochs = 9 total examples
  # TYPICAL VALUES:
  #   - 1-3: Small datasets, quick experiments
  #   - 5-10: Medium datasets
  #   - 1-2: Large datasets (overfitting risk)
  
  warmup_steps: 100
  # WHAT: Gradually increase LR for first N steps
  # WHY: Prevent instability at start of training
  # VALUE: 100 steps
  # BENEFIT: More stable training start
  # TRADE-OFF: Slower initial learning
  # EXPLANATION:
  #   - LR increases linearly from 0 to learning_rate
  #   - Over first 100 steps
  #   - Then follows main schedule
  # TYPICAL VALUES:
  #   - 0: No warmup (may be unstable)
  #   - 100-500: Standard warmup
  #   - 1000+: Very long warmup
  
  logging_steps: 10
  # WHAT: Log metrics every N steps
  # WHY: Monitor training progress
  # VALUE: 10 = log every 10 steps
  # BENEFIT: Frequent feedback on training
  # TRADE-OFF: Too frequent = cluttered output
  # OUTPUT: Loss, learning rate, epoch
  # TYPICAL VALUES:
  #   - 1: Every step (verbose)
  #   - 10: Frequent updates (used here)
  #   - 100: Less frequent
  
  save_steps: 500
  # WHAT: Save checkpoint every N steps
  # WHY: Preserve progress, enable resume
  # VALUE: 500 steps
  # BENEFIT: Can recover from crashes
  # TRADE-OFF: Disk I/O overhead, disk space
  # NOTE: With 3 samples, may never reach 500 steps
  # TYPICAL VALUES:
  #   - 100: Frequent saves (safe but slow)
  #   - 500: Standard (used here)
  #   - 1000+: Less frequent
  
  save_total_limit: 3
  # WHAT: Maximum number of checkpoints to keep
  # WHY: Prevent unlimited disk usage
  # VALUE: 3 = keep last 3 checkpoints only
  # BENEFIT: Bounded disk usage (~9GB max)
  # TRADE-OFF: Old checkpoints deleted automatically
  # BEHAVIOR: Oldest deleted when limit exceeded
  # TYPICAL VALUES:
  #   - 1: Only latest (risky)
  #   - 3: Good balance (used here)
  #   - 5+: More history
  
  eval_steps: 250
  # WHAT: Evaluate every N steps (if eval enabled)
  # WHY: Track validation metrics
  # VALUE: 250 (but eval disabled)
  # BENEFIT: Monitor overfitting
  # TRADE-OFF: Slows training
  # NOTE: Currently unused (evaluation_strategy: "no")
  
  evaluation_strategy: "no"
  # WHAT: When to run evaluation
  # WHY: Validation metrics during training
  # VALUE: "no" = don't evaluate during training
  # BENEFIT: Faster training
  # TRADE-OFF: Can't track overfitting
  # ALTERNATIVES:
  #   - "no": No evaluation (faster)
  #   - "steps": Evaluate every eval_steps
  #   - "epoch": Evaluate every epoch
  # REQUIREMENT: Needs eval_dataset if not "no"
  
  fp16: false
  # WHAT: Use 16-bit floating point
  # WHY: Speed up training on GPU
  # VALUE: false = use float32
  # BENEFIT: CPU compatible, stable
  # TRADE-OFF: 2x slower than fp16 on GPU
  # REQUIREMENT: CPU doesn't support fp16
  # NOTE: Would be true on GPU for 2x speedup
  
  use_cpu: true
  # WHAT: Force CPU usage (ignore GPU)
  # WHY: Ensure training on CPU
  # VALUE: true = use CPU
  # BENEFIT: Works on any machine
  # TRADE-OFF: 10-100x slower than GPU
  # NOTE: Set to false to use GPU if available
  
  optim: "adamw_torch"
  # WHAT: Optimizer algorithm
  # WHY: How to update weights from gradients
  # VALUE: "adamw_torch" = Adam with weight decay
  # BENEFIT: Good default, handles sparse gradients
  # TRADE-OFF: More memory than SGD
  # ALTERNATIVES:
  #   - "adamw_torch": Adam with decoupled weight decay (recommended)
  #   - "sgd": Simple gradient descent (less memory)
  #   - "adafactor": Memory-efficient (for large models)
  
  weight_decay: 0.01
  # WHAT: L2 regularization strength
  # WHY: Prevent overfitting
  # VALUE: 0.01 = mild regularization
  # BENEFIT: Better generalization
  # TRADE-OFF: May slow convergence
  # EXPLANATION: Shrinks weights toward zero
  # TYPICAL VALUES:
  #   - 0.0: No regularization
  #   - 0.01: Light (recommended)
  #   - 0.1: Heavy
  
  max_grad_norm: 1.0
  # WHAT: Clip gradients to this maximum norm
  # WHY: Prevent exploding gradients
  # VALUE: 1.0
  # BENEFIT: Training stability
  # TRADE-OFF: May slow learning if clipping often
  # EXPLANATION: If ||gradient|| > 1.0, scale down
  # TYPICAL VALUES:
  #   - 0.5: Aggressive clipping
  #   - 1.0: Standard (used here)
  #   - 5.0: Lenient clipping
  
  lr_scheduler_type: "cosine"
  # WHAT: Learning rate schedule
  # WHY: Adjust LR during training
  # VALUE: "cosine" = cosine annealing
  # BENEFIT: Better convergence than constant LR
  # TRADE-OFF: More complex
  # BEHAVIOR: LR decreases smoothly to 0
  # ALTERNATIVES:
  #   - "linear": Linear decay
  #   - "cosine": Smooth decay (used here)
  #   - "constant": No change
  #   - "cosine_with_restarts": Periodic restarts


# ============================================================================
# EVALUATION CONFIGURATION SECTION
# ============================================================================
  
# Evaluation Configuration
# WHAT: Evaluation settings (currently disabled)
# WHY: Would control validation during training

evaluation:
  # WHAT: Evaluation configuration block
  
  metric: "accuracy"
  # WHAT: Which metric to track
  # VALUE: "accuracy" (if eval enabled)
  # NOTE: Currently unused (eval disabled)
  
  eval_on_start: false
  # WHAT: Evaluate before training starts
  # VALUE: false = skip initial eval
  # BENEFIT: Faster start
  
  load_best_model_at_end: false
  # WHAT: Load best checkpoint after training
  # WHY: Use best model, not last
  # VALUE: false (no eval = no "best")
  # REQUIREMENT: Needs evaluation_strategy != "no"
  
  metric_for_best_model: "eval_loss"
  # WHAT: Which metric defines "best"
  # VALUE: "eval_loss" (lower is better)
  # NOTE: Currently unused
  
  greater_is_better: false
  # WHAT: Is higher metric better?
  # VALUE: false (for loss, lower is better)
  # NOTE: Would be true for accuracy


# ============================================================================
# GENERATION CONFIGURATION SECTION
# ============================================================================

# Generation Configuration
# WHAT: Text generation settings (for inference)
# WHY: Control model output during chat/eval

generation:
  # WHAT: Generation configuration block
  # NOTE: Used by chat.py and evaluation.py
  
  max_new_tokens: 100
  # WHAT: Maximum tokens to generate
  # VALUE: 100 tokens (~75 words)
  # BENEFIT: Reasonable response length
  # TRADE-OFF: Longer = slower generation
  
  temperature: 0.7
  # WHAT: Sampling randomness
  # VALUE: 0.7 = moderate creativity
  # BENEFIT: More interesting than greedy
  # RANGE: 0.0 (deterministic) to 2.0 (very random)
  
  top_p: 0.9
  # WHAT: Nucleus sampling threshold
  # VALUE: 0.9 = top 90% probability mass
  # BENEFIT: Filter unlikely tokens
  
  top_k: 50
  # WHAT: Top-K sampling
  # VALUE: 50 = consider top 50 tokens
  # BENEFIT: Speed and quality balance
  
  do_sample: true
  # WHAT: Enable probabilistic sampling
  # VALUE: true = use temperature/top_p
  # BENEFIT: Diverse outputs
  
  num_beams: 1
  # WHAT: Beam search width
  # VALUE: 1 = no beam search (faster)
  # BENEFIT: Faster generation


# ============================================================================
# HARDWARE CONFIGURATION SECTION
# ============================================================================
  
# Hardware Configuration
# WHAT: Hardware-specific settings
# WHY: Control resource usage

hardware:
  device: "cpu"
  # WHAT: Which device to use
  # VALUE: "cpu" (not "cuda")
  # BENEFIT: Universal compatibility
  
  num_workers: 4
  # WHAT: Data loading threads
  # VALUE: 4 parallel workers
  # BENEFIT: Faster data loading
  
  pin_memory: false
  # WHAT: Pin memory for GPU transfer
  # VALUE: false (CPU doesn't need this)
  # NOTE: Would be true for GPU


# ============================================================================
# LOGGING CONFIGURATION SECTION
# ============================================================================
  
# Logging Configuration
# WHAT: Logging settings
# WHY: Control progress tracking

logging:
  log_level: "info"
  # WHAT: Minimum log severity
  # VALUE: "info" = informational messages
  # OPTIONS: "debug", "info", "warning", "error"
  
  log_to_file: true
  # WHAT: Save logs to file
  # VALUE: true = save to training.log
  # BENEFIT: Persistent log history
  
  log_file: "training.log"
  # WHAT: Log file path
  # VALUE: "training.log"
  # BENEFIT: Review training later
  
  report_to: []
  # WHAT: External logging integrations
  # VALUE: [] = no external logging
  # OPTIONS: ["wandb"], ["tensorboard"]


# ============================================================================
# REPRODUCIBILITY SECTION
# ============================================================================

# Reproducibility
# WHAT: Settings for reproducible results
# WHY: Get same results across runs

seed: 42
# WHAT: Random seed for all RNGs
# WHY: Reproducible experiments
# VALUE: 42 (arbitrary, traditional)
# BENEFIT: Same results each run
# TRADE-OFF: Not truly random

deterministic: true
# WHAT: Use deterministic algorithms
# WHY: Exact reproducibility
# VALUE: true = deterministic
# BENEFIT: Bit-exact reproduction
# TRADE-OFF: May be slower
```

---

## Usage Examples

### Use Configuration
```bash
python src/train_with_config.py config.yml
```

### Create Custom Config
```bash
cp config.yml my_experiment.yml
# Edit my_experiment.yml
python src/train_with_config.py my_experiment.yml
```

### Modify for GPU
```yaml
model:
  dtype: "float16"  # Changed from float32

training:
  fp16: true  # Changed from false
  use_cpu: false  # Changed from true
  per_device_train_batch_size: 8  # Increased from 1
```

---

## Configuration Groups

### Essential Parameters (Must Configure)
- `model.name`: Which model to train
- `training.output_dir`: Where to save
- `dataset.train_file`: Training data path

### Performance Parameters (Affect Speed/Memory)
- `model.dtype`: float32 (CPU) vs float16 (GPU)
- `training.per_device_train_batch_size`: Memory usage
- `training.gradient_accumulation_steps`: Effective batch size
- `dataset.max_length`: Context window size

### Quality Parameters (Affect Results)
- `lora.r`: Adapter capacity
- `training.learning_rate`: Learning speed
- `training.num_train_epochs`: Training duration
- `lora.target_modules`: Which layers to adapt

### Stability Parameters (Prevent Issues)
- `training.max_grad_norm`: Gradient clipping
- `training.weight_decay`: Regularization
- `lora.lora_dropout`: LoRA regularization
- `training.warmup_steps`: Stable start

---

## Common Modifications

### Increase Model Capacity
```yaml
lora:
  r: 128  # From 64
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"  # Added
```

### Longer Context
```yaml
dataset:
  max_length: 1024  # From 512
```

### Faster Training
```yaml
training:
  per_device_train_batch_size: 4  # From 1
  gradient_accumulation_steps: 1  # From 4
```

### More Regularization
```yaml
lora:
  lora_dropout: 0.1  # From 0.05

training:
  weight_decay: 0.05  # From 0.01
```

---

## Validation Checklist

Before training, verify:
- ✅ `dataset.train_file` exists
- ✅ `model.dtype` matches hardware (float32 for CPU)
- ✅ `training.use_cpu` matches intent
- ✅ `training.output_dir` has write permission
- ✅ YAML syntax is valid (no tabs, proper indentation)
- ✅ `lora.target_modules` are valid for model
- ✅ `training.learning_rate` is reasonable (1e-5 to 5e-4)

---

## Related Files
- `train_with_config.py` - Uses this configuration
- `train.jsonl` - Training data file specified here
- `qwen_finetune/` - Output directory specified here
