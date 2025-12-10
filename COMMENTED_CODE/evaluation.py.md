# evaluation.py - Comprehensive Documentation

## Overview
**Purpose**: Evaluate fine-tuned model's output quality on test dataset with classification metrics.

**What it does**: Loads trained model, generates responses, evaluates quality, calculates accuracy/precision/recall/F1.

**Why it exists**: Measure model performance objectively with quantitative metrics.

**How it works**: Generate responses for test inputs, check quality criteria, compare with labels, compute metrics.

---

## Benefits
✅ **Quantitative Evaluation**: Objective performance measurement  
✅ **Multiple Metrics**: Accuracy, precision, recall, F1, confusion matrix  
✅ **Quality Assessment**: Evaluates response coherence, not just keyword matching  
✅ **Flexible**: Can evaluate datasets or single inputs  
✅ **Sklearn Integration**: Industry-standard metrics library  

---

## Trade-offs
⚠️ **Subjective Quality**: Simple heuristics (length, coherence) may not capture true quality  
⚠️ **Slow Evaluation**: Generation takes time (100 tokens per sample)  
⚠️ **Binary Only**: Only handles "ok" vs "not ok" classification  
⚠️ **No Semantic Analysis**: Doesn't understand meaning, just surface features  

---

## Line-by-Line Commented Code

```python
# ============================================================================
# IMPORTS SECTION
# ============================================================================

import torch
# WHAT: PyTorch deep learning framework
# WHY: Load model, run inference without gradients
# BENEFIT: Mature ecosystem, efficient tensor operations
# TRADE-OFF: Large dependency

from transformers import AutoModelForCausalLM, AutoTokenizer
# WHAT: Hugging Face model loading classes
# WHY: Load fine-tuned Qwen2-1.5B model and tokenizer
# BENEFIT: Simple loading from directory
# TRADE-OFF: Must match training library version

from datasets import load_dataset
# WHAT: Hugging Face datasets library
# WHY: Load test.jsonl efficiently
# BENEFIT: Same API as training, handles JSONL naturally
# TRADE-OFF: Different from pandas/native Python

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
# WHAT: Scikit-learn classification metrics
# WHY: Industry-standard evaluation metrics
# BENEFIT: Well-tested, comprehensive metrics
# TRADE-OFF: Additional dependency
# METRICS:
#   - accuracy_score: Overall correctness (TP+TN)/(TP+TN+FP+FN)
#   - precision_recall_fscore_support: Precision, Recall, F1 in one call
#   - confusion_matrix: 2x2 matrix of predictions vs truth

import numpy as np
# WHAT: Numerical computing library
# WHY: Array operations (though barely used here)
# BENEFIT: Standard for numerical work
# TRADE-OFF: Large dependency for minimal use


# ============================================================================
# MODEL LOADING FUNCTION
# ============================================================================

def load_model_and_tokenizer(model_path):
    """Load fine-tuned model and tokenizer"""
    # WHAT: Loads saved model from training output
    # WHY: Need model for generating responses
    # HOW: Load from directory with AutoModel classes
    # BENEFIT: Simple one-function loading
    # TRADE-OFF: No validation of model compatibility
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # WHAT: Load tokenizer from saved directory
    # WHY: Convert test text to tokens
    # HOW: Reads tokenizer.json, tokenizer_config.json
    # BENEFIT: Guaranteed compatibility with saved model
    # TRADE-OFF: Requires trust_remote_code for Qwen
    
    # Set pad_token if not defined
    if tokenizer.pad_token is None:
        # WHAT: Check if padding token is defined
        # WHY: Some tokenizers don't set pad_token
        # BENEFIT: Prevents errors during batch generation
        
        tokenizer.pad_token = tokenizer.eos_token
        # WHAT: Reuse end-of-sequence token for padding
        # WHY: Common practice when pad_token missing
        # BENEFIT: Simple solution
        # TRADE-OFF: Padding and EOS have same ID (can confuse model)
    
    model = AutoModelForCausalLM.from_pretrained(
        # WHAT: Load fine-tuned model (base + LoRA)
        # WHY: Need model to generate responses
        # HOW: Loads adapter_model.bin merged with base weights
        # BENEFIT: Single call loads everything
        # TRADE-OFF: ~6GB memory usage
        
        model_path,
        # WHAT: Path to saved model directory
        # VALUE: "./qwen_finetune"
        # BENEFIT: Centralized model location
        
        dtype=torch.float32,
        # WHAT: Load weights in float32 precision
        # WHY: CPU requires float32 (no fp16 support)
        # BENEFIT: Full precision, stable
        # TRADE-OFF: 2x memory vs float16, slower
        
        low_cpu_mem_usage=True,
        # WHAT: Memory optimization during loading
        # WHY: Reduce peak memory usage
        # HOW: Load weights incrementally
        # BENEFIT: Can load larger models
        # TRADE-OFF: Slightly slower load time
        
        trust_remote_code=True
        # WHAT: Execute custom code from model repo
        # WHY: Qwen models need custom modeling code
        # BENEFIT: Access full model features
        # TRADE-OFF: Security risk (only trust verified models)
    )
    
    model.eval()
    # WHAT: Set model to evaluation mode
    # WHY: Disable dropout, use eval behavior
    # HOW: Sets model.training = False
    # BENEFIT: Deterministic predictions
    # TRADE-OFF: Must remember to call (easy to forget)
    # IMPORTANT: Affects dropout, batch norm, etc.
    
    return model, tokenizer


# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_ok_or_not(model, tokenizer, text, max_length=512):
    """
    Generate response and evaluate if it's appropriate/good quality
    Returns: prediction ('ok' or 'not ok'), full response
    """
    # WHAT: Generate response and judge quality
    # WHY: Need to evaluate model output quality
    # HOW: Generate text, apply heuristics, return binary label
    # BENEFIT: Automated quality assessment
    # TRADE-OFF: Simple heuristics may miss nuanced quality issues
    
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    # WHAT: Tokenize input text
    # WHY: Model needs token IDs, not strings
    # HOW: Convert to tokens, truncate if >512, return PyTorch tensors
    # BENEFIT: Handles variable-length inputs
    # TRADE-OFF: Long inputs get truncated
    # RETURNS: Dict with input_ids, attention_mask
    
    with torch.no_grad():
        # WHAT: Disable gradient computation
        # WHY: Inference doesn't need gradients (save memory)
        # HOW: Context manager disables autograd
        # BENEFIT: Faster, less memory
        # TRADE-OFF: Can't backpropagate (but we don't need to)
        
        outputs = model.generate(
            # WHAT: Generate text continuation
            # WHY: Get model's response to input
            # HOW: Autoregressive sampling (token by token)
            # BENEFIT: Flexible generation with many parameters
            # TRADE-OFF: Slow (sequential generation)
            
            **inputs,
            # WHAT: Unpack input_ids and attention_mask
            # WHY: generate() needs these as keyword args
            # BENEFIT: Clean syntax
            
            max_new_tokens=100,
            # WHAT: Generate up to 100 new tokens
            # WHY: Limit response length
            # BENEFIT: Faster generation, bounded computation
            # TRADE-OFF: May cut off longer responses
            # NOTE: 100 tokens ≈ 75 words
            
            temperature=0.7,
            # WHAT: Sampling temperature (randomness)
            # WHY: Control diversity of outputs
            # VALUE: 0.7 = moderate randomness
            # BENEFIT: More creative than greedy (temp=0)
            # TRADE-OFF: Less deterministic
            # SCALE: 0.0 = deterministic, 1.0 = neutral, >1.0 = chaotic
            
            do_sample=False,
            # WHAT: Use greedy decoding (pick max probability)
            # WHY: Deterministic, reproducible outputs
            # BENEFIT: Consistent results
            # TRADE-OFF: Ignores temperature (contradiction with temp=0.7!)
            # NOTE: This overrides temperature setting
            
            pad_token_id=tokenizer.eos_token_id
            # WHAT: Token ID for padding
            # WHY: Prevents warnings during generation
            # BENEFIT: Clean output
            # TRADE-OFF: None
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # WHAT: Convert token IDs back to text
    # WHY: Need human-readable response
    # HOW: outputs[0] is first (only) sequence in batch
    # BENEFIT: Clean text without <EOS>, <PAD> tokens
    # TRADE-OFF: Can't see special tokens (usually fine)
    
    # Remove the input from response to get only generated part
    generated_text = response[len(text):].strip()
    # WHAT: Extract only the newly generated text
    # WHY: Input was included in output (causal LM behavior)
    # HOW: Slice string from end of input to end
    # BENEFIT: Isolate model's actual response
    # TRADE-OFF: Simple string slicing (could fail if tokenization changes length)
    
    # Evaluate quality: check if response is meaningful and relevant
    # Good response indicators: reasonable length, not empty, contains relevant words
    response_lower = generated_text.lower()
    # WHAT: Lowercase for easier checking
    # WHY: Case-insensitive quality checks
    # BENEFIT: More robust
    
    # Check for quality indicators
    is_empty = len(generated_text) < 5
    # WHAT: Check if response is too short
    # WHY: Very short responses usually low quality
    # THRESHOLD: 5 characters
    # BENEFIT: Filter out minimal responses
    # TRADE-OFF: "OK." is 3 chars but valid
    
    is_coherent = len(generated_text.split()) > 2  # At least 3 words
    # WHAT: Check word count
    # WHY: Single-word responses usually low quality
    # THRESHOLD: 3+ words
    # BENEFIT: Ensures some substance
    # TRADE-OFF: "I don't know" is 3 words but not informative
    
    has_content = any(char.isalnum() for char in generated_text)
    # WHAT: Check if contains letters/numbers
    # WHY: Responses with only punctuation are invalid
    # BENEFIT: Catches degenerate outputs
    # TRADE-OFF: "..." passes with ellipsis only
    
    # Determine if output quality is "ok" or "not ok"
    if is_empty or not is_coherent or not has_content:
        # WHAT: Check if any quality criterion failed
        # WHY: Fail if too short, not coherent, or no content
        # LOGIC: OR condition - fails if any criterion unmet
        
        prediction = "not ok"  # Poor quality output
        # WHAT: Label as poor quality
        # WHY: Failed at least one quality check
        # BENEFIT: Clear binary classification
        
    else:
        prediction = "ok"  # Good quality output
        # WHAT: Label as good quality
        # WHY: Passed all quality checks
        # BENEFIT: Simple quality gate
        # TRADE-OFF: Doesn't assess correctness, just surface quality
    
    return prediction, generated_text
    # Returns: Tuple of (binary label, generated text)


# ============================================================================
# DATASET EVALUATION FUNCTION
# ============================================================================

def evaluate_dataset(model, tokenizer, dataset_path, text_column="input", label_column="label"):
    """
    Evaluate model on a test dataset
    Expected label format: 'ok' or 'not ok' (or 'nok')
    """
    # WHAT: Evaluate model on entire test dataset
    # WHY: Get overall performance metrics
    # HOW: Generate for each sample, collect predictions, compare with labels
    # BENEFIT: Batch evaluation with progress tracking
    # TRADE-OFF: Sequential processing (slow)
    
    dataset = load_dataset("json", data_files=dataset_path)["train"]
    # WHAT: Load test dataset from JSONL
    # WHY: Need test samples with labels
    # NOTE: ["train"] because load_dataset treats all files as "train" split
    # BENEFIT: Efficient JSONL loading
    # TRADE-OFF: Confusing naming (test data in "train" split)
    
    predictions = []
    true_labels = []
    # WHAT: Initialize lists for predictions and ground truth
    # WHY: Need to collect all before calculating metrics
    # BENEFIT: Simple accumulation
    # TRADE-OFF: All held in memory (fine for small datasets)
    
    print(f"Evaluating {len(dataset)} samples...")
    # WHAT: Log dataset size
    # WHY: User knows what to expect
    
    for i, sample in enumerate(dataset):
        # WHAT: Iterate through all test samples
        # WHY: Generate prediction for each
        # HOW: enumerate gives index and sample
        # BENEFIT: Progress tracking
        
        text = sample.get(text_column, "")
        # WHAT: Extract input text from sample
        # WHY: Need text to generate response
        # DEFAULT: "" if column missing
        # BENEFIT: Graceful handling of missing data
        # TRADE-OFF: Silent failure (should probably error)
        
        true_label = sample.get(label_column, "").lower()
        # WHAT: Extract ground truth label
        # WHY: Need for comparison with prediction
        # HOW: Lowercase for normalization
        # BENEFIT: Case-insensitive matching
        
        # Normalize true label
        if "not ok" in true_label or "nok" in true_label:
            # WHAT: Standardize negative labels
            # WHY: Handle variations ("not ok", "nok", "NOT OK")
            # BENEFIT: Flexible label format
            
            true_label = "not ok"
            # WHAT: Normalize to standard format
            
        elif "ok" in true_label:
            # WHAT: Standardize positive labels
            # WHY: Handle variations ("ok", "OK", "okay")
            
            true_label = "ok"
            
        else:
            continue  # Skip unknown labels
            # WHAT: Skip samples with invalid labels
            # WHY: Can't evaluate without valid ground truth
            # BENEFIT: Robust to data quality issues
            # TRADE-OFF: Silently skips (should log)
        
        prediction, _ = predict_ok_or_not(model, tokenizer, text)
        # WHAT: Generate prediction for this sample
        # WHY: Get model's quality assessment
        # NOTE: Discards generated_text (second return value)
        # BENEFIT: Only keep what's needed
        
        predictions.append(prediction)
        true_labels.append(true_label)
        # WHAT: Accumulate predictions and labels
        # WHY: Need aligned lists for metrics calculation
        
        if (i + 1) % 10 == 0:
            # WHAT: Every 10 samples, log progress
            # WHY: User feedback for long evaluations
            # BENEFIT: Know it's still running
            # TRADE-OFF: Output clutter for small datasets
            
            print(f"Processed {i + 1}/{len(dataset)} samples")
    
    return predictions, true_labels
    # Returns: Two aligned lists for metrics calculation


# ============================================================================
# METRICS CALCULATION FUNCTION
# ============================================================================

def calculate_metrics(predictions, true_labels):
    """Calculate classification metrics"""
    # WHAT: Compute standard classification metrics
    # WHY: Quantify model performance
    # HOW: Use sklearn metrics on binary labels
    # BENEFIT: Industry-standard metrics
    # TRADE-OFF: Requires understanding of metrics
    
    # Convert to binary (ok=1, not ok=0)
    pred_binary = [1 if p == "ok" else 0 for p in predictions]
    true_binary = [1 if t == "ok" else 0 for t in true_labels]
    # WHAT: Convert string labels to binary integers
    # WHY: Sklearn metrics need numeric labels
    # ENCODING: "ok"=1 (positive), "not ok"=0 (negative)
    # BENEFIT: Standard binary classification format
    # TRADE-OFF: Must remember encoding (1=positive)
    
    # Calculate metrics
    accuracy = accuracy_score(true_binary, pred_binary)
    # WHAT: Overall correctness percentage
    # WHY: Most intuitive metric
    # FORMULA: (TP + TN) / (TP + TN + FP + FN)
    # BENEFIT: Easy to understand
    # TRADE-OFF: Misleading on imbalanced datasets
    # EXAMPLE: 90% "not ok" → 90% accuracy by always predicting "not ok"
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        # WHAT: Calculate precision, recall, F1 in one call
        # WHY: More informative than accuracy alone
        # BENEFIT: Efficient computation
        # TRADE-OFF: Returns tuple (must unpack)
        
        true_binary, pred_binary, 
        # WHAT: Ground truth and predictions
        # ORDER: true first, pred second (important!)
        
        average='binary',
        # WHAT: Binary classification averaging
        # WHY: We have 2 classes (ok vs not ok)
        # BENEFIT: Treats "ok" as positive class
        # TRADE-OFF: Ignores "not ok" metrics (could use 'macro')
        
        zero_division=0
        # WHAT: Return 0 if denominator is 0
        # WHY: Prevent division by zero errors
        # CASE: No positive predictions → precision = 0
        # BENEFIT: Graceful handling
        # TRADE-OFF: May hide issues (should investigate why 0)
    )
    # PRECISION: TP / (TP + FP) - Of all "ok" predictions, how many correct?
    # RECALL: TP / (TP + FN) - Of all actual "ok", how many found?
    # F1: 2 * (P * R) / (P + R) - Harmonic mean of P and R
    
    # Confusion matrix
    cm = confusion_matrix(true_binary, pred_binary)
    # WHAT: 2x2 matrix of predictions vs truth
    # WHY: See breakdown of errors
    # STRUCTURE:
    #           Predicted
    #           0    1
    #   Actual 0  TN   FP
    #          1  FN   TP
    # BENEFIT: Visual error analysis
    # TRADE-OFF: Harder to interpret than simple metrics
    
    metrics = {
        # WHAT: Pack all metrics into dictionary
        # WHY: Clean return value
        # BENEFIT: Named access (metrics['accuracy'])
        
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "total_samples": len(predictions),
        "ok_predictions": sum(pred_binary),
        "not_ok_predictions": len(pred_binary) - sum(pred_binary)
    }
    
    return metrics


# ============================================================================
# METRICS PRINTING FUNCTION
# ============================================================================

def print_metrics(metrics):
    """Print evaluation metrics in a readable format"""
    # WHAT: Display metrics in human-readable format
    # WHY: User needs to interpret results
    # HOW: Formatted printing with separators
    # BENEFIT: Clear, professional output
    # TRADE-OFF: Console output only (could also save)
    
    print("\n" + "="*50)
    print("EVALUATION METRICS - OK vs NOT OK Classification")
    print("="*50)
    # WHAT: Header with visual separator
    # WHY: Clear section demarcation
    # BENEFIT: Professional appearance
    
    print(f"Total Samples: {metrics['total_samples']}")
    # WHAT: Show dataset size
    # WHY: Context for other metrics
    
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    # WHAT: Overall correctness
    # FORMAT: 4 decimal places (e.g., 0.7500 = 75%)
    # BENEFIT: Precision without clutter
    
    print(f"Precision: {metrics['precision']:.4f}")
    # WHAT: Correctness of positive predictions
    # WHY: How often "ok" prediction is actually ok?
    # BENEFIT: Understand false positive rate
    
    print(f"Recall: {metrics['recall']:.4f}")
    # WHAT: Coverage of positive class
    # WHY: Are we finding all the "ok" samples?
    # BENEFIT: Understand false negative rate
    
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    # WHAT: Balanced metric
    # WHY: Single number balancing precision and recall
    # BENEFIT: Good summary metric
    # TRADE-OFF: Less interpretable than P and R separately
    
    print(f"\nPredictions:")
    print(f"  OK: {metrics['ok_predictions']}")
    print(f"  NOT OK: {metrics['not_ok_predictions']}")
    # WHAT: Show prediction distribution
    # WHY: Detect bias (always predicting one class)
    # BENEFIT: Quick sanity check
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives (NOT OK): {metrics['confusion_matrix'][0][0]}")
    # WHAT: Correctly predicted "not ok"
    # POSITION: Top-left of matrix [0][0]
    
    print(f"  False Positives (predicted OK): {metrics['confusion_matrix'][0][1]}")
    # WHAT: Wrongly predicted "ok" (actually "not ok")
    # POSITION: Top-right of matrix [0][1]
    # PROBLEM: Model too optimistic
    
    print(f"  False Negatives (predicted NOT OK): {metrics['confusion_matrix'][1][0]}")
    # WHAT: Wrongly predicted "not ok" (actually "ok")
    # POSITION: Bottom-left of matrix [1][0]
    # PROBLEM: Model too pessimistic
    
    print(f"  True Positives (OK): {metrics['confusion_matrix'][1][1]}")
    # WHAT: Correctly predicted "ok"
    # POSITION: Bottom-right of matrix [1][1]
    
    print("="*50 + "\n")
    # WHAT: Footer separator


# ============================================================================
# SINGLE TEXT EVALUATION FUNCTION
# ============================================================================

def evaluate_single_text(model_path, text):
    """Evaluate a single text input"""
    # WHAT: Quick evaluation of one input
    # WHY: Test specific examples
    # HOW: Load model, predict, print
    # BENEFIT: Interactive testing
    # TRADE-OFF: Loads model each time (slow)
    
    model, tokenizer = load_model_and_tokenizer(model_path)
    # WHAT: Load model from disk
    # WHY: Need model for prediction
    # TRADE-OFF: Slow if called repeatedly
    
    prediction, response = predict_ok_or_not(model, tokenizer, text)
    # WHAT: Generate and evaluate response
    
    print(f"\nInput: {text}")
    print(f"Full Response: {response}")
    print(f"Prediction: {prediction.upper()}")
    # WHAT: Display results
    # WHY: User sees input, output, quality judgment
    # BENEFIT: Clear formatting
    
    return prediction, response


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # WHAT: Main execution block
    # WHY: Run evaluation when script executed
    
    # Example usage
    model_path = "./qwen_finetune"  # Path to your fine-tuned model
    test_dataset = "test.jsonl"  # Your test dataset
    # WHAT: Define paths to model and test data
    # WHY: Hardcoded for convenience
    # TRADE-OFF: Could accept command-line args
    
    # Option 1: Evaluate on dataset
    try:
        model, tokenizer = load_model_and_tokenizer(model_path)
        # WHAT: Load model once
        # WHY: Reuse for all samples
        # BENEFIT: Faster than loading per sample
        
        predictions, true_labels = evaluate_dataset(model, tokenizer, test_dataset)
        # WHAT: Run evaluation on all test samples
        # WHY: Get comprehensive metrics
        
        metrics = calculate_metrics(predictions, true_labels)
        # WHAT: Compute all metrics
        
        print_metrics(metrics)
        # WHAT: Display results
        
    except FileNotFoundError:
        # WHAT: Handle missing model or dataset
        # WHY: Graceful error for common issue
        
        print(f"Model not found at {model_path} or test dataset not found")
        # WHAT: Helpful error message
        # BENEFIT: User knows what's wrong
        # TRADE-OFF: Could be more specific (which file?)
    
    # Option 2: Evaluate single text
    # evaluate_single_text(model_path, "Your text here")
    # WHAT: Commented out alternative usage
    # WHY: Show how to use single-text mode
    # BENEFIT: Documentation by example
```

---

## Usage Examples

### Dataset Evaluation
```bash
python src/evaluation.py
```

### Single Text Evaluation
```python
from src.evaluation import evaluate_single_text
evaluate_single_text("./qwen_finetune", "The system is working fine")
```

### Programmatic Use
```python
from src.evaluation import load_model_and_tokenizer, evaluate_dataset, calculate_metrics

model, tokenizer = load_model_and_tokenizer("./qwen_finetune")
preds, labels = evaluate_dataset(model, tokenizer, "test.jsonl")
metrics = calculate_metrics(preds, labels)
print(f"Accuracy: {metrics['accuracy']:.2%}")
```

---

## Metrics Interpretation

### Accuracy
- **Definition**: (TP + TN) / Total
- **Good**: >0.8 (80%+)
- **Warning**: Can be misleading on imbalanced data

### Precision
- **Definition**: TP / (TP + FP)
- **Meaning**: Of "ok" predictions, how many are correct?
- **High**: Few false positives (rarely wrong when predicting "ok")
- **Low**: Many false positives (often wrong when predicting "ok")

### Recall
- **Definition**: TP / (TP + FN)
- **Meaning**: Of actual "ok" samples, how many found?
- **High**: Few false negatives (finds most "ok" samples)
- **Low**: Many false negatives (misses "ok" samples)

### F1 Score
- **Definition**: 2 × (Precision × Recall) / (Precision + Recall)
- **Meaning**: Balanced metric
- **Good**: >0.7
- **Use**: When precision and recall both matter

---

## Quality Criteria Explanation

**Current Heuristics:**
1. **Length Check**: `len(text) >= 5` characters
2. **Coherence Check**: `>= 3` words
3. **Content Check**: Contains alphanumeric characters

**Limitations:**
- No semantic understanding
- No factual correctness check
- No relevance to input
- Simple surface-level checks

**Improvements:**
- Use embedding similarity
- Check for hallucinations
- Verify information accuracy
- Add domain-specific checks

---

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Evaluation Speed** | ~10 sec/sample | CPU generation |
| **Memory Usage** | ~6GB RAM | Model loaded |
| **Batch Size** | 1 | Sequential processing |

---

## Related Files
- `train_with_config.py` - Trains model evaluated here
- `test.jsonl` - Test dataset
- `chat.py` - Interactive inference (similar generation)
