import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np


def load_model_and_tokenizer(model_path):
    """Load fine-tuned model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Set pad_token if not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def predict_ok_or_not(model, tokenizer, text, max_length=512):
    """
    Generate response and evaluate if it's appropriate/good quality
    Returns: prediction ('ok' or 'not ok'), full response
    """
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the input from response to get only generated part
    generated_text = response[len(text):].strip()
    
    # Evaluate quality: check if response is meaningful and relevant
    # Good response indicators: reasonable length, not empty, contains relevant words
    response_lower = generated_text.lower()
    
    # Check for quality indicators
    is_empty = len(generated_text) < 5
    is_coherent = len(generated_text.split()) > 2  # At least 3 words
    has_content = any(char.isalnum() for char in generated_text)
    
    # Determine if output quality is "ok" or "not ok"
    if is_empty or not is_coherent or not has_content:
        prediction = "not ok"  # Poor quality output
    else:
        prediction = "ok"  # Good quality output
    
    return prediction, generated_text


def evaluate_dataset(model, tokenizer, dataset_path, text_column="input", label_column="label"):
    """
    Evaluate model on a test dataset
    Expected label format: 'ok' or 'not ok' (or 'nok')
    """
    dataset = load_dataset("json", data_files=dataset_path)["train"]
    
    predictions = []
    true_labels = []
    
    print(f"Evaluating {len(dataset)} samples...")
    
    for i, sample in enumerate(dataset):
        text = sample.get(text_column, "")
        true_label = sample.get(label_column, "").lower()
        
        # Normalize true label
        if "not ok" in true_label or "nok" in true_label:
            true_label = "not ok"
        elif "ok" in true_label:
            true_label = "ok"
        else:
            continue  # Skip unknown labels
        
        prediction, _ = predict_ok_or_not(model, tokenizer, text)
        
        predictions.append(prediction)
        true_labels.append(true_label)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(dataset)} samples")
    
    return predictions, true_labels


def calculate_metrics(predictions, true_labels):
    """Calculate classification metrics"""
    # Convert to binary (ok=1, not ok=0)
    pred_binary = [1 if p == "ok" else 0 for p in predictions]
    true_binary = [1 if t == "ok" else 0 for t in true_labels]
    
    # Calculate metrics
    accuracy = accuracy_score(true_binary, pred_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_binary, pred_binary, average='binary', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(true_binary, pred_binary)
    
    metrics = {
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


def print_metrics(metrics):
    """Print evaluation metrics in a readable format"""
    print("\n" + "="*50)
    print("EVALUATION METRICS - OK vs NOT OK Classification")
    print("="*50)
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"\nPredictions:")
    print(f"  OK: {metrics['ok_predictions']}")
    print(f"  NOT OK: {metrics['not_ok_predictions']}")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives (NOT OK): {metrics['confusion_matrix'][0][0]}")
    print(f"  False Positives (predicted OK): {metrics['confusion_matrix'][0][1]}")
    print(f"  False Negatives (predicted NOT OK): {metrics['confusion_matrix'][1][0]}")
    print(f"  True Positives (OK): {metrics['confusion_matrix'][1][1]}")
    print("="*50 + "\n")


def evaluate_single_text(model_path, text):
    """Evaluate a single text input"""
    model, tokenizer = load_model_and_tokenizer(model_path)
    prediction, response = predict_ok_or_not(model, tokenizer, text)
    
    print(f"\nInput: {text}")
    print(f"Full Response: {response}")
    print(f"Prediction: {prediction.upper()}")
    
    return prediction, response


if __name__ == "__main__":
    # Example usage
    model_path = "./qwen_finetune"  # Path to your fine-tuned model
    test_dataset = "test.jsonl"  # Your test dataset
    
    # Option 1: Evaluate on dataset
    try:
        model, tokenizer = load_model_and_tokenizer(model_path)
        predictions, true_labels = evaluate_dataset(model, tokenizer, test_dataset)
        metrics = calculate_metrics(predictions, true_labels)
        print_metrics(metrics)
    except FileNotFoundError:
        print(f"Model not found at {model_path} or test dataset not found")
    
    # Option 2: Evaluate single text
    # evaluate_single_text(model_path, "Your text here")
