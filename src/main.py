def run():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model
    from datasets import load_dataset
    import torch

    model_name = "Qwen/Qwen2-1.5B"  # or Qwen/Qwen2.5-3B

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    lora = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj","v_proj"]
    )

    model = get_peft_model(model, lora)

    dataset = load_dataset("json", data_files="train.jsonl")
    
    def tokenize_function(examples):
        # Tokenize the text and use it as both input and labels for causal LM
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        # For causal language modeling, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir="./qwen_finetune",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        fp16=False,
        save_steps=500,
        use_cpu=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
    )

    trainer.train()

if __name__ == "__main__":
    run()
