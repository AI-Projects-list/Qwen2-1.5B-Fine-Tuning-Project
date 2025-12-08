import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_finetuned_model(model_path="./qwen_finetune"):
    """Load the fine-tuned model with LoRA adapters"""
    print(f"Loading model from {model_path}...")
    
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
    print("Model loaded successfully!\n")
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_length=200, temperature=0.7):
    """Generate response from the model"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def chat_loop(model, tokenizer):
    """Interactive chat loop"""
    print("="*60)
    print("Fine-tuned Model Chat Interface")
    print("="*60)
    print("Commands:")
    print("  - Type your message to get a response")
    print("  - Type 'quit' or 'exit' to end the chat")
    print("  - Type 'clear' to clear conversation history")
    print("="*60)
    print()
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if user_input.lower() == 'clear':
                conversation_history = []
                print("\nConversation cleared!\n")
                continue
            
            # Add user input to history
            conversation_history.append(f"User: {user_input}")
            
            # Create prompt from conversation history
            prompt = "\n".join(conversation_history[-5:]) + "\nAssistant:"
            
            # Generate response
            print("Assistant: ", end="", flush=True)
            response = generate_response(model, tokenizer, prompt)
            
            # Extract only the new response (remove prompt)
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            print(response)
            print()
            
            # Add assistant response to history
            conversation_history.append(f"Assistant: {response}")
            
        except KeyboardInterrupt:
            print("\n\nChat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.\n")


def single_prompt(model, tokenizer, prompt):
    """Single prompt mode - get one response and exit"""
    print(f"\nPrompt: {prompt}\n")
    response = generate_response(model, tokenizer, prompt)
    print(f"Response: {response}\n")
    return response


if __name__ == "__main__":
    import sys
    
    # Load the fine-tuned model
    model_path = "./qwen_finetune"
    
    try:
        model, tokenizer = load_finetuned_model(model_path)
        
        # Check if a prompt was provided as command line argument
        if len(sys.argv) > 1:
            # Single prompt mode
            prompt = " ".join(sys.argv[1:])
            single_prompt(model, tokenizer, prompt)
        else:
            # Interactive chat mode
            chat_loop(model, tokenizer)
            
    except FileNotFoundError:
        print(f"Error: Model not found at '{model_path}'")
        print("Please make sure you have trained the model first by running:")
        print("  python src/train_with_config.py")
    except Exception as e:
        print(f"Error loading model: {e}")
