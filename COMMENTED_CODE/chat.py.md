# chat.py - Comprehensive Documentation

## Overview
**Purpose**: Interactive chat interface for using fine-tuned Qwen2-1.5B model.

**What it does**: Loads trained model and provides two modes: interactive chat loop or single-prompt response.

**Why it exists**: User-friendly interface to interact with fine-tuned model without writing code.

**How it works**: Loads model, maintains conversation history, generates responses using model.generate().

---

## Benefits
✅ **Two Modes**: Interactive chat + single prompt  
✅ **Conversation History**: Maintains context across turns  
✅ **User-Friendly**: Simple commands (quit, clear)  
✅ **Flexible Generation**: Configurable temperature, sampling  
✅ **Error Handling**: Graceful interrupt and error recovery  

---

## Trade-offs
⚠️ **Memory Growth**: History grows unbounded (limited to last 5 turns)  
⚠️ **Slow Generation**: CPU inference is slow (~30 sec/response)  
⚠️ **No Persistence**: History lost on exit  
⚠️ **Simple Parsing**: Basic response extraction may fail  

---

## Line-by-Line Commented Code

```python
# ============================================================================
# IMPORTS SECTION
# ============================================================================

import torch
# WHAT: PyTorch deep learning framework
# WHY: Required for model inference
# BENEFIT: Standard deep learning library
# TRADE-OFF: Large dependency

from transformers import AutoModelForCausalLM, AutoTokenizer
# WHAT: Hugging Face model and tokenizer classes
# WHY: Load fine-tuned Qwen2-1.5B model
# BENEFIT: Simple loading interface
# TRADE-OFF: Must match training library version

from peft import PeftModel
# WHAT: Parameter-Efficient Fine-Tuning model wrapper
# WHY: Handle LoRA adapter loading
# NOTE: Actually not used in this version (model pre-merged)
# BENEFIT: Could load base + adapters separately
# TRADE-OFF: Unused import (should remove)


# ============================================================================
# MODEL LOADING FUNCTION
# ============================================================================

def load_finetuned_model(model_path="./qwen_finetune"):
    """Load the fine-tuned model with LoRA adapters"""
    # WHAT: Load saved fine-tuned model from disk
    # WHY: Need model for generating responses
    # HOW: Load from directory saved by training
    # BENEFIT: Single function for complete loading
    # TRADE-OFF: ~10 seconds load time, ~6GB memory
    
    print(f"Loading model from {model_path}...")
    # WHAT: Inform user loading started
    # WHY: Loading takes time, provide feedback
    # BENEFIT: User knows system is working
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # WHAT: Load tokenizer from saved model directory
    # WHY: Convert text ↔ tokens
    # HOW: Reads tokenizer.json, tokenizer_config.json
    # BENEFIT: Guaranteed compatibility with model
    # TRADE-OFF: Requires trust_remote_code for Qwen models
    
    # Set pad_token if not defined
    if tokenizer.pad_token is None:
        # WHAT: Check if padding token exists
        # WHY: Some tokenizers don't define pad_token
        # BENEFIT: Prevents errors during generation
        
        tokenizer.pad_token = tokenizer.eos_token
        # WHAT: Use end-of-sequence token for padding
        # WHY: Common workaround for missing pad_token
        # BENEFIT: Simple solution
        # TRADE-OFF: Padding and EOS share same token ID
    
    model = AutoModelForCausalLM.from_pretrained(
        # WHAT: Load pre-trained causal language model
        # WHY: This is fine-tuned Qwen2-1.5B
        # HOW: Loads weights from adapter_model.bin + base
        # BENEFIT: Single call loads everything
        # TRADE-OFF: Large memory usage (~6GB)
        
        model_path,
        # WHAT: Path to saved model directory
        # VALUE: "./qwen_finetune"
        # BENEFIT: Centralized model location
        
        dtype=torch.float32,
        # WHAT: Load weights in 32-bit floating point
        # WHY: CPU requires float32 (no fp16 support)
        # BENEFIT: Full precision, stable inference
        # TRADE-OFF: 2x memory vs float16, slower
        
        low_cpu_mem_usage=True,
        # WHAT: Memory optimization flag
        # WHY: Reduce peak memory during loading
        # HOW: Load weights incrementally
        # BENEFIT: Can load larger models
        # TRADE-OFF: Slightly slower load time
        
        trust_remote_code=True
        # WHAT: Allow custom code execution
        # WHY: Qwen models need custom modeling code
        # BENEFIT: Full model functionality
        # TRADE-OFF: Security risk (trust source)
    )
    
    model.eval()
    # WHAT: Set model to evaluation mode
    # WHY: Disable dropout, use deterministic behavior
    # HOW: Sets model.training = False
    # BENEFIT: Consistent, reproducible outputs
    # TRADE-OFF: Must remember to call
    
    print("Model loaded successfully!\n")
    # WHAT: Confirm loading completed
    # WHY: User knows ready to chat
    
    return model, tokenizer
    # Returns: Tuple of (model, tokenizer)


# ============================================================================
# RESPONSE GENERATION FUNCTION
# ============================================================================

def generate_response(model, tokenizer, prompt, max_length=200, temperature=0.7):
    """Generate response from the model"""
    # WHAT: Generate text response to prompt
    # WHY: Core inference function
    # HOW: Tokenize input, generate tokens, decode output
    # BENEFIT: Configurable generation parameters
    # TRADE-OFF: Slow on CPU (~30 sec)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    # WHAT: Convert text to token IDs
    # WHY: Model operates on tokens, not strings
    # HOW: Tokenize, truncate if >512, return PyTorch tensors
    # BENEFIT: Handles variable-length prompts
    # TRADE-OFF: Long prompts get truncated
    # RETURNS: Dict with input_ids, attention_mask
    
    with torch.no_grad():
        # WHAT: Disable gradient computation
        # WHY: Inference doesn't need gradients
        # HOW: Context manager disables autograd
        # BENEFIT: Faster, less memory (~30% savings)
        # TRADE-OFF: Can't use for training (but we don't need to)
        
        outputs = model.generate(
            # WHAT: Autoregressive text generation
            # WHY: Produce response token-by-token
            # HOW: Samples from output distribution
            # BENEFIT: Flexible, many control parameters
            # TRADE-OFF: Slow (sequential process)
            
            **inputs,
            # WHAT: Unpack input_ids and attention_mask
            # WHY: generate() expects keyword arguments
            # BENEFIT: Clean syntax
            
            max_new_tokens=max_length,
            # WHAT: Maximum tokens to generate
            # WHY: Control response length, prevent runaway
            # VALUE: 200 tokens (~150 words)
            # BENEFIT: Bounded computation time
            # TRADE-OFF: May cut off longer responses
            
            temperature=temperature,
            # WHAT: Sampling randomness parameter
            # WHY: Control diversity vs coherence
            # VALUE: 0.7 = moderate creativity
            # BENEFIT: More interesting than greedy
            # TRADE-OFF: Less deterministic
            # SCALE:
            #   - 0.0: Always pick most likely (deterministic)
            #   - 0.7: Good balance (default)
            #   - 1.0: Sample from true distribution
            #   - >1.5: Very random, often incoherent
            
            do_sample=True,
            # WHAT: Enable probabilistic sampling
            # WHY: Use temperature/top_p (vs greedy)
            # BENEFIT: More diverse, natural outputs
            # TRADE-OFF: Non-deterministic
            # NOTE: Must be True to use temperature
            
            top_p=0.9,
            # WHAT: Nucleus sampling threshold
            # WHY: Sample from top 90% probability mass
            # HOW: Only consider tokens in top 90% cumulative prob
            # BENEFIT: Avoid very unlikely tokens
            # TRADE-OFF: May exclude creative options
            # WORKS WITH: do_sample=True
            
            top_k=50,
            # WHAT: Top-K sampling
            # WHY: Only consider top 50 most likely tokens
            # BENEFIT: Filter out very unlikely tokens
            # TRADE-OFF: May limit creativity
            # INTERACTION: Used alongside top_p
            
            pad_token_id=tokenizer.eos_token_id
            # WHAT: Token ID for padding
            # WHY: Prevent warnings during generation
            # BENEFIT: Clean output, no errors
            # TRADE-OFF: None
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # WHAT: Convert token IDs back to text
    # WHY: Need human-readable response
    # HOW: outputs[0] = first sequence in batch
    # BENEFIT: Clean text without <EOS>, <PAD>
    # TRADE-OFF: Can't see special tokens
    
    return response
    # Returns: Generated text string


# ============================================================================
# INTERACTIVE CHAT LOOP FUNCTION
# ============================================================================

def chat_loop(model, tokenizer):
    """Interactive chat loop"""
    # WHAT: Main interactive chat interface
    # WHY: Allow multi-turn conversation
    # HOW: Loop reading input, generating response, printing
    # BENEFIT: Natural chat experience
    # TRADE-OFF: No persistence, simple history management
    
    print("="*60)
    print("Fine-tuned Model Chat Interface")
    print("="*60)
    print("Commands:")
    print("  - Type your message to get a response")
    print("  - Type 'quit' or 'exit' to end the chat")
    print("  - Type 'clear' to clear conversation history")
    print("="*60)
    print()
    # WHAT: Display welcome banner and instructions
    # WHY: User knows how to interact
    # BENEFIT: Self-documenting interface
    # TRADE-OFF: Takes screen space
    
    conversation_history = []
    # WHAT: List to store conversation turns
    # WHY: Maintain context for multi-turn chat
    # FORMAT: ["User: ...", "Assistant: ...", "User: ..."]
    # BENEFIT: Model sees conversation context
    # TRADE-OFF: Grows unbounded (memory leak risk)
    
    while True:
        # WHAT: Infinite loop for continuous chat
        # WHY: Keep chatting until user quits
        # BENEFIT: Natural conversation flow
        # TRADE-OFF: Must explicitly break/exit
        
        try:
            # WHAT: Try-except for error handling
            # WHY: Gracefully handle errors, interrupts
            # BENEFIT: Doesn't crash on unexpected input
            # TRADE-OFF: May hide bugs
            
            user_input = input("You: ").strip()
            # WHAT: Read user input from console
            # WHY: Get next message
            # HOW: input() blocks until Enter pressed
            # BENEFIT: Simple, standard input
            # TRADE-OFF: Can't handle multi-line easily
            
            if not user_input:
                continue
                # WHAT: Skip empty inputs
                # WHY: No point processing blank lines
                # BENEFIT: Clean interaction
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                # WHAT: Check for exit commands
                # WHY: Provide multiple ways to quit
                # BENEFIT: User-friendly (multiple options)
                # TRADE-OFF: More code to maintain
                
                print("\nGoodbye!")
                break
                # WHAT: Exit the chat loop
                # WHY: User wants to quit
            
            if user_input.lower() == 'clear':
                # WHAT: Check for clear command
                # WHY: Allow resetting conversation
                # BENEFIT: Start fresh without restarting
                
                conversation_history = []
                # WHAT: Reset history to empty list
                # WHY: Clear all context
                # BENEFIT: Model doesn't see old conversation
                # TRADE-OFF: Loses all context (intentional)
                
                print("\nConversation cleared!\n")
                continue
                # WHAT: Skip to next iteration
                # WHY: Don't generate response for clear command
            
            # Add user input to history
            conversation_history.append(f"User: {user_input}")
            # WHAT: Store user message in history
            # WHY: Model needs to see user's input
            # FORMAT: "User: <message>"
            # BENEFIT: Clear speaker attribution
            
            # Create prompt from conversation history
            prompt = "\n".join(conversation_history[-5:]) + "\nAssistant:"
            # WHAT: Build prompt from recent history
            # WHY: Give model conversation context
            # HOW: Join last 5 turns, add "Assistant:" to trigger response
            # BENEFIT: Model sees recent context
            # TRADE-OFF: Only last 5 turns (older context lost)
            # NOTE: [-5:] limits context window to prevent token overflow
            
            # Generate response
            print("Assistant: ", end="", flush=True)
            # WHAT: Print "Assistant: " without newline
            # WHY: Response appears on same line
            # HOW: end="" prevents newline, flush=True shows immediately
            # BENEFIT: Clean, chat-like appearance
            # TRADE-OFF: Slightly more complex
            
            response = generate_response(model, tokenizer, prompt)
            # WHAT: Generate model response
            # WHY: Get AI's reply to conversation
            # HOW: Calls generate_response() with context
            # BENEFIT: Contextual responses
            # TRADE-OFF: Slow (~30 sec on CPU)
            
            # Extract only the new response (remove prompt)
            if "Assistant:" in response:
                # WHAT: Check if response contains "Assistant:"
                # WHY: Model may include prompt in output
                # BENEFIT: Robust to different model behaviors
                
                response = response.split("Assistant:")[-1].strip()
                # WHAT: Extract everything after last "Assistant:"
                # WHY: Remove prompt, keep only new content
                # HOW: Split on "Assistant:", take last part, strip whitespace
                # BENEFIT: Clean response extraction
                # TRADE-OFF: Fails if "Assistant:" appears in response content
            
            print(response)
            print()
            # WHAT: Print response and blank line
            # WHY: Display AI's reply, visual separation
            # BENEFIT: Clean formatting
            
            # Add assistant response to history
            conversation_history.append(f"Assistant: {response}")
            # WHAT: Store AI response in history
            # WHY: Future turns need full conversation
            # FORMAT: "Assistant: <response>"
            # BENEFIT: Complete conversation context
            # TRADE-OFF: History keeps growing
            
        except KeyboardInterrupt:
            # WHAT: Handle Ctrl+C interrupt
            # WHY: User may want to quit mid-generation
            # BENEFIT: Graceful exit
            # TRADE-OFF: May interrupt mid-response
            
            print("\n\nChat interrupted. Goodbye!")
            break
            # WHAT: Exit loop cleanly
            
        except Exception as e:
            # WHAT: Catch any other errors
            # WHY: Don't crash on unexpected issues
            # BENEFIT: Robust to errors
            # TRADE-OFF: May hide bugs
            
            print(f"\nError: {e}")
            print("Please try again.\n")
            # WHAT: Show error, continue chatting
            # WHY: Let user retry
            # BENEFIT: Recoverable from errors


# ============================================================================
# SINGLE PROMPT FUNCTION
# ============================================================================

def single_prompt(model, tokenizer, prompt):
    """Single prompt mode - get one response and exit"""
    # WHAT: Generate one response and exit
    # WHY: Quick testing, scripting use cases
    # HOW: Generate, print, return
    # BENEFIT: Fast for single queries
    # TRADE-OFF: No conversation context
    
    print(f"\nPrompt: {prompt}\n")
    # WHAT: Echo the prompt
    # WHY: User sees what was processed
    # BENEFIT: Confirmation
    
    response = generate_response(model, tokenizer, prompt)
    # WHAT: Generate response
    # WHY: Get model output
    # HOW: Direct call to generate_response
    # BENEFIT: Simple, clean
    # TRADE-OFF: No context from previous turns
    
    print(f"Response: {response}\n")
    # WHAT: Display response
    # WHY: Show result to user
    # BENEFIT: Clear formatting
    
    return response
    # Returns: Response string (for programmatic use)


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # WHAT: Main execution block
    # WHY: Run chat when script executed
    # BENEFIT: Importable as module or executable as script
    
    import sys
    # WHAT: System module for command-line args
    # WHY: Check if prompt provided as argument
    # BENEFIT: Two modes from one script
    
    # Load the fine-tuned model
    model_path = "./qwen_finetune"
    # WHAT: Path to saved model
    # WHY: Where training saved the model
    # BENEFIT: Standard location
    # TRADE-OFF: Hardcoded (could be CLI arg)
    
    try:
        # WHAT: Try-except for error handling
        # WHY: Gracefully handle missing model
        # BENEFIT: Helpful error messages
        
        model, tokenizer = load_finetuned_model(model_path)
        # WHAT: Load model and tokenizer
        # WHY: Need both for generation
        # HOW: Calls load_finetuned_model()
        # BENEFIT: Ready for either mode
        # TRADE-OFF: ~10 sec load time
        
        # Check if a prompt was provided as command line argument
        if len(sys.argv) > 1:
            # WHAT: Check if CLI arguments provided
            # WHY: Determine which mode to use
            # HOW: sys.argv[0] is script name, [1+] are args
            # BENEFIT: Flexible invocation
            
            # Single prompt mode
            prompt = " ".join(sys.argv[1:])
            # WHAT: Join all arguments into single prompt
            # WHY: Allow multi-word prompts without quotes
            # EXAMPLE: python chat.py Hello how are you
            # BENEFIT: User-friendly CLI
            # TRADE-OFF: Can't distinguish multiple args
            
            single_prompt(model, tokenizer, prompt)
            # WHAT: Generate one response
            # WHY: User provided prompt
            # BENEFIT: Quick usage
            
        else:
            # WHAT: No CLI arguments
            # WHY: User wants interactive chat
            
            # Interactive chat mode
            chat_loop(model, tokenizer)
            # WHAT: Start interactive chat
            # WHY: Default mode
            # BENEFIT: Full conversation experience
            
    except FileNotFoundError:
        # WHAT: Handle missing model directory
        # WHY: Common error (model not trained yet)
        # BENEFIT: Helpful guidance
        
        print(f"Error: Model not found at '{model_path}'")
        print("Please make sure you have trained the model first by running:")
        print("  python src/train_with_config.py")
        # WHAT: Display helpful error message
        # WHY: Tell user how to fix
        # BENEFIT: Self-documenting
        
    except Exception as e:
        # WHAT: Catch any other errors
        # WHY: Don't crash silently
        # BENEFIT: See what went wrong
        
        print(f"Error loading model: {e}")
        # WHAT: Display error message
        # WHY: Debug information
```

---

## Usage Examples

### Interactive Chat Mode
```bash
python src/chat.py
```
Then type messages and press Enter.

### Single Prompt Mode
```bash
python src/chat.py "Explain what a GPU does"
python src/chat.py What is machine learning
```

### Programmatic Use
```python
from src.chat import load_finetuned_model, generate_response

model, tokenizer = load_finetuned_model()
response = generate_response(model, tokenizer, "Hello!")
print(response)
```

---

## Chat Commands

| Command | Action | Example |
|---------|--------|---------|
| `quit`, `exit`, `q` | Exit chat | `quit` |
| `clear` | Clear history | `clear` |
| `(any text)` | Send message | `Hello!` |
| `Ctrl+C` | Interrupt/quit | (keyboard) |

---

## Generation Parameters Explained

### Temperature (0.7)
- **Low (0.1-0.5)**: More focused, deterministic, repetitive
- **Medium (0.6-0.9)**: Balanced creativity and coherence
- **High (1.0-2.0)**: More random, creative, potentially incoherent
- **Current**: 0.7 = good default

### Top-P (0.9)
- **Nucleus sampling**: Sample from smallest set with 90% probability mass
- **Effect**: Filters out very unlikely tokens
- **Trade-off**: Balance diversity and quality

### Top-K (50)
- **Effect**: Only consider top 50 tokens at each step
- **Benefit**: Speed up generation, filter noise
- **Trade-off**: May limit creativity

---

## Conversation History Management

**Current Behavior:**
- Stores all turns in `conversation_history`
- Only last 5 turns sent to model (context window)
- Cleared with `clear` command
- Lost on exit (no persistence)

**Improvements:**
- Save to file (JSON, CSV)
- Load previous conversations
- Configurable context window size
- Automatic summarization for long chats

---

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Load Time** | ~10 seconds | One-time on startup |
| **Generation Speed** | ~30 sec/response | CPU inference |
| **Memory Usage** | ~6GB RAM | Model loaded |
| **Max History** | 5 turns | Context window limit |

---

## Common Issues & Solutions

**Issue**: "Model not found"  
**Solution**: Run `python src/train_with_config.py` first

**Issue**: Slow responses  
**Solution**: CPU is slow, use GPU or reduce max_new_tokens

**Issue**: Repetitive responses  
**Solution**: Increase temperature, use top_p/top_k

**Issue**: Incoherent responses  
**Solution**: Lower temperature, check model training

**Issue**: Memory error  
**Solution**: Reduce max_new_tokens, clear history

---

## Related Files
- `train_with_config.py` - Train model used here
- `evaluation.py` - Evaluate model quality
- `qwen_finetune/` - Model directory
