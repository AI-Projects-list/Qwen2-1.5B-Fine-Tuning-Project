# COMMENTED_CODE Directory

## Overview
This directory contains comprehensive, line-by-line documentation for all code files in the Qwen2-1.5B Fine-Tuning Project.

Each markdown file provides:
- **WHAT**: What the code does
- **WHY**: Why it exists and design rationale
- **HOW**: How it works internally
- **BENEFIT**: Advantages of the approach
- **TRADE-OFF**: Disadvantages and limitations

---

## Documentation Files

### 1. train_with_config.py.md
**Purpose**: Production training script with YAML configuration

**Key Topics:**
- Configuration loading with PyYAML
- Model and tokenizer initialization
- LoRA adapter setup and application
- Dataset tokenization pipeline
- Training arguments configuration
- Hugging Face Trainer usage

**Use When:**
- Understanding the main training workflow
- Learning parameter-efficient fine-tuning
- Configuring production training runs
- Debugging training issues

---

### 2. evaluation.py.md
**Purpose**: Model evaluation with quality metrics

**Key Topics:**
- Model loading for inference
- Response generation
- Quality assessment heuristics
- Sklearn metrics (accuracy, precision, recall, F1)
- Confusion matrix interpretation
- Dataset vs single-text evaluation

**Use When:**
- Understanding evaluation methodology
- Learning classification metrics
- Assessing model performance
- Debugging quality issues

---

### 3. chat.py.md
**Purpose**: Interactive chat interface for inference

**Key Topics:**
- Model loading for chat
- Conversation history management
- Response generation parameters
- Interactive vs single-prompt modes
- Temperature and sampling strategies
- Error handling and graceful exit

**Use When:**
- Building chat applications
- Understanding inference parameters
- Learning conversation management
- Testing trained models

---

### 4. main.py.md
**Purpose**: Simple training script (legacy/educational)

**Key Topics:**
- Self-contained training function
- Hardcoded parameter approach
- Linear training workflow
- Comparison with train_with_config.py
- When to use each approach

**Use When:**
- Learning training fundamentals
- Quick prototyping
- Understanding basic workflow
- Educational purposes

---

### 5. config.yml.md
**Purpose**: Central configuration file documentation

**Key Topics:**
- Every configuration parameter explained
- Model, LoRA, dataset, training sections
- Parameter interactions and dependencies
- Common modifications and presets
- Validation checklist

**Use When:**
- Configuring training experiments
- Understanding hyperparameters
- Tuning model performance
- Troubleshooting configuration issues

---

## How to Use This Documentation

### For Beginners
1. Start with **main.py.md** - simplest training flow
2. Read **config.yml.md** - understand all parameters
3. Study **train_with_config.py.md** - production approach
4. Review **evaluation.py.md** - assess model quality
5. Explore **chat.py.md** - use trained models

### For Experienced Users
- Use as reference for specific parameters
- Jump to relevant sections using headers
- Compare trade-offs for different approaches
- Find typical values and recommendations

### For Debugging
- Check "Common Issues & Solutions" sections
- Review parameter explanations
- Verify configuration requirements
- Understand error messages

---

## Documentation Structure

Each file follows this template:

```markdown
# filename - Comprehensive Documentation

## Overview
Purpose, what it does, why it exists, how it works

## Benefits
✅ Advantages and strengths

## Trade-offs
⚠️ Disadvantages and limitations

## Line-by-Line Commented Code
Complete code with detailed inline comments

## Usage Examples
Practical usage demonstrations

## Related Files
Connections to other components
```

---

## Key Concepts Explained

### LoRA (Low-Rank Adaptation)
- **What**: Parameter-efficient fine-tuning method
- **Why**: Train large models with minimal memory
- **How**: Freeze base model, add small trainable adapters
- **Benefit**: ~0.56% trainable parameters vs 100%
- **Files**: train_with_config.py.md, config.yml.md

### Tokenization
- **What**: Convert text to numeric tokens
- **Why**: Models operate on numbers, not strings
- **How**: Use AutoTokenizer with truncation/padding
- **Benefit**: Handle variable-length inputs
- **Files**: train_with_config.py.md, main.py.md

### Gradient Accumulation
- **What**: Accumulate gradients over multiple batches
- **Why**: Simulate larger batch size without more memory
- **How**: Forward pass N times, one optimizer step
- **Benefit**: Better gradients, fits in limited memory
- **Files**: train_with_config.py.md, config.yml.md

### Quality Metrics
- **What**: Accuracy, precision, recall, F1 score
- **Why**: Quantitatively measure model performance
- **How**: Compare predictions with ground truth
- **Benefit**: Objective performance assessment
- **Files**: evaluation.py.md

---

## Parameter Quick Reference

### Most Important Parameters

| Parameter | Location | Typical Value | Impact |
|-----------|----------|---------------|--------|
| **model.name** | config.yml | Qwen/Qwen2-1.5B | Which model |
| **lora.r** | config.yml | 64 | Adapter capacity |
| **learning_rate** | config.yml | 2e-4 | Training speed |
| **batch_size** | config.yml | 1 | Memory usage |
| **max_length** | config.yml | 512 | Context window |
| **num_epochs** | config.yml | 3 | Training duration |

### For CPU Training

| Parameter | Value | Why |
|-----------|-------|-----|
| **dtype** | float32 | CPU requirement |
| **fp16** | false | CPU doesn't support |
| **use_cpu** | true | Force CPU |
| **batch_size** | 1 | Limited memory |

### For GPU Training

| Parameter | Value | Why |
|-----------|-------|-----|
| **dtype** | float16 | 2x speedup |
| **fp16** | true | Enable mixed precision |
| **use_cpu** | false | Use GPU |
| **batch_size** | 4-8 | More memory available |

---

## Learning Path

### Week 1: Fundamentals
- Read main.py.md completely
- Understand basic training flow
- Run simple training
- Review config.yml.md sections

### Week 2: Configuration
- Study all config.yml parameters
- Experiment with different values
- Read train_with_config.py.md
- Compare with main.py approach

### Week 3: Evaluation
- Read evaluation.py.md
- Understand metrics
- Run evaluations
- Analyze results

### Week 4: Deployment
- Study chat.py.md
- Build chat applications
- Experiment with generation parameters
- Deploy trained models

---

## Common Questions Answered

### "Which training script should I use?"
- **Beginner/Learning**: main.py
- **Production/Experiments**: train_with_config.py
- **Quick Test**: main.py
- **Multiple Configs**: train_with_config.py

### "How do I change [parameter]?"
- Check config.yml.md for explanation
- See typical values and trade-offs
- Modify config.yml
- Re-run training

### "What's a good LoRA rank?"
- **8-16**: Simple tasks, very efficient
- **64**: Good default (recommended)
- **128-256**: Complex tasks, more memory

### "How much training data do I need?"
- **Minimum**: 10-50 samples (proof of concept)
- **Good**: 100-1000 samples (decent quality)
- **Better**: 10K+ samples (production quality)

### "Why is training slow?"
- CPU is 10-100x slower than GPU
- Solutions:
  - Use GPU if available
  - Reduce max_length
  - Reduce batch size
  - Reduce num_epochs

---

## Troubleshooting Guide

### Import Errors
- **File**: Requirements section in each .md
- **Solution**: Install missing dependencies

### Out of Memory
- **File**: config.yml.md → memory parameters
- **Solution**: Reduce batch_size, max_length

### Poor Quality
- **File**: evaluation.py.md → metrics
- **Solution**: More data, higher LoRA rank, more epochs

### Configuration Errors
- **File**: config.yml.md → validation checklist
- **Solution**: Check syntax, required parameters

---

## Additional Resources

### External Documentation
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT Library](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

### Related Project Files
- `../PROJECT_FLOW.md` - Visual workflow diagrams
- `../README.md` - Project overview
- `../src/` - Actual source code

---

## Contributing to Documentation

### Adding Comments
1. Explain WHAT the code does
2. Explain WHY it exists
3. Explain HOW it works
4. List BENEFITS
5. List TRADE-OFFS

### Example Format
```python
variable = some_function(param)
# WHAT: Brief description of what this line does
# WHY: Why this approach was chosen
# HOW: How it works internally
# BENEFIT: Advantages of this approach
# TRADE-OFF: Disadvantages or limitations
```

---

## License
Same as parent project

## Last Updated
December 10, 2025
