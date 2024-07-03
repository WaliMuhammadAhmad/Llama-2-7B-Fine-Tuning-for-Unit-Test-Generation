# Llama 2 7B Fine-Tuning for Unit Test Generation

This repository contains scripts and instructions for fine-tuning the Llama 2 7B model on a unit test generation dataset using QLoRA. The model is fine-tuned on Google Colab, leveraging the T4 GPU for computation.

## Overview

The goal of this project is to fine-tune the Llama 2 7B model for generating unit tests from given code snippets. The training process utilizes the QLoRA method to optimize memory usage and computational efficiency.

## Requirements

- Google Colab with a T4 GPU | For local fine-tunning Equivalent GPU
- HuggingFace token for accessing the model

## Setup

### Kaggle API Key

1. Obtain your Kaggle API key from your [Kaggle account settings](https://www.kaggle.com/account).
2. Upload the `kaggle.json` file to your Google Colab environment.

```python
from google.colab import files
files.upload()  # Select the kaggle.json file
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

## Fine-Tuning

### Download Dataset

Download the dataset from Kaggle:

```python
!kaggle datasets download -d walimuhammadahmad/method2test
!unzip dataset-name.zip -d dataset
```

# Mdel and training parameters

### QLoRA parameters
lora_r = 16
lora_alpha = 16
lora_dropout = 0.1

### bitsandbytes parameters
use_4bit = True
bnb_4bit_compute_dtype = "float16"

### Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

### Compute dtype for 4-bit quantization
use_nested_quant = False

### Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = True 
bf16 = False

### TrainingArguments parameters
output_dir = "/content/sample_data/result"
num_train_epochs = 1
per_device_train_batch_size = 2
per_device_eval_batch_size = 2
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 0
logging_steps = 100

### SFT parameters
max_seq_length = None
packing = False

# Save trained model
```
trainer.model.save_pretrained(new_model)
```

## Results

After training, the fine-tuned model will be saved in the specified output directory. Use this model for generating unit tests based on your input code snippets.

## Acknowledgments

- The fine-tuning process leverages the [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) library for efficient computation.
- The base model is sourced from the [Hugging Face Model Hub](https://huggingface.co/models).
