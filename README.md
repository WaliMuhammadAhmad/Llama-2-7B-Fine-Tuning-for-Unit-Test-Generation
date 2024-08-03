# Llama 2 7B Fine-Tuning for Unit Test Generation

This repository contains scripts and instructions for fine-tuning the Llama 2 7B model on a unit test generation dataset using QLoRA. The model is fine-tuned on Google Colab, using useless but useful for now, the T4 GPU for computation. The training process utilizes the QLoRA method to optimize memory usage and computational efficiency.

## Setup

### Kaggle API Key

You can also get the dataset from kaggle by following this:

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

## Results

After training, the fine-tuned model will be saved in the specified output directory. Use this model for generating unit tests at the end of script. I also added a notebook which loads the model checkpoints seperately and then evaluates the model on a sample dataset, but i failed to do this bcz i didnt fine any useful metrics for evaluation. If you find one then please consider doing a pull request and I'll let you do it!

## Acknowledgments

- The fine-tuning process usomg the [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) library for loading the model on T4 GPU.
- The base model is sourced from the [Hugging Face Model Hub](https://huggingface.co/models).