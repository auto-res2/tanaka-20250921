
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
license: other
license_name: gemma-terms-of-use
license_link: https://ai.google.dev/gemma/terms
base_model: google/gemma-7b
tags:
- alignment-handbook
- trl
- sft
- generated_from_trainer
- trl
- sft
- generated_from_trainer
datasets:
- HuggingFaceH4/deita-10k-v0-sft
model-index:
- name: zephyr-7b-gemma-sft
  results: []
language:
- en
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# zephyr-7b-gemma-sft

This model is a fine-tuned version of [google/gemma-7b](https://huggingface.co/google/gemma-7b) on the HuggingFaceH4/deita-10k-v0-sft dataset.
It achieves the following results on the evaluation set:
- Loss: 0.9732

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- distributed_type: multi-GPU
- num_devices: 16
- gradient_accumulation_steps: 2
- total_train_batch_size: 128
- total_eval_batch_size: 64
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 0.9482        | 1.0   | 299  | 0.9848          |
| 0.8139        | 2.0   | 599  | 0.9610          |
| 0.722         | 2.99  | 897  | 0.9732          |


### Framework versions

- Transformers 4.39.0.dev0
- Pytorch 2.1.2+cu121
- Datasets 2.14.6
- Tokenizers 0.15.1
Output:
{
    "extracted_code": ""
}
