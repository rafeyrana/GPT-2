# GPT-2
## Recreating the GPT-2 Model (originally in TensorFlow) in PyTorch
This implementation of GPT-2 in PyTorch is based on the original paper ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) by Radford et al. (2019). The model is a decoder-only transformer architecture, which is different from the original transformer implementation proposed in the "Attention Is All You Need" Paper in 2017.
<img width="1061" alt="image" src="https://github.com/user-attachments/assets/61b564e7-ec92-490f-9a80-62a37a0d8e81" />
## Optimizations and Technical Details

### Weight Sharing Scheme
The model uses a weight sharing scheme, where the input embedding matrix and the output matrix before the softmax layer are shared. This scheme reduces the number of parameters and makes training more efficient.

### Flash Attention
The model uses Flash Attention, a kernel fusion algorithm that is 7.6% faster than the traditional self-attention mechanism.

### Automated Mixed Precision (AMP)
The model uses AMP to reduce the precision of the tensor representation, which speeds up training and reduces memory usage.

### Fused AdamW Optimizer
The model uses a fused AdamW optimizer, which is a kernel fusion algorithm that combines the Adam optimizer with weight decay.

### Gradient Accumulation
The model uses gradient accumulation to update the gradients in batches, which reduces the memory usage and speeds up training.

### Learning Rate Scheduler
The model uses a Cosine learning rate scheduler that warms up the learning rate from 0 to a maximum value, and then decays it to a minimum value.

### Distributed Training
The model is trained using microbatch and DDP techniques on 8 GPUs in parallel with loss sync.

Note: This implementation is for educational purposes only and may not achieve the same performance as the original GPT-2 model.

## Training

### Standard Text Training
The model can be trained on a plain text file (default: `input.txt`). The `train.py` script handles the training process. You can configure various parameters like batch size, sequence length, learning rate, etc., directly in the script or via command-line arguments (see `python train.py --help`).

To start training with the default settings (single GPU/CPU, using `input.txt`):
```bash
python train.py
```

For Distributed Data Parallel (DDP) training, use `torchrun`:
```bash
# Example for 2 GPUs on a single node
torchrun --nproc_per_node=2 train.py
```

### Training with Conversational Datasets (Alpaca-style)

This implementation now supports training with conversational datasets formatted in an Alpaca-like style. This is useful for fine-tuning the model for instruction-following or question-answering tasks.

**Dataset Format:**

The dataset should be a JSON or JSONL file. Each entry should be a JSON object containing at least an `"instruction"` and an `"output"` field. An optional `"input"` field can be included for additional context to the instruction.

Example JSON entry:
```json
{
    "instruction": "What is the capital of France?",
    "input": "",
    "output": "The capital of France is Paris."
}
```
Or with additional input:
```json
{
    "instruction": "Write a short story about a robot who learns to paint.",
    "input": "The robot's name is Unit 734.",
    "output": "Unit 734 had always processed logic, never art. One day, it found a discarded set of paints..."
}
```
A sample file `alpaca_sample.json` is provided in the repository.

**Training Command:**

A convenience script `train_alpaca.sh` is provided to simplify training with Alpaca-style datasets. You can modify the configuration variables at the top of this script:

-   `NPROC_PER_NODE`: Number of GPUs to use (set to 1 for CPU/single GPU).
-   `DATASET_PATH`: Path to your Alpaca-style JSON/JSONL dataset (defaults to `alpaca_sample.json`).
-   `BATCH_SIZE`, `SEQ_LEN`, `MAX_STEPS`, `WARMUP_STEPS`, `MAX_LR`, `WEIGHT_DECAY`: Training hyperparameters.

To run the training using this script:
```bash
chmod +x train_alpaca.sh
./train_alpaca.sh
```

Alternatively, you can run `train.py` directly and specify the dataset type and path:
```bash
# Example for single GPU/CPU using alpaca_sample.json
python train.py --dataset_type alpaca --dataset_path alpaca_sample.json --batch_size 2 --seq_len 64 --max_steps 20

# Example for DDP with 2 GPUs
torchrun --nproc_per_node=2 train.py --dataset_type alpaca --dataset_path your_dataset.jsonl --batch_size <your_batch_size> --seq_len <your_seq_len>
```
Refer to `python train.py --help` for all available command-line options.

## Future Work and Improvements
- Train on a larger dataset with more compute. (Dataset we can use is the LLM cleaned [Huggingface Fineweb 10B Token](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) but i dont have the compute to train it :(  )
- Add detailed Evaluation Metrics.
