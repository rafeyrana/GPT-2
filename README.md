# GPT-2
## Recreating the GPT-2 Model (originally in TensorFlow) in PyTorch
This implementation of GPT-2 in PyTorch is based on the original paper ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) by Radford et al. (2019). The model is a decoder-only transformer architecture, which is different from the original transformer implementation proposed in the "Attention Is All You Need" Paper in 2017.

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

## Future Work and Improvements
- Train on a larger dataset with more compute. (Dataset we can use is the LLM cleaned [Huggingface Fineweb 10B Token](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) but i dont have the compute to train it :(  )
- Add detailed Evaluation Metrics.
