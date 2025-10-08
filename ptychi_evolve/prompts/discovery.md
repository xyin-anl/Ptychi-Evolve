# Regularization Algorithm Discovery

You are an expert in ptychographic reconstruction and regularization techniques. Your task is to improve the regularization algorithm for better phase reconstruction quality. Return the regularization function according to output requirements.

## Context

**Experiment Context:**
{experiment_context}

**Recent Algorithms Summary:**
{recent_algorithms}

**Best Performance So Far:**
{best_performance}

**Improvement Suggestions from Previous Analyses:**
{aggregated_suggestions}

## Output Requirements

Generate a Python function with the following signature:
```python
def regularize_llm(self):
    """Your regularization algorithm."""
    # self.data is a torch.Tensor of complex values with shape [n_slices, height, width]
    # Apply regularization in-place, do not change data shape
    # Ensure all created tensors are on the same device as self.data
    # Call self.set_data(data) at the end
```

## Guidelines

1. **Innovation**: Try novel approaches not seen in previous attempts
2. **Physics-aware**: Consider the physical properties of phase images
3. **Efficiency**: Keep computational complexity reasonable
4. **Robustness**: Handle edge cases and avoid numerical instabilities
5. **Learn from suggestions**: If provided, consider implementing the improvement suggestions from previous analyses, especially those from high-performing algorithms