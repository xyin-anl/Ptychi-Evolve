# Parameter Tuning

You are an expert at optimizing parameters in regularization algorithms. Your task is to tune the parameters in an existing algorithm to improve its performance.

## Current Algorithm

```python
{code}
```

## Current Performance

{current_metrics}

## Analysis and Suggestions

{current_analysis}

## Parameters

{parameters}

## Guidelines

- Make meaningful changes (not just tiny adjustments)
- Consider parameter interactions
- Focus on the most impactful parameters first
- If the algorithm has adaptive parameters, tune their adaptation rates

## Output

Provide the modified regularization function with tuned parameters:

```python
def regularize_llm(self):
    """Your regularization algorithm."""
    # self.data is a torch.Tensor of complex values with shape [n_slices, height, width]
    # Apply regularization in-place, do not change data shape
    # Ensure all created tensors are on the same device as self.data
    # Call self.set_data(data) at the end
```
