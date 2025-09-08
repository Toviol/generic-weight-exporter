# Generic Weight Exporter - Fashion-MNIST SCNN

## Overview

This module provides a flexible, generic function to export PyTorch model weights to C header files. It has been enhanced with an optional model parameter, making it suitable for various use cases.

## Key Features

### 🆕 Optional Model Parameter
- **`require_model=True`** (default): Original behavior, requires model architecture
- **`require_model=False`**: New behavior, exports directly from saved weights file

### 🔄 Backward Compatibility
- All existing code continues to work unchanged
- Default behavior remains the same (`require_model=True`)

### 📁 Multiple Checkpoint Formats Supported
- Direct state_dict files (our case)
- Checkpoint files with nested state_dict
- Various trainer formats

## Usage Examples

### Mode 1: With Model (Original)
```python
from export_weights_generic import export_weights_to_c_header_generic, create_scnn_complete

# Create model architecture
model = create_scnn_complete(beta=0.7)

# Export with model
export_weights_to_c_header_generic(
    model=model,
    weights_path="trained_scnn_fashion_mnist_weights_only.pt",
    header_path="output_with_model.h",
    require_model=True  # Default
)
```

### Mode 2: Without Model (New) ⭐
```python
from export_weights_generic import export_weights_to_c_header_generic

# Export directly from file - no model needed!
export_weights_to_c_header_generic(
    model=None,  # No model required!
    weights_path="trained_scnn_fashion_mnist_weights_only.pt",
    header_path="output_direct.h",
    require_model=False  # New mode
)
```

### Simple One-Liner Export
```python
from export_weights_generic import export_simple_example

# Quick export without any setup
export_simple_example()
```

## Benefits of the New Approach

| Aspect | Mode 1 (With Model) | Mode 2 (Without Model) |
|--------|-------------------|----------------------|
| **Model Architecture** | Required | Not needed ❌ |
| **Memory Usage** | Higher (loads model) | Lower (direct file read) |
| **Dependencies** | Full model dependencies | Minimal dependencies |
| **Speed** | Slower (model creation) | Faster ⚡ |
| **Use Case** | When you have the architecture | When you only have weights |

## Generated C Header Features

### Automatic Structure Detection
- **1D tensors (bias)** → `float array[size]` - Exported as simple 1D arrays
- **2D tensors (Linear weights)** → `float array[rows][cols]` - Preserved matrix structure for easy access
- **4D tensors (Conv2d weights)** → `float array[out_ch][in_ch][kernel_h][kernel_w]` - **NEW**: Full 4D structure for direct indexing
- **Other dimensions** → `float array[size]` - Flattened for compatibility

### Metadata Generation
```c
#define TENSOR_NAME_DIM0 16      // First dimension
#define TENSOR_NAME_DIM1 3       // Second dimension
#define TENSOR_NAME_NDIMS 2      // Number of dimensions
```

### Header Guards and Type Safety
```c
#ifndef FASHION_MNIST_WEIGHTS_DIRECT_H
#define FASHION_MNIST_WEIGHTS_DIRECT_H

#define NN_NUM_TENSORS 10

// Weight arrays...

#endif // FASHION_MNIST_WEIGHTS_DIRECT_H
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | Required | PyTorch model (can be None if require_model=False) |
| `weights_path` | Required | Path to .pt file |
| `header_path` | `"network_weights.h"` | Output C header file |
| `only_weights_and_bias` | `True` | Export only .weight/.bias tensors |
| `ctype` | `"weight_t"` | C type name (use "float" for direct typing) |
| `emit_typedef_if_builtin` | `True` | Generate typedef for builtin types |
| `line_wrap` | `10` | Values per line in arrays |
| `float_fmt` | `".8f"` | Float precision format |
| `verbose` | `True` | Print progress information |
| `require_model` | `True` | **NEW**: Whether model is required |

## Files Generated in This Project

1. **`fashion_mnist_weights_with_model.h`** - Generated using Mode 1 (with model)
2. **`fashion_mnist_weights_direct.h`** - Generated using Mode 2 (without model)
3. **`simple_export.h`** - Generated using simple example function

All files contain identical weight data but demonstrate different usage approaches.

## Advantages Over Original Hardcoded Function

✅ **Architecture Agnostic**: Works with any PyTorch model  
✅ **No Model Required**: New mode works without model architecture  
✅ **Automatic Shape Detection**: No manual tensor dimension mapping  
✅ **Rich Metadata**: Includes dimension information for C code  
✅ **Maintainable**: No hardcoded layer indices  
✅ **Flexible Formats**: Handles various checkpoint formats  
✅ **Memory Efficient**: Direct file reading option  

## Migration from Original Function

The new function is a drop-in replacement:

```python
# Old way (still works)
export_weights_to_c_header_generic(model, weights_path, header_path)

# New way (more efficient)
export_weights_to_c_header_generic(None, weights_path, header_path, require_model=False)
```

## Error Handling

The function includes robust error handling for:
- Missing weight files
- Invalid checkpoint formats
- Missing model when required
- Tensor processing errors

All errors provide clear, actionable messages to help debug issues.
