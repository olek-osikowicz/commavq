# AGENTS.md - Development Guidelines for commavq

## Project Overview

commaVQ is a video compression research project using VQ-VAE for compressing driving videos and a GPT-based world model for prediction. Python 3.11 required.

## Build & Development Commands

### Package Management (uv)
```bash
uv sync              # Install dependencies
uv sync --group dev  # Install dev dependencies
```

### Running Code
```bash
# Direct Python execution
python compression/compress.py
python compression/decompress.py

# Via notebooks in notebooks/ directory
jupyter lab notebooks/
```

### Running Tests
**No test framework is currently configured.** If adding tests:
```bash
pytest                           # Run all tests
pytest tests/                   # Run tests in specific directory
pytest tests/test_file.py::test_function  # Run single test
pytest -k "test_name"           # Run tests matching pattern
```

To add tests, install pytest: `uv add --group dev pytest`

### Linting & Type Checking
**No linting tools currently configured.** To add:
```bash
ruff check .           # Lint with ruff
ruff check --fix .     # Fix lint issues
ruff format .          # Format code
mypy .                 # Type checking
```

To add linting tools:
```bash
uv add --group dev ruff mypy
```

## Code Style Guidelines

### Indentation
- **2 spaces** (not tabs). This codebase uses 2-space indentation consistently.

### Imports
- Standard library imports first
- Third-party imports second
- Local imports last
- Blank line between groups

```python
# Correct
import os
import multiprocessing
from pathlib import Path

import numpy as np
import torch

from utils.video import write_video
```

### Type Hints
- Use type hints for function parameters and return values
- Use `Optional[X]` instead of `X | None`
- Use `from typing import Optional, List, Dict, Any` as needed

```python
def compress_tokens(tokens: np.ndarray) -> bytes:
    tokens = tokens.astype(np.int16).reshape(-1, 128).T.ravel().tobytes()
    return lzma.compress(tokens)
```

### Naming Conventions
- **Functions/variables**: `snake_case` (e.g., `write_video`, `compression_rate`)
- **Classes**: `PascalCase` (e.g., `GPTConfig`, `KVCache`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `CROP_SIZE`, `OUTPUT_SIZE`)
- **Private functions**: prefix with underscore (e.g., `_internal_func`)

### Dataclasses for Configuration
Use `@dataclass` for configuration classes:

```python
@dataclass
class GPTConfig:
    block_size: int = 20 * 129
    vocab_size: int = 1025
    n_layer: int = 24
```

### Error Handling
- Use assertions for invariants (e.g., `assert input_pos.shape[0] == k_val.shape[2]`)
- Raise descriptive exceptions for recoverable errors
- Let exceptions propagate for unexpected errors

### Torch Best Practices
- Use `@torch.no_grad()` decorator for inference
- Use `.to(dtype=...)` for type conversion
- Use `nn.Module` base class for models
- Register non-trainable buffers with `register_buffer`

```python
@torch.no_grad()
def generate(self, prompt: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    ...
```

### NumPy Conventions
- Use `np.array()` for array conversion
- Use `np.ndarray` for type hints
- Prefer in-place operations where performance matters

### File Structure
```
commavq/
├── compression/       # Compression/decompression scripts
├── utils/            # Utility modules
├── nanogpt/          # Data preparation scripts
├── notebooks/        # Jupyter notebooks for exploration
├── gpt2m/            # Model weights (submodule)
└── pyproject.toml    # Project configuration
```

### Shebang for Scripts
Include shebang for executable scripts:
```python
#!/usr/bin/env python3
```

### Docstrings
- Use triple quotes for module-level and function docstrings
- Keep brief, descriptive comments explaining non-obvious logic
- Document complex math operations or compression tricks

```python
def compress_tokens(tokens: np.ndarray) -> bytes:
    tokens = tokens.astype(np.int16).reshape(-1, 128).T.ravel().tobytes() # transposing increases compression rate ;)
    return lzma.compress(tokens)
```

## Common Development Tasks

### Adding a New Compression Algorithm
1. Create new file in `compression/` directory
2. Implement `compress()` and `decompress()` functions
3. Use the datasets library to load data:
```python
from datasets import load_dataset
data_files = {'train': ['data-0000.tar.gz']}
ds = load_dataset('commaai/commavq', num_proc=num_proc, data_files=data_files)
```

### Working with the Model
```python
from utils.gpt import GPT, GPTConfig
config = GPTConfig()
model = GPT(config)
model.load_state_dict_from_url()
```

### Working with Video Data
```python
from utils.video import write_video, read_video, transform_img
frames = read_video("input.mp4")
transformed = transform_img(frames[0])
```

## Dependencies
- Python 3.11+
- torch==2.2.2
- numpy<2
- datasets==4.0.0
- opencv-python
- einops
- matplotlib>=3.10.8
- tqdm
