# openact-core

Core reading and alignment library for OpenAct.

## Installation
```bash
pip install openact-core
```

## Usage
```python
from openact_core import Run

run = Run("path/to/run")
sample = run[0]
hs = sample.get_text_hidden_states("answer")
```
