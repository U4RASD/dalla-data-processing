# Readability Scoring

Calculate readability scores using Flesch Reading Ease and Osman methods.

## Installation

This feature requires the `[readability]` extra:

```bash
pip install "dalla-data-processing[readability]"
# or install all features: pip install "dalla-data-processing[all]"
```

## CLI Usage

**Command:** `dalla-dp readability [OPTIONS]`

**Arguments:**
- `--add-ranks` / `--no-ranks` - Add ranking and level columns (default: True)

**Examples:**
```bash
dalla-dp -i ./data/raw -o ./data/scored readability

dalla-dp -i ./data/raw -o ./data/scored readability --no-ranks

dalla-dp -i ./data/raw -o ./data/scored -c content readability
```

## Python API

```python
from datasets import load_from_disk
from dalla_data_processing.readability import score_readability

# Load dataset
dataset = load_from_disk("./data/raw")

scored = score_readability(dataset, column="text", add_ranks=True)

# Save result
scored.save_to_disk("./data/scored")
```

## Readability Levels

- `0`: Very Easy
- `1`: Easy
- `2`: Medium
- `3`: Difficult
- `4`: Very Difficult
