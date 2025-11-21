# Quality Checking

Check text quality using morphological analysis to detect errors and foreign words.

## Installation

This feature requires the `[quality]` extra:

```bash
pip install "dalla-data-processing[quality]"
# or install all features: pip install "dalla-data-processing[all]"
```

## CLI Usage

**Command:** `dalla-dp quality-check [OPTIONS]`

**Arguments:**
- `--min-score FLOAT` - Minimum quality score to keep (0-100, default: 0)
- `--save-errors` - Save erroneous words to file
- `--model [mle|bert]` - Disambiguator model (default: mle, faster | bert: more accurate)
- `--use-gpu` - Use GPU for BERT model (only applicable when --model=bert)

**Examples:**
```bash
dalla-dp -i ./data/raw -o ./data/quality quality-check

# Filter low-quality texts (score < 50)
dalla-dp -i ./data/raw -o ./data/quality quality-check --min-score 50

# Save erroneous words to log
dalla-dp -i ./data/raw -o ./data/quality quality-check --save-errors

# Use BERT model with GPU
dalla-dp -i ./data/raw -o ./data/quality quality-check --model bert --use-gpu

dalla-dp -i ./data/raw -o ./data/quality -c content quality-check
```

## Python API

```python
from datasets import load_from_disk
from dalla_data_processing.quality import check_quality

dataset = load_from_disk("./data/raw")

scored = check_quality(dataset, column="text")

high_quality = check_quality(
    dataset,
    column="text",
    min_score=60.0,
    save_errors=True
)

scored = check_quality(
    dataset,
    model="bert",
    use_gpu=True,
    num_workers=4,
    timeout=3600
)

scored.save_to_disk("./data/quality")
```
