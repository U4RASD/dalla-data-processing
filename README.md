# Dalla Data Processing (dalla-dp)

A comprehensive Arabic data processing pipeline with deduplication, stemming, quality checking, and readability scoring, used for the DALLA Models.

## Compatibility

- **Linux**: Fully supported
- **macOS**: Fully supported (Intel or through rosetta)
- **Windows**: Supported through WSL (Windows Subsystem for Linux) only, for native windows: manual build from source works for deduplication.

## Installation

<b>Using uv</b>

```bash
# Install the package
uv pip install dalla-data-processing
```


<b>Using pip</b>

```bash
# Install the package
pip install dalla-data-processing
```


<b>From Source</b>

```bash
git clone https://github.com/U4RASD/dalla-data-processing.git
cd dalla-data-processing

# Using uv
uv pip install -e .

# Or using pip
pip install -e .
```

## Components

### Deduplication

**CLI Usage**

**Command:** `dalla-dp deduplicate [OPTIONS]`

**Arguments:**
- `-t, --threshold FLOAT` - Similarity threshold (0.0-1.0, default: 0.8)
- `--return-pairs` / `--filter-duplicates` - Return dataset with duplicate info (default) or filtered dataset
- `--keep-vert-files` - Keep vertical format files for inspection
- `--vert-dir PATH` - Directory to store vertical files (useful for different disk)
- `--calculate-scores` - Run phase 2 to calculate similarity scores (slower but more precise)
- `--onion-binary PATH` - Path to onion binary (auto-detected if not specified)

**Examples:**
```bash
# Basic deduplication
dalla-dp -i ./data/raw -o ./data/deduped deduplicate

# With custom threshold
dalla-dp -i ./data/raw -o ./data/deduped deduplicate --threshold 0.9

# Return filtered dataset (removes duplicates)
dalla-dp -i ./data/raw -o ./data/clean deduplicate --filter-duplicates

# Keep intermediate files for inspection
dalla-dp -i ./data/raw -o ./data/deduped deduplicate --keep-vert-files

# Calculate precise similarity scores (slower)
dalla-dp -i ./data/raw -o ./data/deduped deduplicate --calculate-scores

# Use custom onion binary
dalla-dp -i ./data/raw -o ./data/deduped deduplicate --onion-binary /path/to/onion
```

**Python API**

```python
from datasets import load_from_disk
from dalla.deduplication import deduplicate_dataset

# Load dataset
dataset = load_from_disk("./data/raw")

# get duplicate information (adds columns: duplicate_cluster, is_duplicate, duplicate_count)
result = deduplicate_dataset(dataset, column="text", threshold=0.8, return_pairs=True)

# filter to see only duplicates
duplicates = result.filter(lambda x: x['is_duplicate'])

deduped.save_to_disk("./data/clean")
```

### Stemming

Apply morphological analysis and stemming using CAMeL Tools.

**CLI Usage**

**Command:** `dalla-dp stem [OPTIONS]`

**Arguments:**
- `--sep-token TEXT` - Separator token for morphological splits (default: `<+>`)
- `--normalize` - Apply Arabic normalization
- `--keep-diacritics` - Keep diacritics in output
- `--model [mle|bert]` - Disambiguator model (default: mle, faster | bert: more accurate)
- `--use-gpu` - Use GPU for BERT model (only applicable when --model=bert)

**Examples:**
```bash
# Basic stemming with MLE model 
dalla-dp -i ./data/raw -o ./data/stemmed stem

# Use BERT model 
dalla-dp -i ./data/raw -o ./data/stemmed stem --model bert

# Use BERT with GPU acceleration
dalla-dp -i ./data/raw -o ./data/stemmed stem --model bert --use-gpu

# Custom separator token
dalla-dp -i ./data/raw -o ./data/stemmed stem --sep-token "<SEP>"

# Apply normalization
dalla-dp -i ./data/raw -o ./data/stemmed stem --normalize

# Keep diacritics in output
dalla-dp -i ./data/raw -o ./data/stemmed stem --keep-diacritics

```

**Python API**

```python
from datasets import load_from_disk
from dalla.stemming import stem_dataset

# Load dataset
dataset = load_from_disk("./data/raw")

stemmed = stem_dataset(dataset, column="text")

stemmed = stem_dataset(
    dataset,
    column="text",
    model="bert",
    use_gpu=True,
    num_proc=8
)

stemmed = stem_dataset(
    dataset,
    column="content",
    sep_token="<+>",
    normalize=True,
    keep_diacritics=True
)

stemmed.save_to_disk("./data/stemmed")
```

### Quality Checking

Check text quality using morphological analysis to detect errors and foreign words.

**CLI Usage**

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

**Python API**

```python
from datasets import load_from_disk
from dalla.quality import check_quality

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

### Readability Scoring

Calculate readability scores using Flesch Reading Ease and Osman methods.

**CLI Usage**

**Command:** `dalla-dp readability [OPTIONS]`

**Arguments:**
- `--add-ranks` / `--no-ranks` - Add ranking and level columns (default: True)

**Examples:**
```bash
dalla-dp -i ./data/raw -o ./data/scored readability

dalla-dp -i ./data/raw -o ./data/scored readability --no-ranks

dalla-dp -i ./data/raw -o ./data/scored -c content readability
```

**Python API**

```python
from datasets import load_from_disk
from dalla.readability import score_readability

# Load dataset
dataset = load_from_disk("./data/raw")

scored = score_readability(dataset, column="text", add_ranks=True)

# Save result
scored.save_to_disk("./data/scored")
```

**Readability Levels:**
- `0`: Very Easy
- `1`: Easy
- `2`: Medium
- `3`: Difficult
- `4`: Very Difficult

### Dataset Management

Utilities for loading, saving, and inspecting datasets.

**CLI Usage**

**Command:** `dalla-dp info [OPTIONS] DATASET_PATH`

**Arguments:**
- `DATASET_PATH` - Path to the dataset (required, positional argument)
- `--split TEXT` - Specific split to show info for

**Examples:**
```bash
# Show dataset information
dalla-dp info ./data/my_dataset

```

**Python API**

```python
from dalla.core.dataset import DatasetManager

dm = DatasetManager()

dataset = dm.load("./data/my_dataset")
train_data = dm.load("./data/my_dataset", split="train")


info = dm.get_info(dataset)
dm.print_info(dataset)

size = dm.get_size(dataset)

filtered = dm.filter_dataset(
    dataset,
    lambda x: x['quality_score'] > 80.0,
    num_proc=4
)

scores = [0.95, 0.87, 0.92, ...]
dataset = dm.add_column(dataset, "my_score", scores)

subset = dm.select_columns(dataset, ["text", "quality_score"])
cleaned = dm.remove_columns(dataset, ["temp_column"])

splits = dm.train_test_split(dataset, test_size=0.2, seed=42)
```

**Working with DatasetDict**

```python
from datasets import DatasetDict, load_from_disk
from dalla.quality import check_quality

dataset_dict = load_from_disk("./data/my_dataset")

processed_dict = DatasetDict({
    split: check_quality(ds, min_score=60.0)
    for split, ds in dataset_dict.items()
})

train_processed = check_quality(dataset_dict['train'], min_score=60.0)
```

## Building Onion from Source

**Build Instructions**

The onion deduplication tool needs to be compiled for your system:

```bash
cd dalla/deduplication/onion/src_sc

# Compile
make -f Makefile.g

```

Alternatively, use the build script:

```bash
chmod +x scripts/build_onion.sh
./scripts/build_onion.sh
```

## Links

- Homepage: https://github.com/U4RASD/dalla-data-processing
- Issues: https://github.com/U4RASD/dalla-data-processing/issues
- Documentation: https://github.com/U4RASD/dalla-data-processing#readme
