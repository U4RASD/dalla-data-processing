# Deduplication

Detect and remove duplicate or near-duplicate documents from your datasets using the Onion algorithm.

## Installation

This feature requires the `[dedup]` extra:

```bash
pip install "dalla-data-processing[dedup]"
# or install all features: pip install "dalla-data-processing[all]"
```

## CLI Usage

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

## Python API

```python
from datasets import load_from_disk
from dalla_data_processing.deduplication import deduplicate_dataset

# Load dataset
dataset = load_from_disk("./data/raw")

# get duplicate information (adds columns: duplicate_cluster, is_duplicate, duplicate_count)
result = deduplicate_dataset(dataset, column="text", threshold=0.8, return_pairs=True)

# filter to see only duplicates
duplicates = result.filter(lambda x: x['is_duplicate'])

deduped.save_to_disk("./data/clean")
```

## Building Onion from Source

The onion deduplication tool needs to be compiled for your system:

```bash
cd dalla_data_processing/deduplication/onion/src_sc

# Compile
make -f Makefile.g

```

Alternatively, use the build script:

```bash
chmod +x scripts/build_onion.sh
./scripts/build_onion.sh
```
