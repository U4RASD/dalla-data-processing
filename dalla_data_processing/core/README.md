# Dataset Management

Utilities for loading, saving, and inspecting datasets.

## CLI Usage

**Command:** `dalla-dp info [OPTIONS] DATASET_PATH`

**Arguments:**
- `DATASET_PATH` - Path to the dataset (required, positional argument)
- `--split TEXT` - Specific split to show info for

**Examples:**
```bash
# Show dataset information
dalla-dp info ./data/my_dataset

```

## Python API

```python
from dalla_data_processing.core.dataset import DatasetManager

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

## Working with DatasetDict

```python
from datasets import DatasetDict, load_from_disk
from dalla_data_processing.quality import check_quality

dataset_dict = load_from_disk("./data/my_dataset")

processed_dict = DatasetDict({
    split: check_quality(ds, min_score=60.0)
    for split, ds in dataset_dict.items()
})

train_processed = check_quality(dataset_dict['train'], min_score=60.0)
```
