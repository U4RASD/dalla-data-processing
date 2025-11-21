# Stemming

Apply morphological analysis and stemming using CAMeL Tools.

## Installation

This feature requires the `[stem]` extra:

```bash
pip install "dalla-data-processing[stem]"
# or install all features: pip install "dalla-data-processing[all]"
```

## CLI Usage

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

## Python API

### Dataset Processing

```python
from datasets import load_from_disk
from dalla_data_processing.stemming import stem_dataset

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

### Direct Text Processing

```python
from dalla_data_processing.stemming import stem

text = "الكتاب الجميل"
result = stem(text)

texts = ["المدرسة الجديدة", "الطالب المجتهد"]
results = stem(texts)

```
