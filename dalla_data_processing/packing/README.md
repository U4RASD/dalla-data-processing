# Dataset Packing

Pack datasets efficiently for LLM training by combining multiple examples into fixed-length sequences. Pre-pack your datasets locally to avoid wasting expensive GPU time on servers.

## Why Use This Tool?

When training large language models, dataset packing is essential for efficient training. This tool allows you to:
- **Save GPU time**: Pack datasets on your local machine before uploading to training servers
- **Preserve data integrity**: Ensures no example is split across multiple packed sequences
- **Handle large datasets**: Process datasets in chunks to manage memory efficiently
- **Flexible tokenization**: Support for both standard text data and chat-formatted (SFT) data

## Installation

This feature requires the `[pack]` extra:

```bash
pip install "dalla-data-processing[pack]"
# or install all features: pip install "dalla-data-processing[all]"
```

## CLI Usage

The packing functionality is integrated into the unified `dalla-dp` CLI:

### Basic Usage

```bash
dalla-dp -i input_dataset -o output_dataset pack --tokenizer-path /path/to/tokenizer
```

### Common Options (from main CLI)

- `-i, --input-dataset`: Path to input HuggingFace dataset (required)
- `-o, --output-dataset`: Path to save output dataset (required)
- `-w, --num-workers`: Number of parallel workers for packing (default: 4)
- `-v, --verbose`: Enable verbose output
- `--overwrite`: Overwrite output if it exists

### Pack-Specific Options

- `--config`: Path to config YAML file (optional)
- `--tokenizer-path`: Path to tokenizer
- `--max-seq-length`: Maximum sequence length (default: 2048)
- `--chunk-size-gb`: Chunk size in GB (default: 2.0)
- `--text-column`: Text column name (default: 'text' or 'messages' for SFT)
- `--subset-order`: Subset processing order
- `--sft`: Enable SFT mode
- `--rbpe`: Use R-BPE tokenizer

### Examples

**Basic packing:**
```bash
dalla-dp -i my_dataset -o packed_dataset pack --tokenizer-path /path/to/tokenizer
```

**Using a config file:**
```bash
dalla-dp -i my_dataset -o packed_dataset pack --config pack_config.yaml
```

**Override config with command line:**
```bash
dalla-dp -i my_dataset -o packed_dataset pack \
  --config pack_config.yaml \
  --max-seq-length 4096
```

**With custom sequence length and workers:**
```bash
dalla-dp -i my_dataset -o packed_dataset -w 8 pack \
  --tokenizer-path /path/to/tokenizer \
  --max-seq-length 4096
```

**SFT mode with subset order:**
```bash
dalla-dp -i my_dataset -o packed_dataset pack \
  --tokenizer-path /path/to/tokenizer \
  --sft \
  --subset-order train --subset-order validation
```

**Using a custom text column:**
```bash
dalla-dp -i my_dataset -o packed_dataset pack \
  --tokenizer-path /path/to/tokenizer \
  --text-column content
```

**With verbose output:**
```bash
dalla-dp -i my_dataset -o packed_dataset -v pack \
  --tokenizer-path /path/to/tokenizer \
  --chunk-size-gb 1.0
```

## Configuration File

```yaml
tokenizer_path: "/path/to/tokenizer"
max_seq_length: 2048
chunk_size_gb: 2.0
sft: false
rbpe: false
text_column: "content"

subset_order:
  - "train"
  - "validation"
```

CLI arguments override config values.

## Python API Usage

You can also use the packing functionality directly in Python:

```python
from dalla_data_processing.packing import DatasetPacker
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("path/to/tokenizer")

packer = DatasetPacker(
    input_dataset="my_dataset",
    output_dataset="packed_dataset",
    tokenizer=tokenizer,
    num_workers=4,
    max_seq_length=2048,
    chunk_size_gb=2.0,
    text_column="content",
)

final_path = packer.process()
```

### API Parameters

- `input_dataset` (str): Path to input dataset
- `output_dataset` (str): Path for output dataset
- `tokenizer`: HuggingFace tokenizer instance
- `subset_order` (list[str], optional): Order to process subsets
- `num_workers` (int): Number of parallel packing processes (default: 4)
- `chunk_size_gb` (float): Size of processing chunks in GB (default: 2.0)
- `max_seq_length` (int): Maximum sequence length (default: 2048)
- `sft` (bool): Enable SFT mode (default: False)
- `rbpe` (bool): Use R-BPE tokenizer (default: False)
- `text_column` (str, optional): Text column name

## The Packing Method

- Packs multiple examples into a single sequence up to `max_seq_length`
- Adds EOS token between examples
- Pads remaining space with PAD tokens
- **Guarantees no example is cut in the middle** - if an example doesn't fit, it starts in the next sequence
- Best for preserving data integrity

## SFT Mode

Enable SFT mode when working with chat-formatted data:

```bash
dalla-dp -i my_dataset -o packed_dataset pack \
  --tokenizer-path /path/to/tokenizer \
  --sft
```

**When to use SFT mode:**
- Your dataset has a `messages` field with chat conversations
- Your tokenizer has a chat template defined
- You're doing supervised fine-tuning (SFT) on conversational data

**When NOT to use SFT mode:**
- Continued pre-training (CPT) on plain text
- Your tokenizer doesn't have a chat template
- Your dataset only has a `text` field

**Input format for SFT mode:**
```python
{
    "messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
}
```

## Custom Text Column

Defaults: `text` (non-SFT) or `messages` (SFT). Override with `--text-column`.

## Dataset Format

### Input
Your dataset should be in Hugging Face datasets format:
```
my_dataset/
  train/
    data-00000-of-00001.arrow
    dataset_info.json
    state.json
  validation/
    data-00000-of-00001.arrow
    dataset_info.json
    state.json
  dataset_dict.json
```

### Output
The packed dataset will be saved to:
```
packed_dataset/
  final_dataset/
    train/
      data-00000-of-00001.arrow
      dataset_info.json
      state.json
```

Each example in the final dataset will have:
- `input_ids`: Token IDs (length = `max_seq_length`)
- `labels`: Same as `input_ids` (or masked for SFT)

## Memory Considerations

The `--chunk-size-gb` parameter controls memory usage:
- **Smaller values** (0.5-1 GB): Lower memory, more chunks
- **Larger values** (2-4 GB): Higher memory, fewer chunks

## How It Works

1. **Analyze**: Calculate sizes of dataset subsets
2. **Split**: Divide datasets into manageable chunks
3. **Tokenize**: Convert text to token IDs using parallel processing
4. **Pack**: Combine multiple examples into fixed-length sequences
5. **Concatenate**: Merge all packed chunks into final dataset

The tool automatically:
- Preserves subset ordering
- Removes intermediate files to save disk space
- Handles empty subsets gracefully
- Skips examples longer than `max_seq_length`
