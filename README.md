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

### 1. [Deduplication](dalla_data_processing/deduplication/README.md)
Detect and remove duplicate or near-duplicate documents from your datasets using the Onion algorithm.

### 2. [Stemming](dalla_data_processing/stemming/README.md)
Apply morphological analysis and stemming using CAMeL Tools.

### 3. [Quality Checking](dalla_data_processing/quality/README.md)
Check text quality using morphological analysis to detect errors and foreign words.

### 4. [Readability Scoring](dalla_data_processing/readability/README.md)
Calculate readability scores using Flesch Reading Ease and Osman methods.
Contains also ranking according to both scores

## Links

- Homepage: https://github.com/U4RASD/dalla-data-processing
- Issues: https://github.com/U4RASD/dalla-data-processing/issues
- Documentation: https://github.com/U4RASD/dalla-data-processing#readme
