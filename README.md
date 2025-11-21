# Dalla Data Processing (dalla-dp)

A comprehensive Arabic data processing pipeline with deduplication, stemming, quality checking, and readability scoring, used for the DALLA Models.

## Compatibility

- **Linux**: Fully supported
- **macOS**: Fully supported (Intel or through rosetta)
- **Windows**: Supported through WSL (Windows Subsystem for Linux) only, for native windows: manual build from source works for deduplication.

## Installation

### Quick Start (All Features)

For most users, install with all features enabled:

<b>Using uv</b>

```bash
uv pip install "dalla-data-processing[all]"
```

<b>Using pip</b>

```bash
pip install "dalla-data-processing[all]"
```

### Modular Installation (Advanced)

Install only the components you need to keep dependencies minimal:

```bash
# Base installation (no processing features, only core dependencies)
pip install dalla-data-processing

# Install specific features
pip install "dalla-data-processing[dedup]"        # Deduplication only
pip install "dalla-data-processing[stem]"         # Stemming only
pip install "dalla-data-processing[quality]"      # Quality checking only
pip install "dalla-data-processing[readability]"  # Readability scoring only
pip install "dalla-data-processing[pack]"         # Dataset packing only

# Combine multiple features
pip install "dalla-data-processing[dedup,stem,quality]"
```

### Development Installation

<b>From Source (with uv - recommended)</b>

```bash
git clone https://github.com/U4RASD/dalla-data-processing.git
cd dalla-data-processing

# Install all features and dev dependencies
uv sync --all-extras

# Or install with specific extras only
uv sync --extra dedup --extra stem
```

<b>From Source (with pip)</b>

```bash
git clone https://github.com/U4RASD/dalla-data-processing.git
cd dalla-data-processing

# Install with all features for development
pip install -e ".[all,dev]"
```

## Components

> **Note:** Each component requires its corresponding extra to be installed. Install with `[all]` to enable all features, or see [Modular Installation](#modular-installation-advanced) to install only what you need.

### 1. [Deduplication](dalla_data_processing/deduplication/README.md)
Detect and remove duplicate or near-duplicate documents from your datasets using the Onion algorithm.
- **Requires:** `[dedup]` extra

### 2. [Stemming](dalla_data_processing/stemming/README.md)
Apply morphological analysis and stemming using CAMeL Tools.
- **Requires:** `[stem]` extra

### 3. [Quality Checking](dalla_data_processing/quality/README.md)
Check text quality using morphological analysis to detect errors and foreign words.
- **Requires:** `[quality]` extra

### 4. [Readability Scoring](dalla_data_processing/readability/README.md)
Calculate readability scores using Flesch Reading Ease and Osman methods.
Contains also ranking according to both scores
- **Requires:** `[readability]` extra

### 5. [Dataset Packing](dalla_data_processing/packing/README.md)
Pack and prepare datasets for training.
- **Requires:** `[pack]` extra

## Links

- Homepage: https://github.com/U4RASD/dalla-data-processing
- Issues: https://github.com/U4RASD/dalla-data-processing/issues
- Documentation: https://github.com/U4RASD/dalla-data-processing#readme
