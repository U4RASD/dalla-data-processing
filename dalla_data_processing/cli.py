"""
Main CLI entry point for dalla-data-processing (dalla-dp).

This module provides the unified command-line interface for all
Arabic data processing operations.
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click

from dalla_data_processing import __version__
from dalla_data_processing.utils.logger import get_logger

if TYPE_CHECKING:
    from datasets import Dataset


logger = get_logger("dalla.cli")


class Context:
    """Shared context for CLI commands."""

    def __init__(self):
        self.input_dataset: Path | None = None
        self.output_dataset: Path | None = None
        self.column: str = "text"
        self.num_workers: int | None = None
        self.verbose: bool = False
        self.overwrite: bool = False
        self.dataset: Dataset | None = None
        self._dataset_manager = None

    @property
    def dataset_manager(self):
        """Lazy load DatasetManager."""
        if self._dataset_manager is None:
            from dalla_data_processing.core.dataset import DatasetManager

            self._dataset_manager = DatasetManager()
        return self._dataset_manager


pass_context = click.make_pass_decorator(Context, ensure=True)


def common_options(func):
    """Decorator to add common dataset processing options to commands.

    This allows options to be specified either before or after the subcommand,
    and ensures they appear in each subcommand's help text.
    """
    decorators = [
        click.option(
            "--input-dataset",
            "-i",
            type=click.Path(exists=True, path_type=Path),
            help="Path to input HuggingFace dataset",
        ),
        click.option(
            "--output-dataset",
            "-o",
            type=click.Path(path_type=Path),
            help="Path to save output HuggingFace dataset",
        ),
        click.option(
            "--column",
            "-c",
            default="text",
            help="Column name to process (default: 'text')",
        ),
        click.option(
            "--num-workers",
            "-w",
            type=int,
            help="Number of parallel workers (default: auto)",
        ),
        click.option(
            "--verbose",
            "-v",
            is_flag=True,
            help="Enable verbose output",
        ),
        click.option(
            "--quiet",
            "-q",
            is_flag=True,
            help="Suppress non-error output",
        ),
        click.option(
            "--overwrite",
            is_flag=True,
            help="Overwrite output dataset if it already exists",
        ),
    ]
    for decorator in reversed(decorators):
        func = decorator(func)
    return func


def _setup_context_and_logging(
    ctx: Context,
    input_dataset: Path | None,
    output_dataset: Path | None,
    column: str,
    num_workers: int | None,
    verbose: bool,
    quiet: bool,
    overwrite: bool,
):
    """Helper to populate context from command-level parameters and setup logging."""
    ctx.input_dataset = input_dataset or ctx.input_dataset
    ctx.output_dataset = output_dataset or ctx.output_dataset
    ctx.column = column if column != "text" else ctx.column or column
    ctx.num_workers = num_workers or ctx.num_workers
    ctx.verbose = verbose or ctx.verbose
    ctx.overwrite = overwrite or ctx.overwrite

    from dalla_data_processing.utils import setup_logging

    if quiet:
        setup_logging(log_format="console", log_level="ERROR")
    elif ctx.verbose:
        setup_logging(log_format="console", log_level="DEBUG")


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="dalla-data-processing")
@click.option(
    "--input-dataset",
    "-i",
    type=click.Path(exists=True, path_type=Path),
    help="Path to input HuggingFace dataset",
)
@click.option(
    "--output-dataset",
    "-o",
    type=click.Path(path_type=Path),
    help="Path to save output HuggingFace dataset",
)
@click.option(
    "--column",
    "-c",
    default="text",
    help="Column name to process (default: 'text')",
)
@click.option(
    "--num-workers",
    "-w",
    type=int,
    help="Number of parallel workers (default: auto)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress non-error output",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite output dataset if it already exists",
)
@pass_context
def cli(
    ctx: Context,
    input_dataset: Path | None,
    output_dataset: Path | None,
    column: str,
    num_workers: int | None,
    verbose: bool,
    quiet: bool,
    overwrite: bool,
):
    """
    Dalla Data Processing - Unified Arabic Data Processing Pipeline

    - Deduplication using onion algorithm
    - Stemming and morphological analysis
    - Quality checking
    - Readability scoring
    - Packing Dataset for training

    """
    ctx.input_dataset = input_dataset
    ctx.output_dataset = output_dataset
    ctx.column = column
    ctx.num_workers = num_workers
    ctx.verbose = verbose
    ctx.overwrite = overwrite

    if quiet or verbose:
        from dalla_data_processing.utils import setup_logging

        if quiet:
            setup_logging(log_format="console", log_level="ERROR")
        elif verbose:
            setup_logging(log_format="console", log_level="DEBUG")


@cli.command(context_settings={"help_option_names": ["-h", "--help"]})
@common_options
@click.option(
    "--threshold",
    "-t",
    type=float,
    default=0.8,
    help="Similarity threshold (0.0-1.0, default: 0.8)",
)
@click.option(
    "--return-pairs/--filter-duplicates",
    default=False,
    help="Return dataset with duplicate info (True) or filtered dataset (False)",
)
@click.option(
    "--keep-vert-files",
    is_flag=True,
    help="Keep vertical format files for inspection",
)
@click.option(
    "--vert-dir",
    type=click.Path(),
    help="Directory to store vertical files (useful for different disk)",
)
@click.option(
    "--calculate-scores",
    is_flag=True,
    help="Run phase 2 to calculate similarity scores (slower but more precise)",
)
@click.option(
    "--onion-binary",
    type=click.Path(exists=True),
    help="Path to onion binary (auto-detected if not specified)",
)
@pass_context
def deduplicate(
    ctx: Context,
    input_dataset: Path | None,
    output_dataset: Path | None,
    column: str,
    num_workers: int | None,
    verbose: bool,
    quiet: bool,
    overwrite: bool,
    threshold: float,
    return_pairs: bool,
    keep_vert_files: bool,
    vert_dir: str | None,
    calculate_scores: bool,
    onion_binary: str | None,
):
    """Remove duplicate entries using onion algorithm."""
    _setup_context_and_logging(
        ctx, input_dataset, output_dataset, column, num_workers, verbose, quiet, overwrite
    )
    _require_io_paths(ctx)

    logger.info(f"Loading dataset from {ctx.input_dataset}")
    dataset = ctx.dataset_manager.load(ctx.input_dataset)
    dataset = _handle_dataset_dict(dataset)

    mode = "pairs" if return_pairs else "filter"
    logger.info(f"Deduplicating with threshold={threshold}, mode={mode}")
    if calculate_scores:
        logger.info("  Phase 2: ON (calculating similarity scores)")
    else:
        logger.info("  Phase 2: OFF (faster, sufficient for most use cases)")

    try:
        from dalla_data_processing.deduplication import deduplicate_dataset
    except ImportError:
        logger.error("Missing dependencies for deduplication")
        logger.error("Install with: pip install 'dalla-data-processing[dedup]'")
        sys.exit(1)

    deduplicated = deduplicate_dataset(
        dataset,
        column=ctx.column,
        threshold=threshold,
        return_pairs=return_pairs,
        keep_vert_files=keep_vert_files,
        vert_dir=Path(vert_dir) if vert_dir else None,
        calculate_scores=calculate_scores,
        onion_binary=Path(onion_binary) if onion_binary else None,
    )

    logger.info(f"Saving deduplicated dataset to {ctx.output_dataset}")
    ctx.dataset_manager.save(deduplicated, ctx.output_dataset, overwrite=ctx.overwrite)

    from dalla_data_processing.core.dataset import DatasetManager

    original_size = DatasetManager.get_size(dataset)
    final_size = DatasetManager.get_size(deduplicated)

    logger.info("✓ Deduplication complete")
    logger.info(f"  Original: {original_size:,} examples")

    if return_pairs:
        num_dups = sum(1 for ex in deduplicated if ex.get("is_duplicate", False))
        logger.info(
            f"  Documents with duplicates: {num_dups:,} ({num_dups / original_size * 100:.1f}%)"
        )
        logger.info("  Added columns: duplicate_cluster, is_duplicate, duplicate_count")
    else:
        removed = original_size - final_size
        logger.info(f"  Removed: {removed:,} duplicates ({removed / original_size * 100:.1f}%)")
        logger.info(f"  Final: {final_size:,} examples")


@cli.command(context_settings={"help_option_names": ["-h", "--help"]})
@common_options
@click.option(
    "--sep-token",
    default="<+>",
    help="Separator token for morphological splits (default: '<+>')",
)
@click.option(
    "--normalize",
    is_flag=True,
    help="Apply Arabic normalization",
)
@click.option(
    "--keep-diacritics",
    is_flag=True,
    help="Keep diacritics in output",
)
@click.option(
    "--model",
    type=click.Choice(["mle", "bert"], case_sensitive=False),
    default="mle",
    help="Disambiguator model (default: mle, faster | bert: more accurate)",
)
@click.option(
    "--use-gpu",
    is_flag=True,
    help="Use GPU for BERT model (only applicable when --model=bert)",
)
@pass_context
def stem(
    ctx: Context,
    input_dataset: Path | None,
    output_dataset: Path | None,
    column: str,
    num_workers: int | None,
    verbose: bool,
    quiet: bool,
    overwrite: bool,
    sep_token: str,
    normalize: bool,
    keep_diacritics: bool,
    model: str,
    use_gpu: bool,
):
    """Apply stemming and morphological analysis."""
    _setup_context_and_logging(
        ctx, input_dataset, output_dataset, column, num_workers, verbose, quiet, overwrite
    )
    _require_io_paths(ctx)

    logger.info(f"Loading dataset from {ctx.input_dataset}")
    dataset = ctx.dataset_manager.load(ctx.input_dataset)
    dataset = _handle_dataset_dict(dataset)

    logger.info(f"Stemming {ctx.column} column (workers={ctx.num_workers or 'auto'})")
    logger.info(f"Model: {model.upper()}{' (GPU enabled)' if model == 'bert' and use_gpu else ''}")

    try:
        from dalla_data_processing.stemming import stem_dataset
    except ImportError:
        logger.error("Missing dependencies for stemming")
        logger.error("Install with: pip install 'dalla-data-processing[stem]'")
        sys.exit(1)

    stemmed = stem_dataset(
        dataset,
        column=ctx.column,
        sep_token=sep_token,
        normalize=normalize,
        keep_diacritics=keep_diacritics,
        num_proc=ctx.num_workers,
        model=model,
        use_gpu=use_gpu,
    )

    logger.info(f"Saving stemmed dataset to {ctx.output_dataset}")
    ctx.dataset_manager.save(stemmed, ctx.output_dataset, overwrite=ctx.overwrite)

    logger.info("✓ Stemming complete")


@cli.command(context_settings={"help_option_names": ["-h", "--help"]})
@common_options
@click.option(
    "--min-score",
    type=float,
    default=0.0,
    help="Minimum quality score to keep (0-100, default: 0)",
)
@click.option(
    "--save-errors",
    is_flag=True,
    help="Save erroneous words to file",
)
@click.option(
    "--model",
    type=click.Choice(["mle", "bert"], case_sensitive=False),
    default="mle",
    help="Disambiguator model (default: mle, faster | bert: more accurate)",
)
@click.option(
    "--use-gpu",
    is_flag=True,
    help="Use GPU for BERT model (only applicable when --model=bert)",
)
@pass_context
def quality_check(
    ctx: Context,
    input_dataset: Path | None,
    output_dataset: Path | None,
    column: str,
    num_workers: int | None,
    verbose: bool,
    quiet: bool,
    overwrite: bool,
    min_score: float,
    save_errors: bool,
    model: str,
    use_gpu: bool,
):
    """Check text quality and calculate scores."""
    _setup_context_and_logging(
        ctx, input_dataset, output_dataset, column, num_workers, verbose, quiet, overwrite
    )
    _require_io_paths(ctx)

    logger.info(f"Loading dataset from {ctx.input_dataset}")
    dataset = ctx.dataset_manager.load(ctx.input_dataset)
    dataset = _handle_dataset_dict(dataset)

    logger.info(f"Checking quality of {ctx.column} column")
    logger.info(f"Model: {model.upper()}{' (GPU enabled)' if model == 'bert' and use_gpu else ''}")

    try:
        from dalla_data_processing.quality import check_quality
    except ImportError:
        logger.error("Missing dependencies for quality checking")
        logger.error("Install with: pip install 'dalla-data-processing[quality]'")
        sys.exit(1)

    scored = check_quality(
        dataset,
        column=ctx.column,
        min_score=min_score,
        save_errors=save_errors,
        num_workers=ctx.num_workers,
        model=model,
        use_gpu=use_gpu,
    )

    logger.info(f"Saving quality-checked dataset to {ctx.output_dataset}")
    ctx.dataset_manager.save(scored, ctx.output_dataset, overwrite=ctx.overwrite)

    from dalla_data_processing.core.dataset import DatasetManager

    original_size = DatasetManager.get_size(dataset)
    final_size = DatasetManager.get_size(scored)

    logger.info("✓ Quality check complete")
    if min_score > 0:
        removed = original_size - final_size
        logger.info(
            f"  Filtered {removed:,} low-quality examples ({removed / original_size * 100:.1f}%)"
        )


@cli.command(context_settings={"help_option_names": ["-h", "--help"]})
@common_options
@click.option(
    "--add-ranks/--no-ranks",
    default=True,
    help="Add ranking and level columns (default: True)",
)
@pass_context
def readability(
    ctx: Context,
    input_dataset: Path | None,
    output_dataset: Path | None,
    column: str,
    num_workers: int | None,
    verbose: bool,
    quiet: bool,
    overwrite: bool,
    add_ranks: bool,
):
    """Calculate readability scores using Flesch and Osman methods."""
    _setup_context_and_logging(
        ctx, input_dataset, output_dataset, column, num_workers, verbose, quiet, overwrite
    )
    _require_io_paths(ctx)

    logger.info(f"Loading dataset from {ctx.input_dataset}")
    dataset = ctx.dataset_manager.load(ctx.input_dataset)
    dataset = _handle_dataset_dict(dataset)

    logger.info(f"Calculating readability scores for {ctx.column} column")
    if add_ranks:
        logger.info("  Including ranking and difficulty levels (0-4)")

    try:
        from dalla_data_processing.readability import score_readability
    except ImportError:
        logger.error("Missing dependencies for readability scoring")
        logger.error("Install with: pip install 'dalla-data-processing[readability]'")
        sys.exit(1)

    scored = score_readability(
        dataset,
        column=ctx.column,
        add_ranks=add_ranks,
        num_proc=ctx.num_workers,
    )

    logger.info(f"Saving scored dataset to {ctx.output_dataset}")
    ctx.dataset_manager.save(scored, ctx.output_dataset, overwrite=ctx.overwrite)

    logger.info("✓ Readability scoring complete")

    if add_ranks:
        logger.info("  Added columns: flesch_score, osman_score, flesch_rank, osman_rank,")
        logger.info("                 readability_level")
    else:
        logger.info("  Added columns: flesch_score, osman_score")


@cli.command(context_settings={"help_option_names": ["-h", "--help"]})
@common_options
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to config YAML file (optional)",
)
@click.option(
    "--tokenizer-path",
    type=str,
    help="Path to tokenizer",
)
@click.option(
    "--max-seq-length",
    type=int,
    help="Maximum sequence length for packing",
)
@click.option(
    "--chunk-size-gb",
    type=float,
    help="Size of each processing chunk in GB",
)
@click.option(
    "--subset-order",
    multiple=True,
    help="Subset processing order (can be specified multiple times)",
)
@click.option(
    "--sft",
    is_flag=True,
    help="Enable SFT mode (uses tokenizer's chat template)",
)
@click.option(
    "--rbpe",
    is_flag=True,
    help="Use R-BPE tokenizer",
)
@click.option(
    "--text-column",
    type=str,
    help="Column name containing text data (defaults: 'text' for non-SFT, 'messages' for SFT)",
)
@pass_context
def pack(
    ctx: Context,
    input_dataset: Path | None,
    output_dataset: Path | None,
    column: str,
    num_workers: int | None,
    verbose: bool,
    quiet: bool,
    overwrite: bool,
    config: Path | None,
    tokenizer_path: str | None,
    max_seq_length: int | None,
    chunk_size_gb: float | None,
    subset_order: tuple[str, ...],
    sft: bool,
    rbpe: bool,
    text_column: str | None,
):
    """Pack datasets for efficient LLM training.

    Combines multiple examples into fixed-length sequences, optimizing for
    efficient training. Preserves data integrity by ensuring no example is
    split across multiple sequences.
    """
    _setup_context_and_logging(
        ctx, input_dataset, output_dataset, column, num_workers, verbose, quiet, overwrite
    )
    _require_io_paths(ctx)

    try:
        import yaml
    except ImportError:
        logger.error("Missing dependencies for packing")
        logger.error("Install with: pip install 'dalla-data-processing[pack]'")
        sys.exit(1)

    config_data = {}
    if config:
        with open(config) as f:
            config_data = yaml.safe_load(f) or {}

    if tokenizer_path:
        config_data["tokenizer_path"] = tokenizer_path
    if rbpe:
        config_data["rbpe"] = True
    if sft:
        config_data["sft"] = True
    if max_seq_length is not None:
        config_data["max_seq_length"] = max_seq_length
    if chunk_size_gb is not None:
        config_data["chunk_size_gb"] = chunk_size_gb
    if subset_order:
        config_data["subset_order"] = list(subset_order)
    if text_column:
        config_data["text_column"] = text_column

    if "tokenizer_path" not in config_data:
        logger.error("Error: --tokenizer-path is required")
        sys.exit(1)

    if config_data.get("rbpe"):
        try:
            from rbpe import RBPETokenizer

            tokenizer = RBPETokenizer.from_pretrained(config_data["tokenizer_path"])
        except ImportError:
            logger.error("Missing rbpe package")
            logger.error(
                "rbpe is not included in the default installation due to "
                "dependency conflicts with camel-tools (transformers version requirements)"
            )
            logger.error("Install separately with: pip install rbpe")
            logger.error(
                "Note: Installing rbpe may require a separate environment "
                "if you also use dedup/stem/quality features"
            )
            sys.exit(1)
    else:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(config_data["tokenizer_path"])
        except ImportError:
            logger.error("Missing transformers package")
            logger.error("Install with: pip install transformers")
            sys.exit(1)

    if "text_column" not in config_data:
        config_data["text_column"] = "messages" if config_data.get("sft", False) else "text"

    logger.info("Starting dataset processing")
    logger.info(f"Input path: {ctx.input_dataset}")
    logger.info(f"Output directory: {ctx.output_dataset}")
    logger.info(f"Using RBPE tokenizer: {config_data.get('rbpe', False)}")
    logger.info(f"Chunk size: {config_data.get('chunk_size_gb', 2)}GB")
    logger.info(f"Max sequence length: {config_data.get('max_seq_length', 2048)}")

    from dalla_data_processing.packing import DatasetPacker

    processor = DatasetPacker(
        input_dataset=str(ctx.input_dataset),
        output_dataset=str(ctx.output_dataset),
        tokenizer=tokenizer,
        subset_order=config_data.get("subset_order"),
        num_workers=ctx.num_workers or 4,
        chunk_size_gb=config_data.get("chunk_size_gb", 2.0),
        max_seq_length=config_data.get("max_seq_length", 2048),
        sft=config_data.get("sft", False),
        rbpe=config_data.get("rbpe", False),
        text_column=config_data.get("text_column"),
    )

    final_path = processor.process()
    logger.info("Processing completed successfully!")
    logger.info(f"Final dataset saved to: {final_path}")


@cli.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--split",
    help="Specific split to show info for",
)
@click.argument(
    "dataset_path",
    type=click.Path(exists=True, path_type=Path),
)
def info(dataset_path: Path, split: str | None):
    """Display information about a dataset."""
    from dalla_data_processing.core.dataset import DatasetManager

    dm = DatasetManager()

    try:
        dataset = dm.load(dataset_path, split=split)
        dm.print_info(dataset)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        sys.exit(1)


def _handle_dataset_dict(dataset, split_preference: str = "train"):
    """Handle DatasetDict by selecting appropriate split."""
    from datasets import DatasetDict

    if isinstance(dataset, DatasetDict):
        splits = list(dataset.keys())
        logger.info(f"Dataset has multiple splits: {', '.join(splits)}")

        if split_preference in dataset:
            logger.info(
                f"Using '{split_preference}' split ({len(dataset[split_preference])} examples)"
            )
            return dataset[split_preference]
        else:
            first_split = splits[0]
            logger.info(f"Using '{first_split}' split ({len(dataset[first_split])} examples)")
            return dataset[first_split]
    else:
        return dataset


def _require_io_paths(ctx: Context):
    """Ensure input and output paths are provided."""
    if ctx.input_dataset is None:
        logger.error("Error: --input-dataset is required")
        logger.info("Use --help for usage information")
        sys.exit(1)

    if ctx.output_dataset is None:
        logger.error("Error: --output-dataset is required")
        logger.info("Use --help for usage information")
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    try:
        cli(obj=Context())
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error: {e}")
        if "--verbose" in sys.argv or "-v" in sys.argv:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
