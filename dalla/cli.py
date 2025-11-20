"""
Main CLI entry point for dalla-process.

This module provides the unified command-line interface for all
Arabic data processing operations.
"""

import sys
from pathlib import Path

import click
from datasets import Dataset, DatasetDict

from dalla import __version__
from dalla.core.dataset import DatasetManager
from dalla.utils import get_logger, setup_logging

setup_logging(log_format="console", log_level="INFO")
logger = get_logger(__name__)


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
        self.dataset_manager = DatasetManager()


pass_context = click.make_pass_decorator(Context, ensure=True)


@click.group(
    context_settings={"help_option_names": ["-h", "--help"], "allow_interspersed_args": True}
)
@click.version_option(version=__version__, prog_name="dalla-process")
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

    A comprehensive toolkit for processing Arabic text data with support for:
    - Deduplication using onion algorithm
    - Stemming and morphological analysis
    - Quality checking
    - Readability scoring

    Examples:

        # Deduplicate a dataset
        dalla-dp -i ./data/raw -o ./data/deduped deduplicate

        # Stem text with 8 workers
        dalla-dp -i ./data/raw -o ./data/stemmed -w 8 stem

        # Check quality with custom column
        dalla-dp -i ./data/raw -o ./data/quality -c content quality-check
    """
    ctx.input_dataset = input_dataset
    ctx.output_dataset = output_dataset
    ctx.column = column
    ctx.num_workers = num_workers
    ctx.verbose = verbose
    ctx.overwrite = overwrite

    if quiet:
        setup_logging(log_format="console", log_level="ERROR")
    elif verbose:
        setup_logging(log_format="console", log_level="DEBUG")


@cli.command(context_settings={"help_option_names": ["-h", "--help"]})
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
    threshold: float,
    return_pairs: bool,
    keep_vert_files: bool,
    vert_dir: str | None,
    calculate_scores: bool,
    onion_binary: str | None,
):
    """Remove duplicate entries using onion algorithm."""
    _require_io_paths(ctx)

    click.echo(f"Loading dataset from {ctx.input_dataset}")
    dataset = ctx.dataset_manager.load(ctx.input_dataset)
    dataset = _handle_dataset_dict(dataset)

    mode = "pairs" if return_pairs else "filter"
    click.echo(f"Deduplicating with threshold={threshold}, mode={mode}")
    if calculate_scores:
        click.echo("  Phase 2: ON (calculating similarity scores)")
    else:
        click.echo("  Phase 2: OFF (faster, sufficient for most use cases)")

    from dalla.deduplication import deduplicate_dataset

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

    click.echo(f"Saving deduplicated dataset to {ctx.output_dataset}")
    ctx.dataset_manager.save(deduplicated, ctx.output_dataset, overwrite=ctx.overwrite)

    original_size = DatasetManager.get_size(dataset)
    final_size = DatasetManager.get_size(deduplicated)

    click.echo(click.style("✓ Deduplication complete", fg="green"))
    click.echo(f"  Original: {original_size:,} examples")

    if return_pairs:
        num_dups = sum(1 for ex in deduplicated if ex.get("is_duplicate", False))
        click.echo(
            f"  Documents with duplicates: {num_dups:,} ({num_dups / original_size * 100:.1f}%)"
        )
        click.echo("  Added columns: duplicate_cluster, is_duplicate, duplicate_count")
    else:
        removed = original_size - final_size
        click.echo(f"  Removed: {removed:,} duplicates ({removed / original_size * 100:.1f}%)")
        click.echo(f"  Final: {final_size:,} examples")


@cli.command(context_settings={"help_option_names": ["-h", "--help"]})
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
    ctx: Context, sep_token: str, normalize: bool, keep_diacritics: bool, model: str, use_gpu: bool
):
    """Apply stemming and morphological analysis."""
    _require_io_paths(ctx)

    click.echo(f"Loading dataset from {ctx.input_dataset}")
    dataset = ctx.dataset_manager.load(ctx.input_dataset)
    dataset = _handle_dataset_dict(dataset)

    click.echo(f"Stemming {ctx.column} column (workers={ctx.num_workers or 'auto'})")
    click.echo(f"Model: {model.upper()}{' (GPU enabled)' if model == 'bert' and use_gpu else ''}")

    from dalla.stemming import stem_dataset

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

    click.echo(f"Saving stemmed dataset to {ctx.output_dataset}")
    ctx.dataset_manager.save(stemmed, ctx.output_dataset, overwrite=ctx.overwrite)

    click.echo(click.style("✓ Stemming complete", fg="green"))


@cli.command(context_settings={"help_option_names": ["-h", "--help"]})
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
def quality_check(ctx: Context, min_score: float, save_errors: bool, model: str, use_gpu: bool):
    """Check text quality and calculate scores."""
    _require_io_paths(ctx)

    click.echo(f"Loading dataset from {ctx.input_dataset}")
    dataset = ctx.dataset_manager.load(ctx.input_dataset)
    dataset = _handle_dataset_dict(dataset)

    click.echo(f"Checking quality of {ctx.column} column")
    click.echo(f"Model: {model.upper()}{' (GPU enabled)' if model == 'bert' and use_gpu else ''}")

    from dalla.quality import check_quality

    scored = check_quality(
        dataset,
        column=ctx.column,
        min_score=min_score,
        save_errors=save_errors,
        num_workers=ctx.num_workers,
        model=model,
        use_gpu=use_gpu,
    )

    click.echo(f"Saving quality-checked dataset to {ctx.output_dataset}")
    ctx.dataset_manager.save(scored, ctx.output_dataset, overwrite=ctx.overwrite)

    original_size = DatasetManager.get_size(dataset)
    final_size = DatasetManager.get_size(scored)

    click.echo(click.style("✓ Quality check complete", fg="green"))
    if min_score > 0:
        removed = original_size - final_size
        click.echo(
            f"  Filtered {removed:,} low-quality examples ({removed / original_size * 100:.1f}%)"
        )


@cli.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--add-ranks/--no-ranks",
    default=True,
    help="Add ranking and level columns (default: True)",
)
@pass_context
def readability(ctx: Context, add_ranks: bool):
    """Calculate readability scores using Flesch and Osman methods."""
    _require_io_paths(ctx)

    click.echo(f"Loading dataset from {ctx.input_dataset}")
    dataset = ctx.dataset_manager.load(ctx.input_dataset)
    dataset = _handle_dataset_dict(dataset)

    click.echo(f"Calculating readability scores for {ctx.column} column")
    if add_ranks:
        click.echo("  Including ranking and difficulty levels (0-4)")

    from dalla.readability import score_readability

    scored = score_readability(
        dataset,
        column=ctx.column,
        add_ranks=add_ranks,
        num_proc=ctx.num_workers,
    )

    click.echo(f"Saving scored dataset to {ctx.output_dataset}")
    ctx.dataset_manager.save(scored, ctx.output_dataset, overwrite=ctx.overwrite)

    click.echo(click.style("✓ Readability scoring complete", fg="green"))

    if add_ranks:
        click.echo("  Added columns: flesch_score, osman_score, flesch_rank, osman_rank,")
        click.echo("                 readability_level")
    else:
        click.echo("  Added columns: flesch_score, osman_score")


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
    dm = DatasetManager()

    try:
        dataset = dm.load(dataset_path, split=split)
        dm.print_info(dataset)
    except Exception as e:
        click.echo(click.style(f"Error loading dataset: {e}", fg="red"), err=True)
        sys.exit(1)


def _handle_dataset_dict(dataset, split_preference: str = "train"):
    """Handle DatasetDict by selecting appropriate split."""

    if isinstance(dataset, DatasetDict):
        splits = list(dataset.keys())
        click.echo(f"Dataset has multiple splits: {', '.join(splits)}")

        if split_preference in dataset:
            click.echo(
                f"Using '{split_preference}' split ({len(dataset[split_preference])} examples)"
            )
            return dataset[split_preference]
        else:
            first_split = splits[0]
            click.echo(f"Using '{first_split}' split ({len(dataset[first_split])} examples)")
            return dataset[first_split]
    else:
        return dataset


def _require_io_paths(ctx: Context):
    """Ensure input and output paths are provided."""
    if ctx.input_dataset is None:
        click.echo(
            click.style("Error: --input-dataset is required", fg="red"),
            err=True,
        )
        click.echo("Use --help for usage information")
        sys.exit(1)

    if ctx.output_dataset is None:
        click.echo(
            click.style("Error: --output-dataset is required", fg="red"),
            err=True,
        )
        click.echo("Use --help for usage information")
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    try:
        cli(obj=Context())
    except KeyboardInterrupt:
        click.echo("\n" + click.style("Interrupted by user", fg="yellow"))
        sys.exit(130)
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        if "--verbose" in sys.argv or "-v" in sys.argv:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
