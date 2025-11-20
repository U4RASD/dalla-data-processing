"""
Wrapper for the onion C binary deduplication tool.
Handles execution of the onion binary and management of temporary files.
"""

import shutil
import subprocess
from pathlib import Path

from dalla_data_processing.utils.logger import get_logger

logger = get_logger(__name__)


def find_onion_binary() -> Path | None:
    """
    Find the onion binary in the system.

    Searches in:
    1. System PATH
    2. Bundled platform-specific binary in package
    3. Generic bundled binary (symlink)
    4. Local build directories

    Returns:
        Path to onion binary or None if not found
    """
    import platform

    onion_path = shutil.which("onion")
    if onion_path:
        logger.info(f"Found onion in system PATH: {onion_path}")
        return Path(onion_path)

    package_dir = Path(__file__).parent
    bin_dir = package_dir / "bin"

    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "linux":
        platform_name = "linux"
    elif system == "darwin":
        platform_name = "darwin"
    elif system == "windows":
        platform_name = "windows"
    else:
        platform_name = system

    platform_binary = bin_dir / f"onion-{platform_name}-{machine}"
    if platform_binary.exists() and platform_binary.is_file():
        logger.info(f"Found bundled platform-specific binary: {platform_binary}")
        return platform_binary

    generic_binary = bin_dir / "onion"
    if generic_binary.exists() and generic_binary.is_file():
        logger.info(f"Found bundled onion binary: {generic_binary}")
        return generic_binary

    Path(__file__).parents[2]
    local_builds = [
        package_dir / "onion" / "src_sc" / "onion",
        package_dir / "onion" / "src" / "onion",
    ]

    for local_bin in local_builds:
        if local_bin.exists() and local_bin.is_file():
            logger.info(f"Found local onion binary: {local_bin}")
            return local_bin

    logger.warning("Onion binary not found in PATH or package")
    return None


def run_onion(
    file_list_path: Path,
    output_dir: Path,
    dataset_name: str = "dataset",
    onion_binary: Path | None = None,
) -> tuple[bool, Path | None]:
    """
    Run the onion deduplication binary.

    Args:
        file_list_path: Path to file containing list of files to process
        output_dir: Directory for output CSV files
        dataset_name: Name of the dataset (for output files)
        onion_binary: Path to onion binary (auto-detected if None)

    Returns:
        Tuple of (success: bool, output_csv_dir: Path)
    """
    if onion_binary is None:
        onion_binary = find_onion_binary()

    if onion_binary is None:
        raise FileNotFoundError(
            "Onion binary not found. Please install onion or provide path to binary.\n"
            "Installation: https://corpus.tools/wiki/Onion"
        )

    onion_binary = Path(onion_binary)
    if not onion_binary.exists():
        raise FileNotFoundError(f"Onion binary not found at: {onion_binary}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # onion expects: ./onion -D <dataset-name> -L <file-list> -O <output-dir>
    cmd = [
        str(onion_binary),
        "-D",
        dataset_name,
        "-L",
        str(file_list_path),
        "-O",
        str(output_dir),
    ]

    logger.info(f"Running onion: {' '.join(cmd)}")
    logger.info(f"Processing file list: {file_list_path}")
    logger.info(f"Output directory: {output_dir}")

    try:
        # Run onion
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=21600,  # 6 hour timeout
        )

        if result.returncode != 0:
            logger.error(f"Onion failed with return code {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return False, None

        logger.info("Onion execution completed successfully")
        if result.stdout:
            logger.debug(f"Onion output: {result.stdout}")

        return True, output_dir

    except subprocess.TimeoutExpired:
        logger.error("Onion execution timed out after 1 hour")
        return False, None
    except Exception as e:
        logger.error(f"Error running onion: {e}")
        return False, None


def compile_onion(source_dir: Path) -> Path | None:
    """
    Compile the onion binary from source.

    Args:
        source_dir: Directory containing onion source code (with Makefile)

    Returns:
        Path to compiled binary or None if compilation failed
    """
    source_dir = Path(source_dir)

    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return None

    makefile = source_dir / "Makefile.g"
    if not makefile.exists():
        makefile = source_dir / "Makefile"

    if not makefile.exists():
        logger.error(f"No Makefile found in {source_dir}")
        return None

    logger.info(f"Compiling onion from source: {source_dir}")

    try:
        # Clean
        subprocess.run(
            ["make", "clean"],
            cwd=source_dir,
            capture_output=True,
            check=False,
        )

        # Compile
        makefile_flag = "-f Makefile.g" if (source_dir / "Makefile.g").exists() else ""
        compile_cmd = f"make onion {makefile_flag}".split()

        result = subprocess.run(
            compile_cmd,
            cwd=source_dir,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            logger.error(f"Compilation failed: {result.stderr}")
            return None

        # Check for binary
        binary_path = source_dir / "onion"
        if binary_path.exists():
            logger.info(f"Successfully compiled onion: {binary_path}")
            return binary_path
        else:
            logger.error("Compilation succeeded but binary not found")
            return None

    except Exception as e:
        logger.error(f"Error during compilation: {e}")
        return None


__all__ = [
    "find_onion_binary",
    "run_onion",
    "compile_onion",
]
