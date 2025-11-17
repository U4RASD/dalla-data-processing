#!/bin/bash
# Build Onion binary and copy to package directory
# This script is used by CI to build platform-specific Onion binaries

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Building Onion Binary ===${NC}"

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ONION_SOURCE="$PROJECT_ROOT/dalla/deduplication/onion/src_sc"
OUTPUT_DIR="$PROJECT_ROOT/dalla/deduplication/bin"

# Check if source exists
if [ ! -d "$ONION_SOURCE" ]; then
    echo -e "${RED}Error: Onion source not found at $ONION_SOURCE${NC}"
    exit 1
fi

echo -e "${YELLOW}Onion source: $ONION_SOURCE${NC}"

# Check for required dependencies
if ! command -v g++ &> /dev/null; then
    echo -e "${RED}Error: g++ compiler not found${NC}"
    echo "Please install build-essential (Ubuntu) or gcc (macOS)"
    exit 1
fi

# Check for Google sparsehash
echo -e "${YELLOW}Checking for Google sparsehash...${NC}"
if ! echo '#include <google/sparse_hash_set>' | g++ -x c++ -c - -o /dev/null 2>/dev/null; then
    echo -e "${YELLOW}Warning: Google sparsehash headers not found${NC}"
    echo "Attempting to install sparsehash..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y libsparsehash-dev
        elif command -v yum &> /dev/null; then
            sudo yum install -y sparsehash-devel
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            brew install google-sparsehash
        fi
    fi
fi

# Clean previous build
echo -e "${YELLOW}Cleaning previous build...${NC}"
cd "$ONION_SOURCE"
make clean 2>/dev/null || true

# Set up compiler flags for macOS
EXTRA_CFLAGS=""
if [[ "$OSTYPE" == "darwin"* ]]; then
    # On macOS, add Homebrew paths for sparsehash
    if command -v brew &> /dev/null; then
        BREW_PREFIX=$(brew --prefix)
        EXTRA_CFLAGS="-I${BREW_PREFIX}/include"
        echo -e "${YELLOW}Using Homebrew prefix: ${BREW_PREFIX}${NC}"
    fi
fi

# Build onion
echo -e "${YELLOW}Compiling Onion...${NC}"
if make CFLAGS="-Wall -O3 ${EXTRA_CFLAGS}"; then
    echo -e "${GREEN}✓ Compilation successful${NC}"
else
    echo -e "${RED}✗ Compilation failed${NC}"
    exit 1
fi

# Check if binary was created
if [ ! -f "$ONION_SOURCE/onion" ]; then
    echo -e "${RED}Error: Binary not found after compilation${NC}"
    exit 1
fi

# Get platform info
PLATFORM=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
echo -e "${YELLOW}Platform: $PLATFORM-$ARCH${NC}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Copy binary to package
echo -e "${YELLOW}Copying binary to package...${NC}"
cp "$ONION_SOURCE/onion" "$OUTPUT_DIR/onion-$PLATFORM-$ARCH"
chmod +x "$OUTPUT_DIR/onion-$PLATFORM-$ARCH"

# Create a generic symlink for the current platform
ln -sf "onion-$PLATFORM-$ARCH" "$OUTPUT_DIR/onion"

echo -e "${GREEN}=== Build Complete ===${NC}"
echo -e "${GREEN}Binary location: $OUTPUT_DIR/onion-$PLATFORM-$ARCH${NC}"

# Verify binary works
if "$OUTPUT_DIR/onion-$PLATFORM-$ARCH" -h &> /dev/null; then
    echo -e "${GREEN}✓ Binary is executable and working${NC}"
else
    echo -e "${YELLOW}Warning: Binary may not be fully functional${NC}"
fi

# Print file size
BINARY_SIZE=$(du -h "$OUTPUT_DIR/onion-$PLATFORM-$ARCH" | cut -f1)
echo -e "${GREEN}Binary size: $BINARY_SIZE${NC}"
