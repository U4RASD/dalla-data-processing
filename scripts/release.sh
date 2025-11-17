#!/bin/bash

set -e

VERSION=$1

if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 0.2.0"
    exit 1
fi

if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Version must be in format X.Y.Z (e.g., 0.2.0)"
    exit 1
fi

echo "==> Preparing release v$VERSION"

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "main" ]; then
    echo "Warning: Not on main branch (currently on $BRANCH)"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

if [[ -n $(git status -s) ]]; then
    echo "Error: Working directory is not clean. Commit or stash changes first."
    git status -s
    exit 1
fi

echo "==> Running linter..."
ruff check dalla/ || {
    echo "Error: Linting failed"
    exit 1
}

echo "==> Checking formatting..."
ruff format --check dalla/ || {
    echo "Error: Code not formatted. Run: ruff format dalla/"
    exit 1
}

echo "==> Updating version in pyproject.toml..."
sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml
rm pyproject.toml.bak

echo "==> Updating version in dalla/__init__.py..."
sed -i.bak "s/^__version__ = \".*\"/__version__ = \"$VERSION\"/" dalla/__init__.py
rm dalla/__init__.py.bak

echo "==> Committing version bump..."
git add pyproject.toml dalla/__init__.py
git commit -m "Bump version to $VERSION"

echo "==> Creating tag v$VERSION..."
git tag -a "v$VERSION" -m "Release version $VERSION"

echo ""
echo "==> Ready to release!"
echo "    Version: $VERSION"
echo "    Tag: v$VERSION"
echo "    Commit: $(git rev-parse HEAD)"
echo ""
echo "Next steps:"
echo "  1. Review changes: git log -1"
echo "  2. Push: git push && git push --tags"
echo "  3. GitHub Actions will automatically:"
echo "     - Build wheels"
echo "     - Publish to PyPI"
echo "     - Create GitHub Release"
echo "     - Build Docker image"
echo ""

read -p "Push now? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git push
    git push --tags
    echo ""
    echo "✓ Release v$VERSION pushed!"
    echo "✓ Check progress: https://github.com/$(git config --get remote.origin.url | sed 's/.*github.com[:/]\(.*\)\.git/\1/')/actions"
else
    echo ""
    echo "Not pushed. You can push manually with:"
    echo "  git push && git push --tags"
fi
