"""Data loaders for different data sources.

This package provides pluggable data loaders for various benchmark data sources,
with support for streaming, caching, and efficient data processing.
"""

from .huggingface import HuggingFaceLoader

# Registry of available loaders
LOADERS = {
    "huggingface": HuggingFaceLoader,
}


def get_loader(source: str):
    """Get the appropriate loader for a data source."""
    if source not in LOADERS:
        raise ValueError(
            f"Unknown data source: {source}. Available: {list(LOADERS.keys())}"
        )

    return LOADERS[source]()


__all__ = ["get_loader", "HuggingFaceLoader"]
