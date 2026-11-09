"""Modular RAG experimentation and evaluation framework."""

from rag_benchmark.paths import configure_cache_environment

configure_cache_environment()

__all__ = ["__version__"]

__version__ = "0.1.0"
