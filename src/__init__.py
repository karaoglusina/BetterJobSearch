"""BetterJobSearch - Job market analysis with NLP pipeline, multi-agent system, and React UI."""

# Set environment variables BEFORE any scientific/ML library imports
# to prevent OpenMP conflicts and segfaults on Apple Silicon (M1/M2/M3).
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
del _os

__version__ = "2.0.0"
__author__ = "Sina Karaoglu"

# Main modules are importable directly from the package
__all__ = []