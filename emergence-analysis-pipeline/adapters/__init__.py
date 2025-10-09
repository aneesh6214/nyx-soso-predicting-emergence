"""Adapters for different experiment types."""

from .base_adapter import BaseAdapter

__all__ = ['BaseAdapter']

# Import specific adapters when available
try:
    from .grokking_adapter import GrokkingAdapter
    __all__.append('GrokkingAdapter')
except ImportError:
    pass

try:
    from .pythia_adapter import PythiaAdapter
    __all__.append('PythiaAdapter')
except ImportError:
    pass

try:
    from .generic_adapter import GenericAdapter
    __all__.append('GenericAdapter')
except ImportError:
    pass
