"""
Neurosymbolic SQL Adapter

A neurosymbolic reasoning adapter for fine-tuned SQL language models.
"""

__version__ = "0.1.0"
__author__ = "Saptak"
__email__ = "saptak@example.com"

# Import available modules
__all__ = []

try:
    from .reasoning.pyreason_engine import PyReasonEngine
    __all__.append("PyReasonEngine")
except ImportError:
    pass

try:
    from .integration.hybrid_model import NeurosymbolicSQLModel
    __all__.append("NeurosymbolicSQLModel")
except ImportError:
    pass

try:
    from .adapters.neurosymbolic_adapter import NeurosymbolicAdapter
    __all__.append("NeurosymbolicAdapter")
except ImportError:
    pass