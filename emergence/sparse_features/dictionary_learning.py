"""
Dictionary-learning based sparse feature extractor (placeholder).

The eventual implementation will likely:
* Accept batched activation tensors from transformer layers.
* Perform dimensionality reduction / whitening as needed.
* Learn an overcomplete dictionary using algorithms such as k-SVD, online
  dictionary learning, or sparse NMF.
* Encode new activations into sparse codes via L1-regularised optimisation.

None of those pieces are implemented yet – this file only sketches the public
API we expect to converge on.
"""

from __future__ import annotations

from typing import Any


class DictionaryLearner:  # noqa: D101 – skeleton
    def __init__(self, *, n_components: int = 1024, alpha: float = 1.0) -> None:  # noqa: D401,E501
        # Parameters are placeholders and subject to change.
        self.n_components = n_components
        self.alpha = alpha

    def fit(self, activations: "Any") -> None:  # noqa: D401,ANN401
        """Placeholder `fit` method – learns the dictionary from activations."""
        raise NotImplementedError

    def transform(self, activations: "Any") -> "Any":  # noqa: D401,ANN401
        """Encodes activations into sparse codes (placeholder)."""
        raise NotImplementedError

    def fit_transform(self, activations: "Any") -> "Any":  # noqa: D401,ANN401
        """Convenience wrapper – not implemented yet."""
        raise NotImplementedError
