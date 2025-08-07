"""
Sparse feature extraction utilities (skeleton).

This sub-package will contain algorithms and helpers for learning and
manipulating sparse representations of transformer activations.

Modules planned
---------------
1. `dictionary_learning` – classic sparse coding / k-SVD style algorithms that
   operate on activation matrices to learn a basis ("dictionary") of feature
   directions.
2. `autoencoder` – torch-based sparse autoencoders that can be trained to
   produce sparse codes directly.
3. `utils` – shared helpers for batching, checkpointing, and visualisation.

At this stage the package only provides stub modules so that other code can
import them without errors. Functional code will be added once we finalise the
exact extraction approach.
"""
