from typing import Sequence, Tuple, List
import numpy as np


def detect_emergence(
    steps: Sequence[int],
    test_acc: Sequence[float],
    jump_threshold: float = 0.20,
    stability_tol: float = 0.02,
    stability_horizon: int = 3,
    require_full_horizon: bool = True,
) -> Tuple[np.ndarray, List[int]]:
    """
    Classify each training step as emergence / not-emergence based on a big jump
    in test accuracy followed by a short plateau of stability.

    A step i (i >= 1) is 'emergence' if:
      1) test_acc[i] - test_acc[i-1] >= jump_threshold
      2) For all k in [i, i + stability_horizon], |test_acc[k] - test_acc[i]| <= stability_tol
         If require_full_horizon=True, require that the full horizon exists.

    Returns:
      is_emergence: boolean array of shape (N,)
      indices: list of indices i where emergence is detected
    """
    a = np.asarray(test_acc, dtype=float)
    n = a.shape[0]
    if n < 2:
        return np.zeros(n, dtype=bool), []

    is_emergence = np.zeros(n, dtype=bool)
    for i in range(1, n):
        delta = a[i] - a[i - 1]
        if delta < jump_threshold:
            continue

        end = min(i + stability_horizon, n - 1)
        if require_full_horizon and end < i + stability_horizon:
            # Not enough future points to validate stability
            continue

        window = a[i : end + 1]
        if np.all(np.abs(window - a[i]) <= stability_tol):
            is_emergence[i] = True

    indices = np.flatnonzero(is_emergence).tolist()
    return is_emergence, indices


def label_emergence_steps(
    steps: Sequence[int],
    test_acc: Sequence[float],
    **kwargs
) -> List[int]:
    """Return the step numbers (not indices) where emergence is detected."""
    _, idx = detect_emergence(steps, test_acc, **kwargs)
    return [int(steps[i]) for i in idx]


