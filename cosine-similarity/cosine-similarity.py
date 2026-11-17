import numpy as np


def cosine_similarity(a: list, b: list) -> float:
    """Computes the cosine similarity between two 1D numeric lists."""
    # Convert input lists to NumPy arrays
    arr_a = np.array(a)
    arr_b = np.array(b)

    # Compute dot product and norms
    num = arr_a @ arr_b  # Modern, clean matrix/vector dot product operator
    denom = np.linalg.norm(arr_a) * np.linalg.norm(arr_b)

    # Guard clause: Return 0.0 if either vector has a zero norm to prevent division by zero
    if denom == 0.0:
        return 0.0

    return float(num / denom)