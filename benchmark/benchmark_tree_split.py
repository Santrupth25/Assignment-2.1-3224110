
"""Compare loop-based and vectorised decision-tree split methods."""

import csv
import time
from pathlib import Path

import numpy as np

from numcompute_stream.tree import DecisionTreeClassifier


def main():
    """Run the benchmark and save the results to a CSV file."""
    rng = np.random.default_rng(42)
    sample_sizes = [100, 500, 1000, 2000]
    repeats = 5
    results = []

    print(
        f"{'Samples':>7} | {'Loop (ms)':>10} | "
        f"{'Vectorised (ms)':>15} | {'Speedup':>8}"
    )
    print("-" * 53)

    tree = DecisionTreeClassifier()

    for size in sample_sizes:
        X = rng.normal(size=(size, 5))
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

        # Initialise the class information required by both methods.
        tree.fit(X, y)

        # Check that both methods return equivalent results.
        loop_result = tree._best_split_loop(X, y)
        vector_result = tree._best_split_vectorized(X, y)

        if loop_result[0] != vector_result[0]:
            raise AssertionError(
                f"Feature mismatch at sample size {size}: "
                f"{loop_result[0]} != {vector_result[0]}"
            )

        if loop_result[0] is not None:
            if not np.isclose(loop_result[1], vector_result[1]):
                raise AssertionError(
                    f"Threshold mismatch at sample size {size}"
                )

        if not np.isclose(loop_result[2], vector_result[2]):
            raise AssertionError(
                f"Score mismatch at sample size {size}"
            )

        # Warm up both methods before measuring execution time.
        tree._best_split_loop(X, y)
        tree._best_split_vectorized(X, y)

        loop_times = []
        vector_times = []

        for _ in range(repeats):
            start = time.perf_counter()
            tree._best_split_loop(X, y)
            loop_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            tree._best_split_vectorized(X, y)
            vector_times.append(time.perf_counter() - start)

        loop_ms = float(np.median(loop_times) * 1000)
        vector_ms = float(np.median(vector_times) * 1000)

        speedup = (
            loop_ms / vector_ms
            if vector_ms > 0
            else float("inf")
        )

        results.append({
            "Samples": size,
            "Loop_ms": loop_ms,
            "Vectorized_ms": vector_ms,
            "Speedup": speedup,
        })

        print(
            f"{size:7d} | {loop_ms:10.3f} | "
            f"{vector_ms:15.3f} | {speedup:7.2f}x"
        )

    output = (
        Path(__file__).resolve().parent
        / "tree_split_results.csv"
    )

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Samples",
                "Loop_ms",
                "Vectorized_ms",
                "Speedup",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()
