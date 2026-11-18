"""Compare simple Python loops with NumPy vectorised operations."""

import csv
import platform
import statistics
import time
from pathlib import Path

import numpy as np


def mean_loop(data):
    """Calculate the mean using a Python loop."""
    if len(data) == 0:
        raise ValueError("Data cannot be empty")
    total = 0.0
    for value in data:
        total += value
    return total / len(data)


def square_loop(data):
    """Square each value using a Python loop."""
    result = []
    for value in data:
        result.append(value * value)
    return result


def distance_loop(a, b):
    """Calculate Euclidean distance using a Python loop."""
    if len(a) != len(b):
        raise ValueError("Arrays must have equal lengths")
    total = 0.0
    for x, y in zip(a, b):
        total += (x - y) ** 2
    return total ** 0.5


def measure_time(func, repeats=5):
    """Return the median execution time in seconds."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def run_benchmarks():
    """Compare loop and NumPy results over several array sizes."""
    rng = np.random.default_rng(42)
    results = []

    for size in [100, 1_000, 10_000, 100_000]:
        a = rng.random(size)
        b = rng.random(size)
        tests = [
            ("Mean", lambda: mean_loop(a), lambda: np.mean(a)),
            ("Square", lambda: square_loop(a), lambda: np.square(a)),
            ("Euclidean distance", lambda: distance_loop(a, b),
             lambda: np.linalg.norm(a - b)),
        ]

        for name, loop_func, numpy_func in tests:
            loop_result = np.asarray(loop_func())
            numpy_result = np.asarray(numpy_func())
            if not np.allclose(loop_result, numpy_result, rtol=1e-9, atol=1e-12):
                raise ValueError(f"Results do not match for {name}")

            loop_func()  # warm-up
            numpy_func()
            loop_seconds = measure_time(loop_func)
            numpy_seconds = measure_time(numpy_func)
            results.append({
                "operation": name,
                "size": size,
                "loop_seconds": loop_seconds,
                "numpy_seconds": numpy_seconds,
                "speedup": loop_seconds / numpy_seconds if numpy_seconds else None,
            })
    return results


def main():
    """Run benchmarks, print a table and save the CSV results."""
    print("NumCompute benchmark")
    print("Python:", platform.python_version())
    print("NumPy:", np.__version__)
    print("System:", platform.platform())
    print("Timing: median of 5 runs\n")
    results = run_benchmarks()

    print(f"{'Operation':<22} {'Size':>8} {'Loop (s)':>12} {'NumPy (s)':>12} {'Speedup':>10}")
    print("-" * 68)
    for row in results:
        speedup = row["speedup"]
        speedup_text = f"{speedup:.2f}x" if speedup is not None else "N/A"
        print(f"{row['operation']:<22} {row['size']:>8} "
              f"{row['loop_seconds']:>12.6g} {row['numpy_seconds']:>12.6g} "
              f"{speedup_text:>10}")

    output = Path(__file__).resolve().parent / "benchmark_results.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved results to: {output}")


if __name__ == "__main__":
    main()
