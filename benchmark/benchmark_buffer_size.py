"""Compare streaming decision-tree performance for different buffer sizes."""

import csv
import time
from pathlib import Path

import numpy as np

from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.stream import StreamTrainer


def main():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(300, 5))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

    chunk_size = 30
    buffer_sizes = [30, 60, 120]
    repeats = 5
    results = []

    print("Buffer | Retrains | Median retrain (ms) | Total fit (ms) | Accuracy")
    print("-" * 74)

    for buffer_size in buffer_sizes:
        repeat_retrains = []
        repeat_total_fit = []
        repeat_accuracy = []
        final_retrain_count = 0

        for _ in range(repeats):
            tree = DecisionTreeClassifier(buffer_size=buffer_size)
            pipeline = Pipeline([("tree", tree)])
            trainer = StreamTrainer(pipeline)

            retrain_times = []
            evaluated_accuracy = []
            total_fit_time = 0.0

            for start in range(0, len(X), chunk_size):
                X_chunk = X[start:start + chunk_size]
                y_chunk = y[start:start + chunk_size]

                # Evaluate before fitting this chunk to avoid scoring on training data.
                record = trainer.score_chunk(X_chunk, y_chunk)
                previous_retrains = tree.retrain_count
                trainer.fit_chunk(X_chunk, y_chunk)

                training_time = trainer.history[-1].get("training_time", 0.0)
                total_fit_time += training_time
                if tree.retrain_count > previous_retrains:
                    retrain_times.append(training_time)

                # Compare buffer settings on the same later chunks.
                if start >= max(buffer_sizes) and record.get("accuracy") is not None:
                    evaluated_accuracy.append(record["accuracy"])

            final_retrain_count = tree.retrain_count
            repeat_retrains.append(len(retrain_times))
            repeat_total_fit.append(total_fit_time * 1000)
            if retrain_times:
                repeat_retrains[-1] = len(retrain_times)
                median_retrain = float(np.median(retrain_times) * 1000)
            else:
                median_retrain = 0.0
            # Store one representative median retraining time per repeat.
            if "repeat_median_retrain" not in locals():
                repeat_median_retrain = []
            repeat_median_retrain.append(median_retrain)
            repeat_accuracy.append(float(np.mean(evaluated_accuracy)) if evaluated_accuracy else float("nan"))

        median_retrain_ms = float(np.median(repeat_median_retrain))
        median_total_ms = float(np.median(repeat_total_fit))
        mean_accuracy = float(np.nanmean(repeat_accuracy))
        results.append([buffer_size, final_retrain_count, median_retrain_ms,
                        median_total_ms, mean_accuracy])
        print(f"{buffer_size:6d} | {final_retrain_count:8d} | "
              f"{median_retrain_ms:19.3f} | {median_total_ms:14.3f} | {mean_accuracy:.3f}")
        del repeat_median_retrain

    output = Path(__file__).resolve().parent / "buffer_size_results.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Buffer_Size", "Retrains", "Median_Retrain_ms",
                         "Total_Fit_ms", "Common_Accuracy"])
        writer.writerows(results)
    print(f"\nSaved results to: {output}")


if __name__ == "__main__":
    main()
