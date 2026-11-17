"""Compare collision-search methods at increasing particle counts.

Run from the repository root, for example::

    PYTHONPATH=src python3 optimisations/benchmark.py \
        --balls 37 100 250 --collisions 100 --repeats 3 \
        --csv timings.csv --plot timings.png

Only ``Simulation.run()`` is timed. Construction, correctness checks, CSV
writing, and plotting are excluded; rebuilding a quadtree during a run is
included. Both methods receive identical initial conditions for each repeat.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
from math import ceil, sqrt
from pathlib import Path
from statistics import median
from time import perf_counter


METHODS = ("brute_force", "quadtree")


@dataclass(frozen=True)
class Timing:
    """One complete run, also used as a raw CSV row."""

    balls: int
    collisions: int
    repeat: int
    seed: int
    method: str
    seconds: float
    simulated_time: float


def simulation_parameters(num_balls):
    """Make separated rings with approximately constant particle density.

    Adjacent rings are three units apart. With six points per ring multiplier,
    neighbouring centres on a ring are at least three units apart as well.
    Radius-0.5 balls therefore have ample clearance. Arbitrary counts leave a
    partially filled outer ring, so the density is only approximately constant.
    """
    nrings = max(1, ceil((sqrt(1 + 4 * (num_balls - 1) / 3) - 1) / 2))
    rmax = 3.0 * nrings
    return {
        "num_balls": num_balls,
        "nrings": nrings,
        "multi": 6,
        "rmax": rmax,
        "c_radius": rmax + 3.0,
        "b_radius": 0.5,
        "b_speed": 10.0,
    }


def _check_same_state(first, second, num_balls, repeat):
    """Check that a faster search has not changed the simulated result."""
    import numpy as np

    first_state = np.concatenate(
        [first.container().pos(), first.container().vel()]
        + [np.concatenate((ball.pos(), ball.vel())) for ball in first.balls()]
    )
    second_state = np.concatenate(
        [second.container().pos(), second.container().vel()]
        + [np.concatenate((ball.pos(), ball.vel())) for ball in second.balls()]
    )
    same_time = np.isclose(first.time(), second.time(), rtol=1e-9, atol=1e-8)
    same_state = np.allclose(first_state, second_state, rtol=1e-9, atol=1e-8)
    same_impulse = np.isclose(
        first.container().dp_tot(), second.container().dp_tot(),
        rtol=1e-9, atol=1e-8,
    )
    if not (same_time and same_state and same_impulse):
        raise RuntimeError(
            f"Collision searches disagree after {num_balls}-ball repeat {repeat}. "
            "Inspect the search results before interpreting their timings."
        )


def benchmark(ball_counts, collisions=100, repeats=3, seed=12345):
    """Time paired runs and return raw results after checking agreement.

    Imports and one warm-up event per method happen before timed runs. Method
    order alternates to reduce a systematic advantage from always running first.
    """
    from thermosnooker.simulations import MultiBallSimulation

    for method in METHODS:
        warm_up = MultiBallSimulation(
            **simulation_parameters(7), seed=seed, collision_search=method,
        )
        warm_up.run(1, animate=False)

    results = []
    print("Median runtime in seconds (construction and plotting excluded)")
    print(f"{'Balls':>7} {'Brute force':>14} {'Quadtree':>14} {'Speedup':>10}")
    for count_index, num_balls in enumerate(ball_counts):
        count_results = []
        for repeat in range(repeats):
            run_seed = seed + repeat
            simulations = {
                method: MultiBallSimulation(
                    **simulation_parameters(num_balls),
                    seed=run_seed,
                    collision_search=method,
                )
                for method in METHODS
            }
            order = METHODS if (count_index + repeat) % 2 == 0 else METHODS[::-1]
            for method in order:
                simulation = simulations[method]
                start = perf_counter()
                simulation.run(collisions, animate=False)
                elapsed = perf_counter() - start
                count_results.append(Timing(
                    balls=num_balls,
                    collisions=collisions,
                    repeat=repeat + 1,
                    seed=run_seed,
                    method=method,
                    seconds=elapsed,
                    simulated_time=simulation.time(),
                ))
            _check_same_state(
                simulations["brute_force"], simulations["quadtree"],
                num_balls, repeat + 1,
            )
        results.extend(count_results)
        times = {
            method: median(row.seconds for row in count_results if row.method == method)
            for method in METHODS
        }
        speedup = times["brute_force"] / times["quadtree"]
        print(
            f"{num_balls:7d} {times['brute_force']:14.6f} "
            f"{times['quadtree']:14.6f} {speedup:9.2f}x",
            flush=True,
        )
    return results


def write_csv(results, path):
    """Save every repeat, rather than only the reported medians."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=Timing.__dataclass_fields__)
        writer.writeheader()
        writer.writerows(asdict(row) for row in results)


def plot_results(results, path):
    """Save runtime scaling and speedup; bars show the repeat range."""
    import matplotlib.pyplot as plt

    counts = sorted({row.balls for row in results})
    fig, (runtime_ax, speedup_ax) = plt.subplots(1, 2, figsize=(11, 4.2))
    medians = {}
    for method in METHODS:
        samples = [
            [row.seconds for row in results if row.balls == count and row.method == method]
            for count in counts
        ]
        values = [median(sample) for sample in samples]
        medians[method] = values
        errors = [
            [value - min(sample) for sample, value in zip(samples, values)],
            [max(sample) - value for sample, value in zip(samples, values)],
        ]
        runtime_ax.errorbar(
            counts, values, yerr=errors, marker="o", capsize=3,
            label=method.replace("_", " "),
        )
    runtime_ax.set(
        xscale="linear", yscale="linear", xlabel="Number of balls",
        ylabel="Median runtime (seconds)", title="Fixed collision count",
    )
    runtime_ax.legend()
    runtime_ax.grid(True, which="both", alpha=0.25)
    speedups = [
        brute / tree
        for brute, tree in zip(medians["brute_force"], medians["quadtree"])
    ]
    speedup_ax.plot(counts, speedups, marker="o")
    speedup_ax.axhline(1, color="grey", linestyle="--", linewidth=1)
    speedup_ax.set(
        xscale="linear", xlabel="Number of balls",
        ylabel="Brute-force runtime / quadtree runtime", title="Speedup (>1 favours quadtree)",
    )
    speedup_ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def main(argv=None):
    """Parse CLI options and run the benchmark without opening GUI windows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--balls", nargs="+", type=_positive_int, default=[37, 100, 250],
        help="particle counts to compare (default: 37 100 250)",
    )
    parser.add_argument(
        "--collisions", type=_positive_int, default=100,
        help="events per run (default: 100)",
    )
    parser.add_argument(
        "--repeats", type=_positive_int, default=3,
        help="paired runs per particle count (default: 3)",
    )
    parser.add_argument(
        "--seed", type=_nonnegative_int, default=12345,
        help="initial seed; each repeat increments it (default: 12345)",
    )
    parser.add_argument("--csv", type=Path, help="save every timing to this CSV path")
    parser.add_argument("--plot", type=Path, help="save a runtime/speedup figure to this path")
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    results = benchmark(
        sorted(set(args.balls)), args.collisions, args.repeats, args.seed,
    )
    if args.csv:
        write_csv(results, args.csv)
        print(f"Saved raw timings to {args.csv}")
    if args.plot:
        plot_results(results, args.plot)
        print(f"Saved comparison plot to {args.plot}")


if __name__ == "__main__":
    main()
