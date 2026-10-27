"""NumPy and matplotlib edition of the Monte Carlo investment simulator."""

# NumPy supplies vectorized simulation and statistical operations; matplotlib
# Creates charts without requiring an interactive display.
# The Agg backend makes chart export reliable in command-line and headless environments, while the standard-library helpers are shared with Version 1.
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from investment_model import format_currency, format_probability, make_seed
from monte_carlo_v1 import collect_inputs, SENSITIVITY_SIMULATIONS


MAX_PLOT_PATHS = 30
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "generated-artifacts" / "monte-carlo"


# Vectorized simulation:
# NumPy generates the return matrix in one operation, with rows representing trials and columns representing years, then compounds each path with cumprod.
# A fixed seed makes the full matrix reproducible; only up to 30 paths are selected for charting to keep the visual uncluttered.
# The -100% floor prevents negative portfolio values while preserving the model's documented caveat that extreme negative-tail outcomes are truncated.
def simulate(inputs, seed=None):
    """Run the simulation with NumPy's generator and vectorized portfolio paths."""
    chosen_seed = inputs["seed"] if seed is None else seed
    generator = np.random.default_rng(chosen_seed)
    expected_return = inputs["expected_return_pct"] / 100
    volatility = inputs["volatility_pct"] / 100

    annual_returns = generator.normal(
        loc=expected_return,
        scale=volatility,
        size=(inputs["num_simulations"], inputs["years"]),
    )
    annual_returns = np.maximum(annual_returns, -1.0)
    growth_factors = 1 + annual_returns

    paths = np.empty(
        (inputs["num_simulations"], inputs["years"] + 1), dtype=np.float64
    )
    paths[:, 0] = inputs["initial_investment"]
    paths[:, 1:] = inputs["initial_investment"] * np.cumprod(
        growth_factors, axis=1
    )
    final_values = paths[:, -1]
    sample_indices = np.linspace(
        0,
        inputs["num_simulations"] - 1,
        num=min(MAX_PLOT_PATHS, inputs["num_simulations"]),
        dtype=int,
    )

    return {
        "final_values": final_values,
        "paths": paths,
        "sample_paths": paths[sample_indices],
        "years": inputs["years"],
    }


# Statistical analysis:
# NumPy calculates the mean, median, range, and percentile cutoffs directly from the vector of final values, avoiding Python-level loops over simulation results.
# The loss probability is the fraction of trials ending below the initial capital; average gain/loss is the mean terminal value minus that starting capital.
def summarize_numpy(final_values, initial_investment):
    """Calculate simulation summary statistics with NumPy."""
    values = np.asarray(final_values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("At least one final portfolio value is required.")

    mean_value = float(np.mean(values))
    percentiles = np.percentile(values, [5, 25, 50, 75, 95])
    return {
        "mean": mean_value,
        "median": float(percentiles[2]),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "p5": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "p75": float(percentiles[3]),
        "p95": float(percentiles[4]),
        "probability_below_initial": float(np.mean(values < initial_investment)),
        "average_gain_loss": mean_value - initial_investment,
    }


# Results reporting:
# Presents the same financial measures as Version 1 so users can compare the standard-library and vectorized implementations on equivalent assumptions.
# Percentiles describe the simulated outcome distribution; they are not confidence guarantees or forecasts of actual market performance.
def print_summary(summary, initial_investment):
    """Print the main results in a readable format."""
    print("\nSimulation results")
    print("-" * 44)
    print(f"Initial investment:                 {format_currency(initial_investment)}")
    print(f"Mean final value:                   {format_currency(summary['mean'])}")
    print(f"Median final value:                 {format_currency(summary['median'])}")
    print(f"Minimum final value:                {format_currency(summary['minimum'])}")
    print(f"Maximum final value:                {format_currency(summary['maximum'])}")
    print(f"5th percentile:                     {format_currency(summary['p5'])}")
    print(f"25th percentile:                    {format_currency(summary['p25'])}")
    print(f"75th percentile:                    {format_currency(summary['p75'])}")
    print(f"95th percentile:                    {format_currency(summary['p95'])}")
    print(
        "Chance of finishing below initial: "
        f"{format_probability(summary['probability_below_initial'])}"
    )
    print(f"Average gain or loss:               {format_currency(summary['average_gain_loss'])}")


# Sensitivity analysis:
# Re-runs a baseline and a one-variable alternative with the same seed, helping attribute differences to the selected assumption rather than random noise.
# The scenario table compares median value and loss probability; a separate simulation-count comparison illustrates Monte Carlo sampling stability.
def run_one_sensitivity_case(base_inputs, changes, seed, simulation_count):
    """Run a baseline and one changed assumption with the same seeded randomness."""
    baseline_inputs = dict(base_inputs)
    baseline_inputs["num_simulations"] = simulation_count
    baseline_inputs["seed"] = seed

    changed_inputs = dict(baseline_inputs)
    changed_inputs.update(changes)

    baseline_result = simulate(baseline_inputs, seed)
    changed_result = simulate(changed_inputs, seed)
    baseline_summary = summarize_numpy(
        baseline_result["final_values"], baseline_inputs["initial_investment"]
    )
    changed_summary = summarize_numpy(
        changed_result["final_values"], changed_inputs["initial_investment"]
    )
    return baseline_summary, changed_summary


def print_sensitivity(inputs, seed):
    """Compare one assumption at a time using reproducible paired simulations."""
    scenario_count = max(1_000, min(inputs["num_simulations"], SENSITIVITY_SIMULATIONS))
    expected_change = inputs["expected_return_pct"] + 3
    volatility_change = inputs["volatility_pct"] + 15

    print("\nOne-at-a-time sensitivity checks")
    print(
        f"Each comparison holds other assumptions steady; return/volatility/horizon "
        f"checks use {scenario_count:,} simulations."
    )
    print("Changes use the brief's example increments: +3 percentage points return,")
    print("+15 points volatility, and a comparison with a 20-year horizon.")
    print("-" * 88)
    print(
        f"{'Assumption change':<29} {'Median: baseline':>19} "
        f"{'Median: changed':>19} {'Below initial':>17}"
    )
    print("-" * 88)

    scenarios = []
    if expected_change <= 1_000:
        scenarios.append(
            (
                f"Return {inputs['expected_return_pct']:g}% → {expected_change:g}%",
                {"expected_return_pct": expected_change},
            )
        )
    else:
        print("Skipped expected return comparison: +3 points would exceed the input limit.")

    if volatility_change <= 500:
        scenarios.append(
            (
                f"Volatility {inputs['volatility_pct']:g}% → {volatility_change:g}%",
                {"volatility_pct": volatility_change},
            )
        )
    else:
        print("Skipped volatility comparison: +15 points would exceed the input limit.")

    comparison_years = 5 if inputs["years"] == 20 else 20
    scenarios.append(
        (
            f"Horizon {inputs['years']} → {comparison_years} years",
            {"years": comparison_years},
        )
    )

    for label, changes in scenarios:
        baseline, changed = run_one_sensitivity_case(
            inputs, changes, seed, scenario_count
        )
        print(
            f"{label:<29} {format_currency(baseline['median']):>19} "
            f"{format_currency(changed['median']):>19} "
            f"{format_probability(baseline['probability_below_initial'])} → "
            f"{format_probability(changed['probability_below_initial'])}"
        )

    low_count = 1_000
    high_count = 10_000
    low_inputs = dict(inputs)
    low_inputs["num_simulations"] = low_count
    low_inputs["seed"] = seed
    high_inputs = dict(inputs)
    high_inputs["num_simulations"] = high_count
    high_inputs["seed"] = seed
    low_result = simulate(low_inputs, seed)
    high_result = simulate(high_inputs, seed)
    low_summary = summarize_numpy(
        low_result["final_values"], inputs["initial_investment"]
    )
    high_summary = summarize_numpy(
        high_result["final_values"], inputs["initial_investment"]
    )
    print(
        f"Simulations {low_count:,} → {high_count:,} "
        f"{format_currency(low_summary['median']):>19} "
        f"{format_currency(high_summary['median']):>19} "
        f"{format_probability(low_summary['probability_below_initial'])} → "
        f"{format_probability(high_summary['probability_below_initial'])}"
    )
    print(
        "More simulations improve the stability of estimated statistics; they do not "
        "change the market assumptions."
    )


# Chart generation:
# Exports a final-value histogram, a limited set of sample paths, and a comparison of starting capital with mean and median terminal values.
# Seed-specific filenames preserve the link between the input assumptions and generated charts; output is written to the ignored generated-artifacts folder.
def save_charts(result, summary, initial_investment, seed):
    """Save the histogram, sample paths, and initial/mean/median comparison."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_values = result["final_values"]
    chart_paths = []

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.hist(
        final_values,
        bins=min(60, max(12, int(np.sqrt(len(final_values))))),
        color="#5b7cfa",
        edgecolor="white",
        alpha=0.9,
    )
    axis.axvline(
        initial_investment,
        color="#d35454",
        linestyle="--",
        linewidth=2,
        label=f"Initial: {format_currency(initial_investment)}",
    )
    axis.axvline(
        summary["median"],
        color="#1b8a72",
        linestyle="-",
        linewidth=2,
        label=f"Median: {format_currency(summary['median'])}",
    )
    axis.set_title("Distribution of final portfolio values")
    axis.set_xlabel("Final portfolio value (£)")
    axis.set_ylabel("Number of simulations")
    axis.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"£{value:,.0f}")
    )
    axis.legend()
    figure.tight_layout()
    chart_path = OUTPUT_DIR / f"final-value-histogram-seed-{seed}.png"
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)
    chart_paths.append(chart_path)

    figure, axis = plt.subplots(figsize=(10, 6))
    for path in result["sample_paths"]:
        axis.plot(range(result["years"] + 1), path, alpha=0.45, linewidth=1.2)
    axis.axhline(
        initial_investment,
        color="#d35454",
        linestyle="--",
        linewidth=2,
        label="Initial investment",
    )
    axis.set_title("Sample simulated investment paths")
    axis.set_xlabel("Year")
    axis.set_ylabel("Portfolio value (£)")
    axis.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"£{value:,.0f}")
    )
    axis.legend()
    figure.tight_layout()
    chart_path = OUTPUT_DIR / f"sample-paths-seed-{seed}.png"
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)
    chart_paths.append(chart_path)

    labels = ["Initial investment", "Mean final value", "Median final value"]
    values = [initial_investment, summary["mean"], summary["median"]]
    colors = ["#d3a949", "#5b7cfa", "#1b8a72"]
    figure, axis = plt.subplots(figsize=(9, 6))
    bars = axis.bar(labels, values, color=colors, width=0.62)
    axis.set_title("Initial investment compared with simulated outcomes")
    axis.set_ylabel("Portfolio value (£)")
    axis.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"£{value:,.0f}")
    )
    axis.tick_params(axis="x", rotation=12)
    axis.bar_label(
        bars,
        labels=[format_currency(value) for value in values],
        padding=4,
        fontsize=9,
    )
    figure.tight_layout()
    chart_path = OUTPUT_DIR / f"outcome-comparison-seed-{seed}.png"
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)
    chart_paths.append(chart_path)

    return chart_paths


# Program entry point:
# Reuses Version 1's validated prompts, then runs NumPy simulation, analysis, sensitivity checks, and chart export in a single command-line workflow.
# Keeping execution behind this guard allows the functions to be imported safely by tests or other Python code without starting an interactive run.
def main():
    inputs = collect_inputs()
    effective_seed = make_seed(inputs["seed"])
    print(f"\nUsing random seed: {effective_seed}")

    result = simulate(inputs, effective_seed)
    summary = summarize_numpy(result["final_values"], inputs["initial_investment"])
    print_summary(summary, inputs["initial_investment"])
    print_sensitivity(inputs, effective_seed)

    chart_paths = save_charts(
        result, summary, inputs["initial_investment"], effective_seed
    )
    print("\nCharts saved:")
    for chart_path in chart_paths:
        print(f"- {chart_path}")

    print(
        "\nA simulation is not a guarantee or forecast of actual market performance."
    )


if __name__ == "__main__":
    main()
