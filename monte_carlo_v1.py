# This edition is intentionally limited to Python's standard library to keep the
# Simulation logic transparent and easy to compare with the NumPy implementation.
# Shared validation and summary helpers live in investment_model.py; this file
# Focuses on simulation flow, command-line interaction, and sensitivity checks.
import random

from investment_model import (
    MAX_SIMULATIONS,
    MAX_YEARS,
    format_currency,
    format_probability,
    make_seed,
    summarize_final_values,
    validate_inputs,
)


MAX_SAMPLE_PATHS = 12
SENSITIVITY_SIMULATIONS = 10_000


# Simulation engine:
# Each trial draws independent annual simple returns from a normal distribution, compounds them into a portfolio path, and records the terminal value.
# Returns are floored at -100% so a sampled year cannot produce a negative balance.
# All terminal values are retained for statistics, while only a small, evenly spaced sample of paths is kept for visualization.
def simulate(inputs, seed=None):
    """Run the simulation with loops, lists, and random.Random."""
    chosen_seed = inputs["seed"] if seed is None else seed
    generator = random.Random(chosen_seed)
    expected_return = inputs["expected_return_pct"] / 100
    volatility = inputs["volatility_pct"] / 100

    sample_count = min(MAX_SAMPLE_PATHS, inputs["num_simulations"])
    if sample_count == 1:
        sample_indices = {0}
    else:
        sample_indices = {
            round(
                index * (inputs["num_simulations"] - 1) / (sample_count - 1)
            )
            for index in range(sample_count)
        }

    final_values = []
    sample_paths = []

    for simulation_index in range(inputs["num_simulations"]):
        portfolio_value = inputs["initial_investment"]
        path = [portfolio_value]

        for _ in range(inputs["years"]):
            annual_return = generator.gauss(expected_return, volatility)
            annual_return = max(-1.0, annual_return)
            portfolio_value = max(0.0, portfolio_value * (1 + annual_return))
            path.append(portfolio_value)

        final_values.append(portfolio_value)
        if simulation_index in sample_indices:
            sample_paths.append(path)

    return {
        "final_values": final_values,
        "sample_paths": sample_paths,
        "years": inputs["years"],
    }


# Command-line input:
# Parsing is kept separate from the simulation so user interaction does not affect the model's calculations or make those calculations harder to test.
# Numeric and range checks are centralized in validate_inputs; the optional seed lets users reproduce a run and compare scenarios on the same random sequence.
def prompt_for_float(label, default=None):
    """Read a floating-point value, repeating until the user enters a valid number."""
    while True:
        prompt = f"{label}"
        if default is not None:
            prompt += f" [{default}]"
        prompt += ": "
        answer = input(prompt).strip()

        if not answer and default is not None:
            return float(default)
        try:
            return float(answer)
        except ValueError:
            print("Please enter a valid number.")


def prompt_for_integer(label, default=None, allow_blank=False):
    """Read a whole number, repeating until the user enters a valid integer."""
    while True:
        prompt = f"{label}"
        if default is not None:
            prompt += f" [{default}]"
        elif allow_blank:
            prompt += " [optional]"
        prompt += ": "
        answer = input(prompt).strip()

        if not answer:
            if default is not None:
                return default
            if allow_blank:
                return None
            print("Please enter a whole number.")
            continue

        try:
            return int(answer)
        except ValueError:
            print("Please enter a whole number.")


def collect_inputs():
    """Prompt for assumptions and explain the supported input ranges."""
    print("Monte Carlo Investment Simulator — Version 1 (standard library)")
    print("Enter annual percentages as numbers, such as 5 for 5%.")
    print(
        f"Supported ranges: {MAX_YEARS} years maximum, "
        f"{MAX_SIMULATIONS:,} simulations maximum."
    )

    while True:
        initial = prompt_for_float("Initial investment (£)", 10_000)
        expected_return = prompt_for_float("Expected annual return (%)", 5)
        volatility = prompt_for_float("Annual volatility (%)", 10)
        years = prompt_for_integer("Investment horizon (years)", 5)
        simulations = prompt_for_integer("Number of simulations", 10_000)
        seed = prompt_for_integer("Random seed", allow_blank=True)

        try:
            return validate_inputs(
                initial,
                expected_return,
                volatility,
                years,
                simulations,
                seed,
            )
        except ValueError as error:
            print(f"\nInput error: {error}")
            print("Please enter the assumptions again.\n")


# Results reporting:
# The main summary describes the distribution of terminal portfolio values, including percentile cutoffs and the simulated probability of a loss.
# Monetary values are formatted as GBP for readable command-line output.
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
# Each experiment changes one assumption while holding the others fixed, isolating how return, volatility, horizon, or simulation count affects the results.
# A shared seed reduces random noise between paired scenarios; higher simulation counts improve estimate stability but do not change the model assumptions.
def run_one_sensitivity_case(base_inputs, changes, seed, simulation_count):
    """Run a baseline and one changed assumption with a shared random seed."""
    baseline_inputs = dict(base_inputs)
    baseline_inputs["num_simulations"] = simulation_count
    baseline_inputs["seed"] = seed

    changed_inputs = dict(baseline_inputs)
    changed_inputs.update(changes)

    baseline_result = simulate(baseline_inputs, seed)
    changed_result = simulate(changed_inputs, seed)
    baseline_summary = summarize_final_values(
        baseline_result["final_values"], baseline_inputs["initial_investment"]
    )
    changed_summary = summarize_final_values(
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
    baseline_inputs = dict(inputs)
    baseline_inputs["num_simulations"] = low_count
    baseline_inputs["seed"] = seed
    high_inputs = dict(inputs)
    high_inputs["num_simulations"] = high_count
    high_inputs["seed"] = seed
    low_result = simulate(baseline_inputs, seed)
    high_result = simulate(high_inputs, seed)
    low_summary = summarize_final_values(
        low_result["final_values"], inputs["initial_investment"]
    )
    high_summary = summarize_final_values(
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


# Program entry point:
# Collects and validates assumptions, selects a reproducible seed, and runs the primary simulation before computing the summary and sensitivity comparisons.
# Keeps the model available for importing and testing without prompting on import.
def main():
    inputs = collect_inputs()
    effective_seed = make_seed(inputs["seed"])
    print(f"\nUsing random seed: {effective_seed}")

    result = simulate(inputs, effective_seed)
    summary = summarize_final_values(
        result["final_values"], inputs["initial_investment"]
    )
    print_summary(summary, inputs["initial_investment"])
    print_sensitivity(inputs, effective_seed)

    print(
        "\nA simulation is not a guarantee or forecast of actual market performance."
    )


if __name__ == "__main__":
    main()
