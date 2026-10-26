import math
import random


MAX_YEARS = 100
MAX_SIMULATIONS = 100_000
MIN_EXPECTED_RETURN_PCT = -100.0
MAX_EXPECTED_RETURN_PCT = 1_000.0
MAX_VOLATILITY_PCT = 500.0


def validate_inputs(
    initial_investment,
    expected_return_pct,
    volatility_pct,
    years,
    num_simulations,
    seed=None,
):
    """Validate raw user inputs and return them in a consistent dictionary."""
    numeric_values = {
        "Initial investment": initial_investment,
        "Expected annual return": expected_return_pct,
        "Annual volatility": volatility_pct,
    }

    for label, value in numeric_values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be a number.")
        if not math.isfinite(value):
            raise ValueError(f"{label} must be a finite number.")

    if initial_investment < 0:
        raise ValueError("Initial investment cannot be negative.")
    if not MIN_EXPECTED_RETURN_PCT <= expected_return_pct <= MAX_EXPECTED_RETURN_PCT:
        raise ValueError(
            "Expected annual return must be between -100% and 1,000%."
        )
    if not 0 <= volatility_pct <= MAX_VOLATILITY_PCT:
        raise ValueError("Annual volatility must be between 0% and 500%.")

    if isinstance(years, bool) or not isinstance(years, int):
        raise ValueError("Investment horizon must be a whole number of years.")
    if not 1 <= years <= MAX_YEARS:
        raise ValueError(f"Investment horizon must be between 1 and {MAX_YEARS} years.")

    if isinstance(num_simulations, bool) or not isinstance(num_simulations, int):
        raise ValueError("Number of simulations must be a whole number.")
    if not 1 <= num_simulations <= MAX_SIMULATIONS:
        raise ValueError(
            f"Number of simulations must be between 1 and {MAX_SIMULATIONS:,}."
        )

    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("Random seed must be a whole number or left blank.")

    return {
        "initial_investment": float(initial_investment),
        "expected_return_pct": float(expected_return_pct),
        "volatility_pct": float(volatility_pct),
        "years": years,
        "num_simulations": num_simulations,
        "seed": seed,
    }


def make_seed(seed=None):
    """Return the supplied seed, or generate one that can be shown to the user."""
    if seed is not None:
        return seed
    return random.SystemRandom().randrange(0, 2**32)


def percentile(values, percentile_value):
    """Calculate a percentile with linear interpolation using only the standard library."""
    if not values:
        raise ValueError("Cannot calculate a percentile from an empty list.")
    if not isinstance(percentile_value, (int, float)) or not math.isfinite(
        percentile_value
    ):
        raise ValueError("Percentile must be a finite number.")
    if not 0 <= percentile_value <= 100:
        raise ValueError("Percentile must be between 0 and 100.")

    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * percentile_value / 100
    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return float(sorted_values[lower_index])

    weight = position - lower_index
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return float(lower_value + (upper_value - lower_value) * weight)


def summarize_final_values(final_values, initial_investment):
    """Calculate the requested summary statistics with standard-library tools."""
    if not final_values:
        raise ValueError("At least one final portfolio value is required.")

    values = sorted(final_values)
    count = len(values)
    mean_value = math.fsum(values) / count

    return {
        "mean": mean_value,
        "median": percentile(values, 50),
        "minimum": values[0],
        "maximum": values[-1],
        "p5": percentile(values, 5),
        "p25": percentile(values, 25),
        "p75": percentile(values, 75),
        "p95": percentile(values, 95),
        "probability_below_initial": sum(
            value < initial_investment for value in values
        )
        / count,
        "average_gain_loss": mean_value - initial_investment,
    }


def format_currency(amount):
    """Format a number as pounds sterling."""
    return f"£{amount:,.2f}"


def format_probability(probability):
    """Format a probability as a percentage."""
    return f"{probability * 100:.1f}%"
