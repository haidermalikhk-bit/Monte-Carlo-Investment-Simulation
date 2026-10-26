Monte Carlo Investment Simulator
An educational Python project that models thousands of possible investment outcomes from a few assumptions. It includes a beginner-friendly standard-library implementation and a NumPy/matplotlib implementation with charts.

This project is for learning and demonstration. Its simulated results are not financial advice, a forecast, or a guarantee of investment performance.

Versions
Version 1: standard library
Uses Python's random, math, and other built-in modules. It demonstrates input validation, functions, loops, lists, summary statistics, and reproducible random seeds without installing third-party packages.

cd monte-carlo-investment-simulator
python3 monte_carlo_v1.py
Version 2: NumPy and matplotlib
Uses NumPy's random generator and vectorized operations to run the simulation, and matplotlib to create three charts.

cd monte-carlo-investment-simulator
python3 -m pip install -r requirements.txt
python3 monte_carlo_v2.py
The charts are saved to generated-artifacts/monte-carlo/ at the project root. That generated output directory is excluded from Git.

Assumptions and financial mathematics
Each simulated year draws a simple annual return from a normal distribution:

annual return ~ Normal(expected annual return, annual volatility)
new portfolio value = old portfolio value × (1 + annual return)
Returns are independent from year to year. Expected return and volatility remain constant for the whole investment horizon. The investment has no additional deposits or withdrawals.

A normal distribution can produce a return below -100%, which would create an impossible negative portfolio value. Both versions cap each sampled annual return at -100%, so the portfolio can reach zero but cannot go below it. This floor changes the lower tail of the normal distribution, especially for unusually high volatility.

The model also excludes taxes, fees, inflation, changing market conditions, and extreme market events. A normal distribution may not represent real market returns well, particularly in the tails. Results are illustrative outputs from simplified assumptions—not predictions or guarantees.

Understanding the results
Mean: arithmetic average final value across all simulations.
Median: the middle simulated final value; half the simulated outcomes are below it and half above.
Percentiles: for example, the 5th percentile is a value at or below which about 5% of simulated outcomes fall.
Probability below initial: the share of simulations ending below the initial investment.
Average gain or loss: mean final value minus the initial investment.
Random seed: enter a seed to reproduce the same simulation. If left blank, the program generates and prints a seed that can be reused.
The sensitivity section changes one assumption at a time. It compares expected return with a +3 percentage-point change, volatility with a +15-point change, a chosen horizon with 20 years, and 1,000 versus 10,000 simulations. When a comparison is run, its other assumptions are held constant and both sides use the same seed. Increasing the number of simulations improves the stability of estimated statistics; it does not change the underlying assumptions.

Tests
The tests use Python's built-in unittest framework. The standard-library tests run without NumPy or matplotlib; NumPy-specific tests run when Version 2's dependencies are installed.

cd monte-carlo-investment-simulator
python3 -m unittest -v
Tests cover input validation, percentile calculations, summary statistics, fixed-seed reproducibility, deterministic zero-volatility behavior, and the non-negative portfolio floor.

Resume-ready description
Project: Monte Carlo Investment Simulator — a Python educational tool that models a distribution of possible investment outcomes using configurable return, volatility, horizon, and simulation-count assumptions.

Example resume bullets:

Built standard-library and NumPy implementations of a seeded Monte Carlo investment simulator, with validated inputs and percentile-based outcome summaries.
Added matplotlib visualizations and one-variable-at-a-time sensitivity analysis; documented simplifying assumptions and tested core calculations with unittest.
nevermind i can do it myself

Free
3 minutes ago
Clarifying project status

Clarifying project status
Understood. I haven’t created or pushed a GitHub repository. The latest comment updates to monte_carlo_v2.py are still local and uncommitted; the rest of the project is in the workspace.


Worked for 11 seconds

Checkpoint made
3 minutes ago
For my project, do I need to upload only those files under "monte-carlo-investment-simulator" onto my Github?

Free
2 minutes ago

4 actions
Yes—if you want a standalone GitHub repository for your resume project, upload the contents of monte-carlo-investment-simulator/ to the repository root.

Include the Python files, README.md, requirements.txt, test_simulator.py, and results/monte_carlo_final_values.png. Skip __pycache__ folders. The workspace-level pyproject.toml and uv.lock aren’t needed for the standalone project because its requirements.txt lists its dependencies.

The latest comments in monte_carlo_v2.py are local and uncommitted; the file in the folder already contains them.


Worked for 25 seconds
Time worked
25 seconds

Scroll to latest


