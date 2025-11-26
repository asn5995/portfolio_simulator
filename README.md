📈 Monte Carlo Portfolio Simulator

A modular Python engine for simulating portfolio returns under uncertain economic scenarios.

🔍 Overview

This project implements a Monte Carlo–based portfolio simulation framework where:

The economy can be in one of several discrete states (Awful, Stable, Great).

Each state has its own probabilities, expected returns, and return volatilities for each stock.

Given the state, each stock’s return is normally distributed and independent across assets.

A portfolio value is simulated over 1 year across thousands of scenarios.

Key metrics such as expected return, final portfolio distribution, and tail-risk probabilities are computed.

This project has two layers:

src/ → The pure simulation engine (vanilla Python).

streamlit_app/ → An interactive interface allowing users to modify probabilities, returns, and run simulations dynamically.

📁 Project Structure
monte-carlo-portfolio-simulator/
│
├── src/
│   ├── simulation_engine.py   # Monte Carlo logic (vectorized)
│   ├── portfolio.py           # Portfolio representation & valuation
│   ├── economy.py             # State distributions and parameters
│   ├── run_simulation.py      # Vanilla script (CLI version)
│   └── utils.py               # Common helpers
│
├── streamlit_app/
│   └── app.py                 # Interactive Streamlit dashboard
│
├── tests/
│   └── test_simulation.py     # Unit tests for core logic
│
├── README.md
└── requirements.txt

🚀 Features
✔ Monte Carlo Simulation

User-defined number of trials

State-dependent mean/variance of stock returns

Independent draws conditional on economy state

Portfolio-level value and return computation

✔ Risk Metrics

Expected final portfolio value

Expected portfolio return

Probability of hitting a target (e.g., ≥ 25%)

Histogram of outcomes

✔ Streamlit UI (Dynamic Version)

Modify economic state probabilities

Edit mean returns & volatilities in real-time

Adjust portfolio holdings

Choose number of simulations

View results instantly with charts

🧠 How It Works
1. Sample the economic state

According to the probability vector across:
Awful • Stable • Great

2. Generate stock returns

Using state-specific Normal(mean, std) distributions.

3. Compute next-year portfolio value

New prices = current prices × (1 + simulated returns)

Portfolio value = Σ(shares × new prices)

4. Repeat thousands of times

to build a full distribution of outcomes.

📊 Example Output
Initial value: $103,999.00  
Expected value: $113,820.47  
Expected return: 9.45%  
Probability(return ≥ 25%): 18.72%


Charts include:

Return histogram

Value distribution

Scenario comparison

🧪 Running the Vanilla Script
python src/run_simulation.py

🎛 Running the Streamlit App
cd streamlit_app
streamlit run app.py
