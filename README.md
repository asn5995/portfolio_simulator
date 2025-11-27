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

**src/** → The pure simulation engine (vanilla Python). Can be used independently or imported as a library.

**streamlit_app/** → An interactive web interface allowing users to:
- Select any stocks or enter custom tickers
- Fetch real-time prices from Yahoo Finance
- Modify economic state probabilities
- Edit return parameters for each stock
- Adjust portfolio holdings
- Run simulations with instant visual feedback

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

Select any stocks from a list or enter custom tickers

Fetch real-time stock prices from Yahoo Finance

Modify economic state probabilities

Edit mean returns & volatilities in real-time

Adjust portfolio holdings

Choose number of simulations

View results instantly with charts

Robust error handling for invalid stock tickers

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

**CLI Output:**
```
Initial value:     $43,500.00
Expected value:    $46,551.37
Expected return:   7.01%
Probability(return >= 25%): 12.67%
```

**Streamlit Dashboard:**
The interactive dashboard includes:

- Key metrics cards (Initial Value, Expected Value, Expected Return, Target Probability)
- Return distribution histogram with target and expected return lines
- Portfolio value distribution histogram
- Detailed statistics tables (percentiles, probabilities)
- Economic state distribution
- Box plot visualization

All charts are interactive and update in real-time as you modify parameters.

📦 Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- pandas
- streamlit
- plotly
- yfinance (for fetching real-time stock prices)

🧪 Running the Vanilla Script

```bash
python src/run_simulation.py
```

This runs a default simulation with AAPL, GOOGL, and MSFT.

🎛 Running the Streamlit App

```bash
python -m streamlit run streamlit_app/app.py
```

Or:

```bash
cd streamlit_app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

✨ New Features

**Stock Selection:**
- Choose from popular stocks (AAPL, GOOGL, MSFT, TSLA, AMZN, NVDA, META, NFLX)
- Enter any custom ticker symbol (e.g., BRK-B, BTC-USD)
- Multi-select interface for building your portfolio

**Real-Time Price Fetching:**
- Click "🔄 Fetch Latest Prices" to get current prices from Yahoo Finance
- Prices are automatically filled in and can be manually adjusted
- Cached for 5 minutes to reduce API calls

**Error Handling:**
- Validates stock tickers before adding them
- Shows clear error messages for invalid tickers
- Prevents app crashes from invalid symbols
- Validates all stocks before running simulations

💡 Usage Example

**Using the Streamlit App:**

1. **Select Stocks:**
   - Choose from the dropdown (AAPL, GOOGL, MSFT, etc.)
   - Or enter a custom ticker like "BRK-B" or "BTC-USD"
   - Click "Add Custom Stock" (ticker will be validated)

2. **Fetch Prices:**
   - Click "🔄 Fetch Latest Prices" to get real-time prices from Yahoo Finance
   - Prices will auto-fill in the price fields
   - You can manually adjust prices if needed

3. **Configure Portfolio:**
   - Set number of shares for each stock
   - Adjust economic state probabilities (Awful, Stable, Great)
   - Configure return parameters for each stock in each state

4. **Run Simulation:**
   - Click "🚀 Run Simulation"
   - View results with interactive charts and statistics

**Using the CLI Script:**

The CLI version uses default parameters and runs a quick simulation:

```bash
python src/run_simulation.py
```

For custom portfolios, use the Streamlit app or modify the code in `src/run_simulation.py`.

🧪 Testing

Run the test suite:

```bash
python -m pytest tests/
```

Or:

```bash
python -m unittest tests.test_simulation
```
