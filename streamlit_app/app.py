"""
Interactive Streamlit dashboard for portfolio simulation.

Run with: streamlit run streamlit_app/app.py
"""

import sys
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.economy import Economy
from src.portfolio import Portfolio
from src.simulation_engine import SimulationEngine
from src.utils import format_currency, format_percent


def validate_ticker(ticker):
    """
    Validate if a ticker exists on Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Tuple of (is_valid: bool, error_message: str or None)
    """
    if not ticker or not ticker.strip():
        return False, "Ticker cannot be empty"
    
    ticker = ticker.strip().upper()
    
    try:
        stock = yf.Ticker(ticker)
        # Try to get info to validate the ticker exists
        info = stock.info
        
        # Check if ticker is valid - invalid tickers often return empty info or have 'symbol' mismatch
        if not info or len(info) == 0:
            return False, f"'{ticker}' not found on Yahoo Finance"
        
        # Check if the symbol matches (sometimes Yahoo Finance returns different symbol)
        if 'symbol' in info and info['symbol'].upper() != ticker:
            # Sometimes tickers are valid but have different casing or format
            pass
        
        # Try to get price data
        hist = stock.history(period="5d")
        if hist.empty:
            # Check if we can get price from info
            if 'currentPrice' not in info and 'regularMarketPrice' not in info:
                return False, f"'{ticker}' found but no price data available"
        
        return True, None
    except Exception as e:
        error_msg = str(e).lower()
        if 'not found' in error_msg or 'invalid' in error_msg or 'no data' in error_msg:
            return False, f"'{ticker}' not found on Yahoo Finance"
        else:
            return False, f"Error validating '{ticker}': {str(e)}"


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_prices(tickers):
    """
    Fetch latest stock prices from Yahoo Finance.
    
    Args:
        tickers: List of stock ticker symbols
        
    Returns:
        Dictionary mapping ticker to price, and any errors
    """
    prices = {}
    errors = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            # Get latest price (last close price)
            hist = stock.history(period="1d")
            if not hist.empty:
                prices[ticker] = float(hist['Close'].iloc[-1])
            else:
                # Fallback to info if history is empty
                try:
                    info = stock.info
                    if info and len(info) > 0:
                        if 'currentPrice' in info and info['currentPrice'] is not None:
                            prices[ticker] = float(info['currentPrice'])
                        elif 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                            prices[ticker] = float(info['regularMarketPrice'])
                        else:
                            errors.append(f"'{ticker}': Could not fetch price - no price data available")
                    else:
                        errors.append(f"'{ticker}': Ticker not found on Yahoo Finance")
                except Exception as info_error:
                    errors.append(f"'{ticker}': Could not fetch price - {str(info_error)}")
        except Exception as e:
            error_msg = str(e).lower()
            if 'not found' in error_msg or 'invalid' in error_msg:
                errors.append(f"'{ticker}': Ticker not found on Yahoo Finance")
            else:
                errors.append(f"'{ticker}': Error fetching price - {str(e)}")
    
    return prices, errors


def get_default_stock_params(stock):
    """Get default return parameters for a stock."""
    # Default parameters (can be customized per stock)
    defaults = {
        "Awful": (-0.15, 0.25),
        "Stable": (0.08, 0.15),
        "Great": (0.25, 0.20)
    }
    return defaults


def main():
    st.set_page_config(page_title="Monte Carlo Portfolio Simulator", layout="wide")
    
    st.title("📈 Monte Carlo Portfolio Simulator")
    st.markdown("Simulate portfolio returns under uncertain economic scenarios")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Number of simulations
        n_trials = st.slider("Number of Simulations", 1000, 50000, 10000, 1000)
        
        # Economic state probabilities
        st.subheader("Economic State Probabilities")
        prob_awful = st.slider("Awful", 0.0, 1.0, 0.2, 0.01)
        prob_stable = st.slider("Stable", 0.0, 1.0, 0.6, 0.01)
        prob_great = st.slider("Great", 0.0, 1.0, 0.2, 0.01)
        
        # Normalize probabilities
        total_prob = prob_awful + prob_stable + prob_great
        if total_prob > 0:
            prob_awful /= total_prob
            prob_stable /= total_prob
            prob_great /= total_prob
        else:
            prob_awful = prob_stable = prob_great = 1/3
        
        st.info(f"Total: {total_prob:.2f} (auto-normalized)")
        
        # Stock selection
        st.subheader("Stock Selection")
        default_stocks = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX"]
        
        # Initialize session state for selected stocks
        if 'selected_stocks' not in st.session_state:
            st.session_state.selected_stocks = ["AAPL", "GOOGL", "MSFT"]
        
        # Multi-select for stocks
        selected_stocks = st.multiselect(
            "Select Stocks",
            options=default_stocks,
            default=st.session_state.selected_stocks,
            help="Select one or more stocks to include in your portfolio"
        )
        
        # Allow custom stock entry
        custom_stock = st.text_input("Or enter custom ticker (e.g., BRK-B, BTC-USD)", "")
        if custom_stock and custom_stock.upper() not in selected_stocks:
            if st.button("Add Custom Stock"):
                ticker = custom_stock.strip().upper()
                # Validate ticker before adding
                is_valid, error_msg = validate_ticker(ticker)
                if is_valid:
                    if ticker not in selected_stocks:
                        selected_stocks.append(ticker)
                        st.session_state.selected_stocks = selected_stocks
                        st.success(f"✅ Added {ticker}")
                        st.rerun()
                    else:
                        st.warning(f"'{ticker}' is already in your selection")
                else:
                    st.error(f"❌ {error_msg}")
                    st.info("💡 Please check the ticker symbol and try again. Make sure it exists on Yahoo Finance.")
        
        if not selected_stocks:
            st.warning("⚠️ Please select at least one stock")
            st.stop()
        
        # Update session state
        st.session_state.selected_stocks = selected_stocks
        
        # Portfolio configuration
        st.subheader("Portfolio Holdings")
        
        # Fetch prices from Yahoo Finance
        if st.button("🔄 Fetch Latest Prices", help="Get current prices from Yahoo Finance"):
            with st.spinner("Fetching prices..."):
                fetched_prices, errors = fetch_stock_prices(selected_stocks)
                if fetched_prices:
                    st.session_state.fetched_prices = fetched_prices
                    st.success(f"✅ Fetched prices for {len(fetched_prices)} stock(s)")
                if errors:
                    st.error("❌ Error fetching prices:")
                    for error in errors:
                        st.error(f"  • {error}")
                    # Remove invalid tickers from selection
                    invalid_tickers = [err.split("'")[1] for err in errors if "'" in err]
                    if invalid_tickers:
                        st.warning(f"⚠️ Consider removing invalid tickers: {', '.join(invalid_tickers)}")
        
        # Initialize fetched prices in session state if not exists
        if 'fetched_prices' not in st.session_state:
            st.session_state.fetched_prices = {}
        
        # Stock holdings and prices
        holdings = {}
        prices = {}
        for stock in selected_stocks:
            col1, col2 = st.columns(2)
            with col1:
                shares = st.number_input(
                    f"{stock} Shares", 
                    0, 10000, 
                    100 if stock == "AAPL" else (50 if stock == "GOOGL" else 75), 
                    key=f"shares_{stock}"
                )
            with col2:
                # Use fetched price if available, otherwise allow manual entry
                default_price = st.session_state.fetched_prices.get(stock, 150.0)
                price = st.number_input(
                    f"{stock} Price", 
                    0.0, 10000.0, 
                    float(default_price), 
                    0.01, 
                    key=f"price_{stock}",
                    help="Price will be auto-filled when you fetch latest prices"
                )
            holdings[stock] = shares
            prices[stock] = price
        
        # Show fetched prices if available
        if st.session_state.fetched_prices:
            st.info("💡 Prices fetched from Yahoo Finance. Adjust manually if needed.")
        
        # Stock return parameters
        st.subheader("Stock Return Parameters")
        
        stock_params = {}
        for stock in selected_stocks:
            default_params = get_default_stock_params(stock)
            with st.expander(f"{stock} Parameters"):
                st.write("**Awful State**")
                awful_mean = st.number_input(
                    "Mean Return", -1.0, 1.0, default_params["Awful"][0], 0.01, 
                    key=f"{stock}_awful_mean"
                )
                awful_std = st.number_input(
                    "Std Dev", 0.0, 1.0, default_params["Awful"][1], 0.01, 
                    key=f"{stock}_awful_std"
                )
                
                st.write("**Stable State**")
                stable_mean = st.number_input(
                    "Mean Return", -1.0, 1.0, default_params["Stable"][0], 0.01, 
                    key=f"{stock}_stable_mean"
                )
                stable_std = st.number_input(
                    "Std Dev", 0.0, 1.0, default_params["Stable"][1], 0.01, 
                    key=f"{stock}_stable_std"
                )
                
                st.write("**Great State**")
                great_mean = st.number_input(
                    "Mean Return", -1.0, 1.0, default_params["Great"][0], 0.01, 
                    key=f"{stock}_great_mean"
                )
                great_std = st.number_input(
                    "Std Dev", 0.0, 1.0, default_params["Great"][1], 0.01, 
                    key=f"{stock}_great_std"
                )
                
                stock_params[stock] = {
                    "Awful": (awful_mean, awful_std),
                    "Stable": (stable_mean, stable_std),
                    "Great": (great_mean, great_std)
                }
        
        # Target return
        target_return = st.slider("Target Return (%)", 0.0, 100.0, 25.0, 1.0) / 100.0
        
        # Run simulation button
        run_sim = st.button("🚀 Run Simulation", type="primary", use_container_width=True)
    
    # Main content area
    if run_sim or 'results' not in st.session_state:
        if not selected_stocks:
            st.warning("Please select at least one stock to run the simulation.")
            st.stop()
        
        # Validate all stocks before running simulation
        invalid_stocks = []
        for stock in selected_stocks:
            is_valid, error_msg = validate_ticker(stock)
            if not is_valid:
                invalid_stocks.append((stock, error_msg))
        
        if invalid_stocks:
            st.error("❌ Cannot run simulation with invalid stock tickers:")
            for stock, error_msg in invalid_stocks:
                st.error(f"  • {stock}: {error_msg}")
            st.warning("⚠️ Please remove invalid tickers or use valid stock symbols from Yahoo Finance.")
            st.stop()
        
        try:
            # Initialize economy and portfolio
            economy = Economy(
                state_probabilities=[prob_awful, prob_stable, prob_great],
                stock_returns=stock_params
            )
            
            portfolio = Portfolio(holdings=holdings, prices=prices)
            
            # Create engine and run simulation
            engine = SimulationEngine(economy, portfolio)
            
            with st.spinner("Running simulation..."):
                results = engine.run_simulation(n_trials=n_trials, random_seed=42)
                risk_metrics = engine.compute_risk_metrics(results, target_return=target_return)
            
            # Store in session state
            st.session_state['results'] = results
            st.session_state['risk_metrics'] = risk_metrics
            st.session_state['economy'] = economy
            st.session_state['portfolio'] = portfolio
            st.session_state['target_return'] = target_return
        except Exception as e:
            st.error(f"❌ Error running simulation: {str(e)}")
            st.info("💡 Please check your stock selections and parameters, then try again.")
            st.stop()
    
    # Display results
    results = st.session_state['results']
    risk_metrics = st.session_state['risk_metrics']
    portfolio = st.session_state['portfolio']
    economy = st.session_state['economy']
    target_return = st.session_state.get('target_return', 0.25)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Initial Value", format_currency(results['initial_value']))
    with col2:
        st.metric("Expected Value", format_currency(results['expected_value']))
    with col3:
        st.metric("Expected Return", format_percent(results['expected_return']))
    with col4:
        st.metric(f"P(Return >= {target_return*100:.0f}%)", format_percent(risk_metrics['prob_target']))
    
    # Charts
    st.subheader("📊 Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Return histogram
        fig_returns = px.histogram(
            x=results['returns'] * 100,
            nbins=50,
            title="Portfolio Return Distribution",
            labels={"x": "Return (%)", "y": "Frequency"},
            color_discrete_sequence=['#1f77b4']
        )
        fig_returns.add_vline(
            x=target_return * 100,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Target: {target_return*100:.0f}%"
        )
        fig_returns.add_vline(
            x=results['expected_return'] * 100,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Expected: {results['expected_return']*100:.2f}%"
        )
        st.plotly_chart(fig_returns, width='stretch')
    
    with col2:
        # Value distribution
        fig_values = px.histogram(
            x=results['final_values'],
            nbins=50,
            title="Final Portfolio Value Distribution",
            labels={"x": "Portfolio Value ($)", "y": "Frequency"},
            color_discrete_sequence=['#2ca02c']
        )
        fig_values.add_vline(
            x=results['initial_value'],
            line_dash="dash",
            line_color="blue",
            annotation_text=f"Initial: {format_currency(results['initial_value'])}"
        )
        fig_values.add_vline(
            x=results['expected_value'],
            line_dash="dash",
            line_color="green",
            annotation_text=f"Expected: {format_currency(results['expected_value'])}"
        )
        st.plotly_chart(fig_values, width='stretch')
    
    # Detailed statistics
    st.subheader("📈 Detailed Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Return Statistics**")
        stats_data = {
            "Metric": [
                "Minimum Return",
                "5th Percentile",
                "25th Percentile",
                "Median Return",
                "75th Percentile",
                "95th Percentile",
                "Maximum Return",
                "Standard Deviation"
            ],
            "Value": [
                format_percent(risk_metrics['min_return']),
                format_percent(risk_metrics['percentile_5']),
                format_percent(risk_metrics['percentile_25']),
                format_percent(risk_metrics['percentile_50']),
                format_percent(risk_metrics['percentile_75']),
                format_percent(risk_metrics['percentile_95']),
                format_percent(risk_metrics['max_return']),
                format_percent(risk_metrics['std_return'])
            ]
        }
        st.dataframe(pd.DataFrame(stats_data), width='stretch', hide_index=True)
    
    with col2:
        st.markdown("**Probability Metrics**")
        prob_data = {
            "Metric": [
                f"Probability(Return >= {target_return*100:.0f}%)",
                "Probability(Loss)",
                "Probability(Gain)"
            ],
            "Value": [
                format_percent(risk_metrics['prob_target']),
                format_percent(risk_metrics['prob_loss']),
                format_percent(risk_metrics['prob_gain'])
            ]
        }
        st.dataframe(pd.DataFrame(prob_data), width='stretch', hide_index=True)
    
    # Economic state distribution
    st.subheader("🌍 Economic State Distribution")
    states = results['states']
    state_counts = np.bincount(states, minlength=len(economy.STATES))
    state_probs = state_counts / len(states)
    
    state_df = pd.DataFrame({
        "State": economy.STATES,
        "Count": state_counts,
        "Probability": [format_percent(p) for p in state_probs]
    })
    st.dataframe(state_df, width='stretch', hide_index=True)
    
    # Box plot comparison
    st.subheader("📦 Return Distribution Box Plot")
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(
        y=results['returns'] * 100,
        name="Portfolio Returns",
        boxmean='sd'
    ))
    fig_box.update_layout(
        title="Portfolio Return Distribution (Box Plot)",
        yaxis_title="Return (%)",
        showlegend=False
    )
    st.plotly_chart(fig_box, width='stretch')


if __name__ == "__main__":
    main()
