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

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.economy import Economy
from src.portfolio import Portfolio
from src.simulation_engine import SimulationEngine
from src.utils import format_currency, format_percent


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
        
        # Portfolio configuration
        st.subheader("Portfolio Holdings")
        
        # Default stocks
        default_stocks = ["AAPL", "GOOGL", "MSFT"]
        
        # Stock holdings
        holdings = {}
        prices = {}
        for stock in default_stocks:
            col1, col2 = st.columns(2)
            with col1:
                shares = st.number_input(f"{stock} Shares", 0, 1000, 100 if stock == "AAPL" else (50 if stock == "GOOGL" else 75), key=f"shares_{stock}")
            with col2:
                price = st.number_input(f"{stock} Price", 0.0, 1000.0, 150.0 if stock == "AAPL" else (120.0 if stock == "GOOGL" else 300.0), 1.0, key=f"price_{stock}")
            holdings[stock] = shares
            prices[stock] = price
        
        # Stock return parameters
        st.subheader("Stock Return Parameters")
        
        stock_params = {}
        for stock in default_stocks:
            with st.expander(f"{stock} Parameters"):
                st.write("**Awful State**")
                awful_mean = st.number_input("Mean Return", -1.0, 1.0, -0.15 if stock == "AAPL" else (-0.20 if stock == "GOOGL" else -0.12), 0.01, key=f"{stock}_awful_mean")
                awful_std = st.number_input("Std Dev", 0.0, 1.0, 0.25 if stock == "AAPL" else (0.30 if stock == "GOOGL" else 0.22), 0.01, key=f"{stock}_awful_std")
                
                st.write("**Stable State**")
                stable_mean = st.number_input("Mean Return", -1.0, 1.0, 0.08 if stock == "AAPL" else (0.10 if stock == "GOOGL" else 0.09), 0.01, key=f"{stock}_stable_mean")
                stable_std = st.number_input("Std Dev", 0.0, 1.0, 0.15 if stock == "AAPL" else (0.18 if stock == "GOOGL" else 0.16), 0.01, key=f"{stock}_stable_std")
                
                st.write("**Great State**")
                great_mean = st.number_input("Mean Return", -1.0, 1.0, 0.25 if stock == "AAPL" else (0.30 if stock == "GOOGL" else 0.22), 0.01, key=f"{stock}_great_mean")
                great_std = st.number_input("Std Dev", 0.0, 1.0, 0.20 if stock == "AAPL" else (0.22 if stock == "GOOGL" else 0.19), 0.01, key=f"{stock}_great_std")
                
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
    
    # Display results
    results = st.session_state['results']
    risk_metrics = st.session_state['risk_metrics']
    portfolio = st.session_state['portfolio']
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Initial Value", format_currency(results['initial_value']))
    with col2:
        st.metric("Expected Value", format_currency(results['expected_value']))
    with col3:
        st.metric("Expected Return", format_percent(results['expected_return']))
    with col4:
        st.metric(f"P(Return ≥ {target_return*100:.0f}%)", format_percent(risk_metrics['prob_target']))
    
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
        st.plotly_chart(fig_returns, use_container_width=True)
    
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
        st.plotly_chart(fig_values, use_container_width=True)
    
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
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("**Probability Metrics**")
        prob_data = {
            "Metric": [
                f"Probability(Return ≥ {target_return*100:.0f}%)",
                "Probability(Loss)",
                "Probability(Gain)"
            ],
            "Value": [
                format_percent(risk_metrics['prob_target']),
                format_percent(risk_metrics['prob_loss']),
                format_percent(risk_metrics['prob_gain'])
            ]
        }
        st.dataframe(pd.DataFrame(prob_data), use_container_width=True, hide_index=True)
    
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
    st.dataframe(state_df, use_container_width=True, hide_index=True)
    
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
    st.plotly_chart(fig_box, use_container_width=True)


if __name__ == "__main__":
    main()

