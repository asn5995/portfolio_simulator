"""
Vanilla CLI script to run portfolio simulation.

Example usage:
    python src/run_simulation.py
"""

import sys
import os
import numpy as np

# Add parent directory to path for imports when running as script
if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

from src.economy import Economy
from src.portfolio import Portfolio
from src.simulation_engine import SimulationEngine
from src.utils import format_currency, format_percent


def main():
    """Run a Monte Carlo portfolio simulation."""
    # Initialize economy with default parameters
    economy = Economy()
    
    # Initialize portfolio with default holdings
    portfolio = Portfolio()
    
    # Create simulation engine
    engine = SimulationEngine(economy, portfolio)
    
    # Run simulation
    print("Running Monte Carlo simulation...")
    print(f"Number of trials: 10,000")
    print(f"Portfolio stocks: {', '.join(portfolio.stocks)}")
    print()
    
    results = engine.run_simulation(n_trials=10000, random_seed=42)
    
    # Compute risk metrics
    risk_metrics = engine.compute_risk_metrics(results, target_return=0.25)
    
    # Display results
    print("=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print(f"Initial value:     {format_currency(results['initial_value'])}")
    print(f"Expected value:    {format_currency(results['expected_value'])}")
    print(f"Expected return:   {format_percent(results['expected_return'])}")
    print()
    print("Risk Metrics:")
    print(f"  Probability(return >= 25%): {format_percent(risk_metrics['prob_target'])}")
    print(f"  Probability(loss):          {format_percent(risk_metrics['prob_loss'])}")
    print(f"  Probability(gain):          {format_percent(risk_metrics['prob_gain'])}")
    print()
    print("Return Distribution:")
    print(f"  Min:    {format_percent(risk_metrics['min_return'])}")
    print(f"  5th %:  {format_percent(risk_metrics['percentile_5'])}")
    print(f"  25th %: {format_percent(risk_metrics['percentile_25'])}")
    print(f"  Median: {format_percent(risk_metrics['percentile_50'])}")
    print(f"  75th %: {format_percent(risk_metrics['percentile_75'])}")
    print(f"  95th %: {format_percent(risk_metrics['percentile_95'])}")
    print(f"  Max:    {format_percent(risk_metrics['max_return'])}")
    print(f"  Std:    {format_percent(risk_metrics['std_return'])}")
    print()
    
    # State distribution
    states = results['states']
    state_counts = np.bincount(states, minlength=len(economy.STATES))
    state_probs = state_counts / len(states)
    print("Economic State Distribution:")
    for i, state_name in enumerate(economy.STATES):
        print(f"  {state_name}: {format_percent(state_probs[i])}")
    print()


if __name__ == "__main__":
    main()

