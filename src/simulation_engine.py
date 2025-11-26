"""
Monte Carlo simulation engine (vectorized).

Core logic for running portfolio simulations across thousands of scenarios.
"""

from typing import Dict, Tuple
import numpy as np
from .economy import Economy
from .portfolio import Portfolio


class SimulationEngine:
    """
    Monte Carlo simulation engine for portfolio returns.
    """
    
    def __init__(self, economy: Economy, portfolio: Portfolio):
        """
        Initialize simulation engine.
        
        Args:
            economy: Economy instance with state probabilities and stock parameters
            portfolio: Portfolio instance with holdings and prices
        """
        self.economy = economy
        self.portfolio = portfolio
        
        # Ensure portfolio stocks match economy stocks
        portfolio_stocks = set(portfolio.stocks)
        economy_stocks = set(economy.stock_names)
        if not portfolio_stocks.issubset(economy_stocks):
            missing = portfolio_stocks - economy_stocks
            raise ValueError(f"Portfolio contains stocks not in economy: {missing}")
    
    def run_simulation(
        self,
        n_trials: int = 10000,
        random_seed: int = None
    ) -> Dict:
        """
        Run Monte Carlo simulation.
        
        Args:
            n_trials: Number of simulation trials
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with simulation results:
            - initial_value: Starting portfolio value
            - final_values: Array of final portfolio values (n_trials,)
            - returns: Array of portfolio returns (n_trials,)
            - expected_value: Expected final portfolio value
            - expected_return: Expected portfolio return
            - states: Array of sampled states (n_trials,)
            - stock_returns: Array of stock returns (n_trials, n_stocks)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Get initial portfolio value
        initial_value = self.portfolio.get_current_value()
        
        # Get stock order for consistent indexing
        stock_order = self.portfolio.get_stock_order()
        n_stocks = len(stock_order)
        
        # Sample economic states for all trials
        states = self.economy.sample_state(n_trials)
        
        # Generate stock returns for all trials (vectorized)
        stock_returns = np.zeros((n_trials, n_stocks))
        
        for i, stock in enumerate(stock_order):
            for state_idx in range(len(self.economy.STATES)):
                # Find trials with this state
                mask = states == state_idx
                n_in_state = np.sum(mask)
                
                if n_in_state > 0:
                    # Get parameters for this stock in this state
                    mean, std = self.economy.get_stock_params_by_idx(stock, state_idx)
                    # Generate returns for trials in this state
                    stock_returns[mask, i] = np.random.normal(mean, std, n_in_state)
        
        # Calculate portfolio values for all scenarios (vectorized)
        final_values = self.portfolio.get_future_value_vectorized(stock_returns)
        
        # Calculate portfolio returns
        returns = (final_values - initial_value) / initial_value
        
        # Compute summary statistics
        expected_value = np.mean(final_values)
        expected_return = np.mean(returns)
        
        return {
            "initial_value": initial_value,
            "final_values": final_values,
            "returns": returns,
            "expected_value": expected_value,
            "expected_return": expected_return,
            "states": states,
            "stock_returns": stock_returns,
            "stock_order": stock_order
        }
    
    def compute_risk_metrics(
        self,
        results: Dict,
        target_return: float = 0.25
    ) -> Dict:
        """
        Compute additional risk metrics from simulation results.
        
        Args:
            results: Results dictionary from run_simulation()
            target_return: Target return threshold (e.g., 0.25 for 25%)
            
        Returns:
            Dictionary with risk metrics:
            - prob_target: Probability of achieving target return
            - prob_loss: Probability of negative return
            - prob_gain: Probability of positive return
            - percentile_5: 5th percentile return
            - percentile_25: 25th percentile return
            - percentile_50: Median return
            - percentile_75: 75th percentile return
            - percentile_95: 95th percentile return
            - std_return: Standard deviation of returns
            - min_return: Minimum return
            - max_return: Maximum return
        """
        returns = results["returns"]
        
        prob_target = np.mean(returns >= target_return)
        prob_loss = np.mean(returns < 0)
        prob_gain = np.mean(returns > 0)
        
        percentiles = np.percentile(returns, [5, 25, 50, 75, 95])
        
        return {
            "prob_target": prob_target,
            "prob_loss": prob_loss,
            "prob_gain": prob_gain,
            "percentile_5": percentiles[0],
            "percentile_25": percentiles[1],
            "percentile_50": percentiles[2],
            "percentile_75": percentiles[3],
            "percentile_95": percentiles[4],
            "std_return": np.std(returns),
            "min_return": np.min(returns),
            "max_return": np.max(returns)
        }

