"""
Economy state distributions and parameters.

Defines the economic states (Awful, Stable, Great) and their associated
probabilities, expected returns, and volatilities for each stock.
"""

from typing import Dict, List, Tuple
import numpy as np


class Economy:
    """
    Represents the economic state space and parameters.
    
    States: Awful, Stable, Great
    Each state has probabilities and stock-specific return parameters.
    """
    
    STATES = ["Awful", "Stable", "Great"]
    
    def __init__(
        self,
        state_probabilities: List[float] = None,
        stock_returns: Dict[str, Dict[str, Tuple[float, float]]] = None
    ):
        """
        Initialize economy with state probabilities and stock return parameters.
        
        Args:
            state_probabilities: Probability of each state [Awful, Stable, Great].
                                 Defaults to [0.2, 0.6, 0.2]
            stock_returns: Dict mapping stock names to state-specific (mean, std).
                          Format: {stock_name: {state: (mean, std), ...}, ...}
                          Defaults to example stocks if not provided.
        """
        if state_probabilities is None:
            state_probabilities = [0.2, 0.6, 0.2]
        
        if len(state_probabilities) != len(self.STATES):
            raise ValueError(f"Must provide probabilities for all {len(self.STATES)} states")
        
        if not np.isclose(sum(state_probabilities), 1.0):
            raise ValueError("State probabilities must sum to 1.0")
        
        self.state_probabilities = np.array(state_probabilities)
        
        # Default stock returns if not provided
        if stock_returns is None:
            stock_returns = {
                "AAPL": {
                    "Awful": (-0.15, 0.25),
                    "Stable": (0.08, 0.15),
                    "Great": (0.25, 0.20)
                },
                "GOOGL": {
                    "Awful": (-0.20, 0.30),
                    "Stable": (0.10, 0.18),
                    "Great": (0.30, 0.22)
                },
                "MSFT": {
                    "Awful": (-0.12, 0.22),
                    "Stable": (0.09, 0.16),
                    "Great": (0.22, 0.19)
                }
            }
        
        self.stock_returns = stock_returns
        self.stock_names = list(stock_returns.keys())
    
    def sample_state(self, n_samples: int = 1) -> np.ndarray:
        """
        Sample economic states according to probabilities.
        
        Args:
            n_samples: Number of samples to draw
            
        Returns:
            Array of state indices (0=Awful, 1=Stable, 2=Great)
        """
        return np.random.choice(len(self.STATES), size=n_samples, p=self.state_probabilities)
    
    def get_state_name(self, state_idx: int) -> str:
        """Get state name from index."""
        return self.STATES[state_idx]
    
    def get_stock_params(self, stock_name: str, state: str) -> Tuple[float, float]:
        """
        Get (mean, std) return parameters for a stock in a given state.
        
        Args:
            stock_name: Name of the stock
            state: State name ("Awful", "Stable", or "Great")
            
        Returns:
            Tuple of (mean_return, std_return)
        """
        if stock_name not in self.stock_returns:
            raise ValueError(f"Stock {stock_name} not found")
        if state not in self.stock_returns[stock_name]:
            raise ValueError(f"State {state} not found for stock {stock_name}")
        
        return self.stock_returns[stock_name][state]
    
    def get_stock_params_by_idx(self, stock_name: str, state_idx: int) -> Tuple[float, float]:
        """Get stock parameters using state index instead of name."""
        state_name = self.get_state_name(state_idx)
        return self.get_stock_params(stock_name, state_name)
    
    def update_state_probabilities(self, probabilities: List[float]):
        """Update state probabilities."""
        if len(probabilities) != len(self.STATES):
            raise ValueError(f"Must provide probabilities for all {len(self.STATES)} states")
        if not np.isclose(sum(probabilities), 1.0):
            raise ValueError("State probabilities must sum to 1.0")
        self.state_probabilities = np.array(probabilities)
    
    def update_stock_params(self, stock_name: str, state: str, mean: float, std: float):
        """Update return parameters for a stock in a given state."""
        if stock_name not in self.stock_returns:
            self.stock_returns[stock_name] = {}
            if stock_name not in self.stock_names:
                self.stock_names.append(stock_name)
        self.stock_returns[stock_name][state] = (mean, std)

