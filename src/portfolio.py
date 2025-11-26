"""
Portfolio representation and valuation.

Handles portfolio holdings, current prices, and value calculations.
"""

from typing import Dict, List
import numpy as np


class Portfolio:
    """
    Represents a portfolio of stocks with holdings and prices.
    """
    
    def __init__(self, holdings: Dict[str, int] = None, prices: Dict[str, float] = None):
        """
        Initialize portfolio with holdings and current prices.
        
        Args:
            holdings: Dict mapping stock names to number of shares.
                     Defaults to example portfolio if not provided.
            prices: Dict mapping stock names to current prices.
                   Defaults to example prices if not provided.
        """
        if holdings is None:
            holdings = {
                "AAPL": 100,
                "GOOGL": 50,
                "MSFT": 75
            }
        
        if prices is None:
            prices = {
                "AAPL": 150.0,
                "GOOGL": 120.0,
                "MSFT": 300.0
            }
        
        # Ensure all holdings have prices
        for stock in holdings:
            if stock not in prices:
                raise ValueError(f"No price provided for {stock}")
        
        self.holdings = holdings.copy()
        self.prices = prices.copy()
        self.stocks = list(holdings.keys())
    
    def get_current_value(self) -> float:
        """
        Calculate current portfolio value.
        
        Returns:
            Total portfolio value (sum of shares * prices)
        """
        return sum(self.holdings[stock] * self.prices[stock] for stock in self.stocks)
    
    def get_future_value(self, returns: Dict[str, float]) -> float:
        """
        Calculate portfolio value after applying returns.
        
        Args:
            returns: Dict mapping stock names to return percentages (e.g., 0.08 for 8%)
            
        Returns:
            Future portfolio value
        """
        total_value = 0.0
        for stock in self.stocks:
            if stock not in returns:
                raise ValueError(f"No return provided for {stock}")
            new_price = self.prices[stock] * (1 + returns[stock])
            total_value += self.holdings[stock] * new_price
        return total_value
    
    def get_future_value_vectorized(self, returns_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate portfolio values for multiple scenarios (vectorized).
        
        Args:
            returns_matrix: Array of shape (n_scenarios, n_stocks) with returns
            
        Returns:
            Array of portfolio values for each scenario
        """
        # Create price vector and holdings vector in same order as self.stocks
        price_vector = np.array([self.prices[stock] for stock in self.stocks])
        holdings_vector = np.array([self.holdings[stock] for stock in self.stocks])
        
        # Calculate new prices: (n_scenarios, n_stocks) * (n_stocks,) -> (n_scenarios, n_stocks)
        new_prices = price_vector * (1 + returns_matrix)
        
        # Calculate portfolio value: sum over stocks axis
        portfolio_values = np.sum(new_prices * holdings_vector, axis=1)
        
        return portfolio_values
    
    def update_holdings(self, stock: str, shares: int):
        """Update number of shares for a stock."""
        if shares < 0:
            raise ValueError("Holdings cannot be negative")
        self.holdings[stock] = shares
        if stock not in self.stocks:
            self.stocks.append(stock)
            if stock not in self.prices:
                raise ValueError(f"No price set for {stock}")
    
    def update_price(self, stock: str, price: float):
        """Update current price for a stock."""
        if price <= 0:
            raise ValueError("Price must be positive")
        self.prices[stock] = price
        if stock not in self.stocks:
            self.stocks.append(stock)
            if stock not in self.holdings:
                self.holdings[stock] = 0
    
    def get_stock_order(self) -> List[str]:
        """Get ordered list of stock names (for consistent indexing)."""
        return self.stocks.copy()

