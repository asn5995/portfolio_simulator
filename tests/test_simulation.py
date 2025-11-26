"""
Unit tests for Monte Carlo portfolio simulation.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.economy import Economy
from src.portfolio import Portfolio
from src.simulation_engine import SimulationEngine


class TestEconomy(unittest.TestCase):
    """Test Economy class."""
    
    def test_default_initialization(self):
        """Test default economy initialization."""
        economy = Economy()
        self.assertEqual(len(economy.STATES), 3)
        self.assertTrue(np.isclose(sum(economy.state_probabilities), 1.0))
        self.assertGreater(len(economy.stock_names), 0)
    
    def test_custom_initialization(self):
        """Test custom economy initialization."""
        probs = [0.3, 0.5, 0.2]
        stock_returns = {
            "TEST": {
                "Awful": (-0.1, 0.2),
                "Stable": (0.05, 0.15),
                "Great": (0.2, 0.18)
            }
        }
        economy = Economy(state_probabilities=probs, stock_returns=stock_returns)
        np.testing.assert_array_almost_equal(economy.state_probabilities, probs)
        self.assertEqual(economy.stock_names, ["TEST"])
    
    def test_invalid_probabilities(self):
        """Test that invalid probabilities raise errors."""
        with self.assertRaises(ValueError):
            Economy(state_probabilities=[0.5, 0.5])  # Wrong length
        
        with self.assertRaises(ValueError):
            Economy(state_probabilities=[0.5, 0.5, 0.5])  # Doesn't sum to 1
    
    def test_sample_state(self):
        """Test state sampling."""
        economy = Economy()
        states = economy.sample_state(1000)
        self.assertEqual(len(states), 1000)
        self.assertTrue(all(0 <= s < len(economy.STATES) for s in states))
    
    def test_get_stock_params(self):
        """Test getting stock parameters."""
        economy = Economy()
        mean, std = economy.get_stock_params("AAPL", "Stable")
        self.assertIsInstance(mean, (int, float))
        self.assertIsInstance(std, (int, float))
        self.assertGreater(std, 0)


class TestPortfolio(unittest.TestCase):
    """Test Portfolio class."""
    
    def test_default_initialization(self):
        """Test default portfolio initialization."""
        portfolio = Portfolio()
        self.assertGreater(len(portfolio.stocks), 0)
        self.assertGreater(portfolio.get_current_value(), 0)
    
    def test_custom_initialization(self):
        """Test custom portfolio initialization."""
        holdings = {"AAPL": 50, "GOOGL": 25}
        prices = {"AAPL": 150.0, "GOOGL": 120.0}
        portfolio = Portfolio(holdings=holdings, prices=prices)
        self.assertEqual(portfolio.get_current_value(), 50 * 150.0 + 25 * 120.0)
    
    def test_get_future_value(self):
        """Test future value calculation."""
        portfolio = Portfolio()
        returns = {stock: 0.1 for stock in portfolio.stocks}
        future_value = portfolio.get_future_value(returns)
        current_value = portfolio.get_current_value()
        self.assertAlmostEqual(future_value, current_value * 1.1, places=2)
    
    def test_get_future_value_vectorized(self):
        """Test vectorized future value calculation."""
        portfolio = Portfolio()
        n_scenarios = 100
        n_stocks = len(portfolio.stocks)
        returns_matrix = np.random.normal(0.05, 0.1, (n_scenarios, n_stocks))
        future_values = portfolio.get_future_value_vectorized(returns_matrix)
        self.assertEqual(len(future_values), n_scenarios)
        self.assertTrue(all(fv > 0 for fv in future_values))


class TestSimulationEngine(unittest.TestCase):
    """Test SimulationEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.economy = Economy()
        self.portfolio = Portfolio()
        self.engine = SimulationEngine(self.economy, self.portfolio)
    
    def test_initialization(self):
        """Test engine initialization."""
        self.assertIsNotNone(self.engine.economy)
        self.assertIsNotNone(self.engine.portfolio)
    
    def test_run_simulation(self):
        """Test running a simulation."""
        results = self.engine.run_simulation(n_trials=1000, random_seed=42)
        
        self.assertIn("initial_value", results)
        self.assertIn("final_values", results)
        self.assertIn("returns", results)
        self.assertIn("expected_value", results)
        self.assertIn("expected_return", results)
        
        self.assertEqual(len(results["final_values"]), 1000)
        self.assertEqual(len(results["returns"]), 1000)
        self.assertGreater(results["expected_value"], 0)
    
    def test_compute_risk_metrics(self):
        """Test risk metrics computation."""
        results = self.engine.run_simulation(n_trials=1000, random_seed=42)
        risk_metrics = self.engine.compute_risk_metrics(results, target_return=0.25)
        
        self.assertIn("prob_target", risk_metrics)
        self.assertIn("prob_loss", risk_metrics)
        self.assertIn("prob_gain", risk_metrics)
        self.assertIn("percentile_5", risk_metrics)
        self.assertIn("percentile_50", risk_metrics)
        self.assertIn("percentile_95", risk_metrics)
        
        # Check probabilities are between 0 and 1
        self.assertGreaterEqual(risk_metrics["prob_target"], 0)
        self.assertLessEqual(risk_metrics["prob_target"], 1)
        self.assertGreaterEqual(risk_metrics["prob_loss"], 0)
        self.assertLessEqual(risk_metrics["prob_loss"], 1)
    
    def test_simulation_reproducibility(self):
        """Test that simulations are reproducible with same seed."""
        results1 = self.engine.run_simulation(n_trials=100, random_seed=42)
        results2 = self.engine.run_simulation(n_trials=100, random_seed=42)
        
        np.testing.assert_array_equal(results1["final_values"], results2["final_values"])
        np.testing.assert_array_equal(results1["returns"], results2["returns"])


if __name__ == "__main__":
    unittest.main()

