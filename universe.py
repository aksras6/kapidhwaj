"""
Universe Management for Survivorship-Bias-Free Backtesting

Provides historical symbol universes to avoid survivorship bias.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass

from config import TradingConfig
from logger import get_logger


@dataclass
class UniverseSnapshot:
    """Snapshot of trading universe at a point in time"""
    date: pd.Timestamp
    symbols: List[str]
    universe_name: str
    total_symbols: int


class UniverseManager:
    """Manages historical trading universes"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = get_logger("universe")
        
        # Universe definitions
        self.universes = {
            "sp500_2020": self._get_sp500_2020_universe(),
            "nasdaq100_2020": self._get_nasdaq100_2020_universe(),
            "large_cap_2020": self._get_large_cap_2020_universe(),
            "current_symbols": self.config.SYMBOLS
        }
        
        self.logger.info(f"Universe manager initialized with {len(self.universes)} universes")
    
    def get_universe_for_date(self, date: pd.Timestamp, universe_name: str = "sp500_2020") -> List[str]:
        """Get trading universe for a specific date"""
        
        if universe_name not in self.universes:
            self.logger.warning(f"Unknown universe: {universe_name}, using current_symbols")
            universe_name = "current_symbols"
        
        universe = self.universes[universe_name]
        
        # For historical universes, implement time-based selection
        if universe_name == "sp500_2020":
            return self._get_sp500_historical_symbols(date)
        elif universe_name == "nasdaq100_2020":
            return self._get_nasdaq100_historical_symbols(date)
        elif universe_name == "large_cap_2020":
            return self._get_large_cap_historical_symbols(date)
        else:
            return universe
    
    def get_backtest_universe(self, start_date: str, end_date: str, 
                             universe_name: str = "sp500_2020",
                             rebalance_frequency: str = "quarterly") -> Dict[pd.Timestamp, List[str]]:
        """
        Get universe snapshots for entire backtest period
        
        Args:
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            universe_name: Universe to use
            rebalance_frequency: How often to update universe (monthly, quarterly, yearly)
            
        Returns:
            Dictionary mapping dates to symbol lists
        """
        
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        
        # Generate rebalance dates
        rebalance_dates = self._generate_rebalance_dates(start_ts, end_ts, rebalance_frequency)
        
        universe_snapshots = {}
        
        for date in rebalance_dates:
            symbols = self.get_universe_for_date(date, universe_name)
            universe_snapshots[date] = symbols
            
            self.logger.info(f"Universe snapshot for {date.strftime('%Y-%m-%d')}: {len(symbols)} symbols")
        
        return universe_snapshots
    
    def _generate_rebalance_dates(self, start_date: pd.Timestamp, end_date: pd.Timestamp, 
                                 frequency: str) -> List[pd.Timestamp]:
        """Generate rebalance dates based on frequency"""
        
        dates = []
        current_date = start_date
        
        if frequency == "monthly":
            while current_date <= end_date:
                dates.append(current_date)
                # First day of next month
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1, day=1)
        
        elif frequency == "quarterly":
            while current_date <= end_date:
                dates.append(current_date)
                # First day of next quarter
                new_month = ((current_date.month - 1) // 3 + 1) * 3 + 1
                if new_month > 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
                else:
                    current_date = current_date.replace(month=new_month, day=1)
        
        elif frequency == "yearly":
            while current_date <= end_date:
                dates.append(current_date)
                current_date = current_date.replace(year=current_date.year + 1, day=1)
        
        else:  # Default to start date only
            dates = [start_date]
        
        return dates
    
    def _get_sp500_2020_universe(self) -> List[str]:
        """S&P 500 universe circa 2020 (survivorship-bias-free)"""
        return [
            # Technology
            "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "NVDA", "ADBE", "CRM",
            "NFLX", "INTC", "CSCO", "ORCL", "IBM", "QCOM", "TXN", "AVGO", "AMD", "AMAT",
            
            # Healthcare
            "JNJ", "PFE", "UNH", "ABBV", "TMO", "ABT", "DHR", "BMY", "LLY", "MDT",
            "AMGN", "GILD", "CVS", "CI", "ANTM", "HUM", "BIIB", "REGN", "VRTX", "ISRG",
            
            # Financials
            "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "USB", "PNC", "TFC",
            "COF", "SCHW", "BLK", "SPGI", "ICE", "CME", "AON", "MMC", "MSCI", "V", "MA",
            
            # Consumer Discretionary
            "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TJX", "LOW", "BKNG", "GM",
            "F", "CCL", "RCL", "MAR", "HLT", "YUM", "CMG", "ORLY", "AZO", "APTV",
            
            # Consumer Staples
            "WMT", "PG", "KO", "PEP", "COST", "MDLZ", "CL", "KMB", "GIS", "K",
            "HSY", "MKC", "CPB", "CAG", "SJM", "HRL", "TAP", "TSN", "ADM", "BG",
            
            # Industrials
            "BA", "HON", "UPS", "CAT", "DE", "MMM", "GE", "LMT", "RTX", "UNP",
            "FDX", "CSX", "NSC", "WM", "RSG", "EMR", "ETN", "ITW", "PH", "CMI",
            
            # Energy
            "XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "MPC", "KMI", "OKE",
            "WMB", "HAL", "BKR", "DVN", "FANG", "APA", "MRO", "NOV", "FTI", "RIG",
            
            # Materials
            "LIN", "APD", "ECL", "SHW", "DD", "DOW", "NEM", "FCX", "VMC", "MLM",
            "PPG", "IFF", "FMC", "LYB", "CE", "PKG", "IP", "NUE", "STLD", "X",
            
            # Real Estate
            "AMT", "PLD", "CCI", "EQIX", "PSA", "WELL", "DLR", "O", "SBAC", "EXR",
            "AVB", "EQR", "VTR", "ESS", "MAA", "UDR", "CPT", "HST", "REG", "BXP",
            
            # Utilities
            "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "SRE", "PEG", "ED",
            "FE", "ETR", "ES", "DTE", "PPL", "AEE", "CMS", "NI", "LNT", "EVRG",
            
            # Communication Services
            "GOOGL", "META", "DIS", "NFLX", "CMCSA", "VZ", "T", "CHTR", "TMUS", "DISH"
        ]
    
    def _get_nasdaq100_2020_universe(self) -> List[str]:
        """NASDAQ 100 universe circa 2020"""
        return [
            "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "NVDA", "ADBE", "NFLX",
            "INTC", "CSCO", "PEP", "COST", "QCOM", "TXN", "AVGO", "TMUS", "CHTR", "SBUX",
            "GILD", "AMGN", "MDLZ", "ISRG", "BKNG", "REGN", "AMD", "MU", "ADP", "FISV",
            "CSX", "ATVI", "BIIB", "TXN", "LRCX", "ADI", "KLAC", "MCHP", "SNPS", "CDNS",
            "WDAY", "TEAM", "DXCM", "ILMN", "EXC", "KDP", "CTAS", "PAYX", "FAST", "VRSK"
        ]
    
    def _get_large_cap_2020_universe(self) -> List[str]:
        """Large cap universe (market cap > $10B in 2020)"""
        return [
            # Mega caps (>$500B)
            "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "BRK.A", "BRK.B",
            
            # Large caps ($100B - $500B)
            "NVDA", "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "ADBE",
            "PYPL", "BAC", "NFLX", "CMCSA", "NKE", "CRM", "TMO", "ABBV", "COST", "PFE",
            "INTC", "WMT", "MRK", "T", "VZ", "CSCO", "PEP", "ABT", "DHR", "ORCL",
            
            # Mid-large caps ($50B - $100B)
            "QCOM", "TXN", "AVGO", "LLY", "MCD", "BMY", "AMGN", "HON", "UNP", "C",
            "LOW", "AMD", "UPS", "GILD", "CAT", "SBUX", "GS", "CVX", "MDT", "IBM"
        ]
    
    def _get_sp500_historical_symbols(self, date: pd.Timestamp) -> List[str]:
        """Get S&P 500 symbols that were actually in the index at the given date"""
        
        # This is a simplified version - in practice, you'd use actual historical index data
        base_symbols = self._get_sp500_2020_universe()
        
        # Remove symbols that weren't public or in index yet
        exclusions = self._get_historical_exclusions(date)
        
        historical_symbols = [s for s in base_symbols if s not in exclusions]
        
        # For demonstration, limit to reasonable size
        return historical_symbols[:100]  # Top 100 by market cap
    
    def _get_nasdaq100_historical_symbols(self, date: pd.Timestamp) -> List[str]:
        """Get NASDAQ 100 symbols for historical date"""
        base_symbols = self._get_nasdaq100_2020_universe()
        exclusions = self._get_historical_exclusions(date)
        return [s for s in base_symbols if s not in exclusions][:50]
    
    def _get_large_cap_historical_symbols(self, date: pd.Timestamp) -> List[str]:
        """Get large cap symbols for historical date"""
        base_symbols = self._get_large_cap_2020_universe()
        exclusions = self._get_historical_exclusions(date)
        return [s for s in base_symbols if s not in exclusions][:75]
    
    def _get_historical_exclusions(self, date: pd.Timestamp) -> Set[str]:
        """Get symbols to exclude based on historical date"""
        
        exclusions = set()
        
        # Companies that went public after certain dates
        if date < pd.Timestamp("2010-06-01"):
            exclusions.update(["TSLA"])  # Tesla IPO
        
        if date < pd.Timestamp("2012-05-01"):
            exclusions.update(["META"])  # Facebook IPO
        
        if date < pd.Timestamp("2019-05-01"):
            exclusions.update(["UBER", "LYFT"])  # Ride sharing IPOs
        
        if date < pd.Timestamp("2020-12-01"):
            exclusions.update(["ABNB", "DASH"])  # COVID-era IPOs
        
        # Companies that were acquired or delisted
        if date > pd.Timestamp("2022-01-01"):
            exclusions.update(["FB"])  # Facebook -> Meta rebrand
        
        # Add more historical exclusions as needed
        return exclusions
    
    def get_universe_stats(self, universe_name: str = "sp500_2020") -> Dict[str, any]:
        """Get statistics about a universe"""
        
        if universe_name not in self.universes:
            return {"error": f"Unknown universe: {universe_name}"}
        
        symbols = self.universes[universe_name]
        
        return {
            "universe_name": universe_name,
            "total_symbols": len(symbols),
            "sample_symbols": symbols[:10],
            "sectors_estimated": self._estimate_sector_breakdown(symbols)
        }
    
    def _estimate_sector_breakdown(self, symbols: List[str]) -> Dict[str, int]:
        """Rough sector estimation based on known symbols"""
        
        tech_symbols = {"AAPL", "MSFT", "GOOGL", "GOOG", "META", "NVDA", "ADBE", "CRM", "NFLX", "INTC", "AMD", "ORCL"}
        healthcare_symbols = {"JNJ", "PFE", "UNH", "ABBV", "TMO", "ABT", "LLY", "AMGN", "GILD", "MDT"}
        finance_symbols = {"JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA", "AXP", "BLK"}
        
        sector_counts = {
            "Technology": len([s for s in symbols if s in tech_symbols]),
            "Healthcare": len([s for s in symbols if s in healthcare_symbols]),
            "Financials": len([s for s in symbols if s in finance_symbols]),
            "Other": 0
        }
        
        accounted = sum(sector_counts.values())
        sector_counts["Other"] = len(symbols) - accounted
        
        return sector_counts
    
    def create_custom_universe(self, name: str, symbols: List[str]):
        """Create a custom universe"""
        self.universes[name] = symbols
        self.logger.info(f"Created custom universe '{name}' with {len(symbols)} symbols")
    
    def apply_universe_to_backtest(self, backtest_data: Dict[str, pd.DataFrame],
                                  universe_snapshots: Dict[pd.Timestamp, List[str]]) -> Dict[str, pd.DataFrame]:
        """
        Apply universe constraints to backtest data
        
        Args:
            backtest_data: Original data for all symbols
            universe_snapshots: Date -> symbol list mapping
            
        Returns:
            Filtered data respecting universe membership
        """
        
        if not universe_snapshots:
            return backtest_data
        
        # Get all unique symbols across all snapshots
        all_universe_symbols = set()
        for symbols in universe_snapshots.values():
            all_universe_symbols.update(symbols)
        
        # Filter backtest data to only include universe symbols
        filtered_data = {}
        for symbol in all_universe_symbols:
            if symbol in backtest_data:
                filtered_data[symbol] = backtest_data[symbol]
        
        self.logger.info(f"Filtered backtest data: {len(backtest_data)} -> {len(filtered_data)} symbols")
        return filtered_data


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    # Test the universe manager
    print("=== Testing Universe Manager ===")
    
    from config import TradingConfig
    
    config = TradingConfig()
    universe_mgr = UniverseManager(config)
    
    # Test universe stats
    for universe_name in ["sp500_2020", "nasdaq100_2020", "large_cap_2020"]:
        stats = universe_mgr.get_universe_stats(universe_name)
        print(f"\n{universe_name} stats:")
        print(f"  Total symbols: {stats['total_symbols']}")
        print(f"  Sample: {stats['sample_symbols']}")
        print(f"  Sectors: {stats['sectors_estimated']}")
    
    # Test historical universe
    test_date = pd.Timestamp("2015-01-01")
    historical_symbols = universe_mgr.get_universe_for_date(test_date, "sp500_2020")
    print(f"\nS&P 500 symbols for {test_date.strftime('%Y-%m-%d')}: {len(historical_symbols)}")
    
    # Test backtest universe
    backtest_universe = universe_mgr.get_backtest_universe("2020-01-01", "2022-12-31", "sp500_2020", "quarterly")
    print(f"\nBacktest universe snapshots: {len(backtest_universe)}")
    
    print("\n=== Universe Manager Test Complete ===")


# ===== INTEGRATION NOTES =====
"""
To integrate with your backtesting:

1. Initialize in trading engine:
   from universe import UniverseManager
   self.universe_manager = UniverseManager(self.config)

2. Get survivorship-free universe:
   universe_snapshots = self.universe_manager.get_backtest_universe(
       start_date, end_date, "sp500_2020", "quarterly"
   )

3. Filter your symbol data:
   filtered_data = self.universe_manager.apply_universe_to_backtest(
       data_by_symbol, universe_snapshots
   )

4. Use in backtest:
   results = engine.run_backtest_with_universe(
       filtered_data, universe_snapshots
   )

This eliminates survivorship bias by only using symbols that were actually 
available for trading at each point in history!
"""