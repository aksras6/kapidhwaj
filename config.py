"""
Trading System Configuration Management

This centralizes all configuration parameters that were scattered throughout
your original code. Easy to modify without touching core logic.
"""

from dataclasses import dataclass
from typing import List, Optional
import os
from datetime import datetime

@dataclass
class TradingConfig:
    """Main configuration class for the trading system"""
    
    # ===== STRATEGY PARAMETERS =====
    CHANNEL_LENGTH: int = 20                    # Donchian channel lookback period
    ADX_LENGTH: int = 15                        # ADX calculation period  
    ADX_THRESHOLD: float = 20.0                 # ADX threshold for trading eligibility
    WARMUP_BARS: int = None                     # Auto-calculated from other params
    
    # ===== PORTFOLIO MANAGEMENT =====
    TRADE_PERCENT: float = 15.0                 # % of equity per position
    ACCOUNT_SIZE: float = 10_000.0              # Starting capital
    MAX_POSITIONS: int = 10                     # Maximum concurrent positions
    MAX_EXPOSURE_PCT: float = 100.0             # Maximum portfolio exposure %
    
    # ===== TRADING COSTS =====
    COMMISSION_PER_SHARE: float = 0.01          # Commission per share
    SLIPPAGE_PER_SHARE: float = 0.02           # Slippage per share
    
    # ===== INTERACTIVE BROKERS CONNECTION =====
    IB_HOST: str = "127.0.0.1"                 # IB Gateway/TWS host
    IB_PAPER_PORT: int = 7497                  # Paper trading port
    IB_LIVE_PORT: int = 7496                   # Live trading port
    IB_CLIENT_ID: int = 44                     # Client ID for connection
    IB_TIMEOUT: int = 10                       # Connection timeout seconds
    
    # ===== DATA SETTINGS =====
    BAR_SIZE: str = "1 day"                    # Bar size for historical data
    USE_RTH: int = 1                           # Use regular trading hours
    WHAT_TO_SHOW: str = "TRADES"               # Data type to show
    START_DATE: Optional[str] = "2020-01-01"   # Backtest start date (YYYY-MM-DD)
    DURATION_YEARS: str = "8 Y"                # Fallback duration if START_DATE fails
    
    # ===== SYMBOLS TO TRADE =====
    SYMBOLS: List[str] = None                  # Will be set in __post_init__
    
    # ===== ORDER MANAGEMENT =====
    PRICE_TOLERANCE_FIXED: float = 0.01       # Fixed tolerance for very low prices
    PRICE_TOLERANCE_RELATIVE: float = 1e-4    # Relative tolerance (0.01%)
    ORDER_TIF: str = "GTC"                     # Time in force for orders
    
    # ===== FILE PATHS =====
    OUTPUT_DIR: str = "output"                 # Directory for CSV exports
    LOG_DIR: str = "logs"                      # Directory for log files
    STRATEGY_DIR: str = "strategies"           # Directory for strategy files
    
    # ===== SYMBOL NORMALIZATION =====
    SYMBOL_NORMALIZE: dict = None              # Symbol normalization map
    
    def __post_init__(self):
        """Initialize computed values and defaults"""
        
        # Set default symbols if not provided
        if self.SYMBOLS is None:
            self.SYMBOLS = [
                "MSFT", "NVDA", "AAPL", "AMZN", "GOOGL", 
                "META", "AVGO", "TSLA", "WMT", "JPM", 
                "V", "SPY", "BRK.A"
            ]
        
        # Calculate warmup bars if not set
        if self.WARMUP_BARS is None:
            self.WARMUP_BARS = max(self.ADX_LENGTH * 5, self.CHANNEL_LENGTH + 5)
        
        # Set symbol normalization if not provided
        if self.SYMBOL_NORMALIZE is None:
            self.SYMBOL_NORMALIZE = {
                "BRK.A": "BRK A",
                "BRK.B": "BRK B",
            }
        
        # Ensure directories exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        for directory in [self.OUTPUT_DIR, self.LOG_DIR, self.STRATEGY_DIR]:
            os.makedirs(directory, exist_ok=True)
    
    def get_price_tolerance(self, price: float) -> float:
        """Get relative price tolerance for order idempotency"""
        return max(self.PRICE_TOLERANCE_FIXED, self.PRICE_TOLERANCE_RELATIVE * price)
    
    def prices_equal(self, price1: float, price2: float) -> bool:
        """Check if two prices are equal within tolerance"""
        if price1 == 0 and price2 == 0:
            return True
        avg_price = (abs(price1) + abs(price2)) / 2
        tolerance = self.get_price_tolerance(avg_price)
        return abs(price1 - price2) <= tolerance
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        if self.TRADE_PERCENT <= 0 or self.TRADE_PERCENT > 100:
            issues.append("TRADE_PERCENT must be between 0 and 100")
        
        if self.MAX_POSITIONS <= 0:
            issues.append("MAX_POSITIONS must be positive")
        
        if self.ACCOUNT_SIZE <= 0:
            issues.append("ACCOUNT_SIZE must be positive")
        
        if self.ADX_LENGTH <= 0:
            issues.append("ADX_LENGTH must be positive")
        
        if self.CHANNEL_LENGTH <= 0:
            issues.append("CHANNEL_LENGTH must be positive")
        
        if not self.SYMBOLS:
            issues.append("SYMBOLS list cannot be empty")
        
        return issues
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for saving/loading"""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TradingConfig':
        """Create config from dictionary"""
        return cls(**data)
    
    def save_to_file(self, filepath: str):
        """Save configuration to file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'TradingConfig':
        """Load configuration from file"""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# ===== ENVIRONMENT-SPECIFIC CONFIGS =====

class PaperTradingConfig(TradingConfig):
    """Configuration optimized for paper trading"""
    
    def __post_init__(self):
        super().__post_init__()
        self.IB_CLIENT_ID = 44  # Different client ID for paper
        self.OUTPUT_DIR = "output/paper"
        self._create_directories()


class LiveTradingConfig(TradingConfig):
    """Configuration for live trading with additional safety checks"""
    
    def __post_init__(self):
        super().__post_init__()
        self.IB_CLIENT_ID = 55  # Different client ID for live
        self.OUTPUT_DIR = "output/live"
        
        # More conservative settings for live trading
        self.MAX_POSITIONS = min(self.MAX_POSITIONS, 5)  # Limit positions
        self.TRADE_PERCENT = min(self.TRADE_PERCENT, 10.0)  # Smaller position sizes
        
        self._create_directories()
    
    def validate(self) -> List[str]:
        """Additional validation for live trading"""
        issues = super().validate()
        
        if self.TRADE_PERCENT > 20:
            issues.append("TRADE_PERCENT too high for live trading (max 20%)")
        
        if self.MAX_POSITIONS > 10:
            issues.append("MAX_POSITIONS too high for live trading (max 10)")
        
        return issues


class BacktestConfig(TradingConfig):
    """Configuration optimized for backtesting"""
    
    def __post_init__(self):
        super().__post_init__()
        self.OUTPUT_DIR = "output/backtest"
        self._create_directories()
        
        # Backtesting can handle more symbols
        if len(self.SYMBOLS) < 20:
            # Add more symbols for comprehensive backtesting
            extended_symbols = [
                "MSFT", "NVDA", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "TSLA", 
                "WMT", "JPM", "V", "SPY", "BRK.A", "UNH", "XOM", "PG", "JNJ", 
                "HD", "CVX", "MA", "PFE", "ABBV", "BAC", "KO", "PEP", "COST", 
                "ADBE", "CRM", "NFLX", "DIS"
            ]
            self.SYMBOLS = extended_symbols[:25]  # Limit to 25 for performance


# ===== GLOBAL CONFIG INSTANCE =====
# This is the main config object used throughout the application
CONFIG = TradingConfig()


# ===== CONFIGURATION UTILITIES =====

def get_config_for_mode(mode: str) -> TradingConfig:
    """Get appropriate config for different modes"""
    if mode.lower() == "paper":
        return PaperTradingConfig()
    elif mode.lower() == "live":
        return LiveTradingConfig()
    elif mode.lower() == "backtest":
        return BacktestConfig()
    else:
        return TradingConfig()


def validate_environment():
    """Validate the trading environment setup"""
    issues = CONFIG.validate()
    if issues:
        print("Configuration Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    return True


if __name__ == "__main__":
    # Test the configuration system
    print("=== Testing Configuration System ===")
    
    # Test default config
    config = TradingConfig()
    issues = config.validate()
    print(f"Default config validation: {'PASSED' if not issues else 'FAILED'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    
    # Test saving/loading
    config.save_to_file("test_config.json")
    loaded_config = TradingConfig.load_from_file("test_config.json")
    print(f"Save/load test: {'PASSED' if config.to_dict() == loaded_config.to_dict() else 'FAILED'}")
    
    # Test different modes
    paper_config = get_config_for_mode("paper")
    live_config = get_config_for_mode("live")
    backtest_config = get_config_for_mode("backtest")
    
    print(f"Paper config client ID: {paper_config.IB_CLIENT_ID}")
    print(f"Live config client ID: {live_config.IB_CLIENT_ID}")
    print(f"Backtest symbols count: {len(backtest_config.SYMBOLS)}")
    
    print("\n=== Configuration Test Complete ===")