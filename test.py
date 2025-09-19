
# Test the visual strategy executor
from visual_strategy_executor import VisualStrategyExecutor
from config import TradingConfig

# Create test config
config = TradingConfig()

# Create test strategy (MACD example)
strategy_config = {
    "strategy_name": "Test MACD Strategy",
    "strategy_type": "visual_generated",
    "entry_conditions": [
        {
            "indicator": "MACD",
            "parameters": {"fast": "12", "slow": "26", "signal": "9"},
            "operator": ">",
            "target": "0"
        }
    ],
    "exit_conditions": [
        {
            "indicator": "MACD",
            "parameters": {"fast": "12", "slow": "26", "signal": "9"},
            "operator": "<",
            "target": "0"
        }
    ],
    "risk_settings": {
        "position_size_pct": 5.0,
        "stop_loss_pct": 2.0,
        "take_profit_pct": 6.0
    }
}

# Test creation
try:
    executor = VisualStrategyExecutor(config, strategy_config)
    print("✅ Visual Strategy Executor works!")
except Exception as e:
    print(f"❌ Visual Strategy Executor failed: {e}")
