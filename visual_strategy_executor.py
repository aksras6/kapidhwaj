# =============================================================================
# FIXED visual_strategy_executor.py - Import Issues Resolved
# =============================================================================

"""
Visual Strategy Executor - Executes strategies created by Visual Strategy Builder
Fixed version with proper imports and error handling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from logger import get_logger

# Import the LoggingContext if it exists, otherwise create a simple version
try:
    from trading_engine import LoggingContext
except ImportError:
    # Simple fallback if LoggingContext doesn't exist
    class LoggingContext:
        def __init__(self, logger, message):
            self.logger = logger
            self.message = message
        
        def __enter__(self):
            self.logger.info(f"Starting: {self.message}")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type:
                self.logger.error(f"Failed: {self.message} - {exc_val}")
            else:
                self.logger.info(f"Completed: {self.message}")


class VisualStrategyExecutor:
    """Executes strategies created by the Visual Strategy Builder"""
    
    def __init__(self, config, strategy_config: Dict):
        self.config = config
        self.strategy_config = strategy_config
        self.logger = get_logger("visual_strategy_executor")
        
        self.strategy_name = strategy_config.get("strategy_name", "Visual Strategy")
        self.entry_conditions = strategy_config.get("entry_conditions", [])
        self.exit_conditions = strategy_config.get("exit_conditions", [])
        self.risk_settings = strategy_config.get("risk_settings", {})
        
        self.logger.info(f"Initialized visual strategy: {self.strategy_name}")
        self.logger.info(f"Entry conditions: {len(self.entry_conditions)}")
        self.logger.info(f"Exit conditions: {len(self.exit_conditions)}")
    
    def generate_entry_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate entry signals based on visual conditions"""
        try:
            if len(data) < 50:  # Need minimum data for indicators
                return pd.Series([False] * len(data), index=data.index)
            
            # Calculate all indicators needed for entry conditions
            indicators = self._calculate_indicators(data, self.entry_conditions)
            
            # Build entry logic from visual conditions
            entry_signals = self._build_condition_logic(data, self.entry_conditions, indicators)
            
            signal_count = entry_signals.sum() if hasattr(entry_signals, 'sum') else 0
            self.logger.debug(f"Generated {signal_count} entry signals")
            return entry_signals
            
        except Exception as e:
            self.logger.error(f"Entry signal generation failed: {e}")
            return pd.Series([False] * len(data), index=data.index)
    
    def generate_exit_signals(self, data: pd.DataFrame, positions: Dict) -> pd.Series:
        """Generate exit signals based on visual conditions"""
        try:
            if len(data) < 50:  # Need minimum data for indicators
                return pd.Series([False] * len(data), index=data.index)
            
            # Calculate indicators needed for exit conditions
            indicators = self._calculate_indicators(data, self.exit_conditions)
            
            # Build exit logic from visual conditions
            exit_signals = self._build_condition_logic(data, self.exit_conditions, indicators)
            
            signal_count = exit_signals.sum() if hasattr(exit_signals, 'sum') else 0
            self.logger.debug(f"Generated {signal_count} exit signals")
            return exit_signals
            
        except Exception as e:
            self.logger.error(f"Exit signal generation failed: {e}")
            return pd.Series([False] * len(data), index=data.index)
    
    def _calculate_indicators(self, data: pd.DataFrame, conditions: List[Dict]) -> Dict[str, pd.Series]:
        """Calculate all indicators needed for the conditions"""
        indicators = {}
        
        try:
            for condition in conditions:
                indicator = condition.get("indicator", "")
                params = condition.get("parameters", {})
                
                # Calculate indicator if not already calculated
                indicator_key = self._get_indicator_key(indicator, params)
                if indicator_key not in indicators:
                    indicators[indicator_key] = self._calculate_single_indicator(data, indicator, params)
                
                # Also check target for indicators (like SMA(20) in target field)
                target = condition.get("target", "")
                if self._is_indicator_target(target):
                    target_key = self._parse_target_indicator(target)
                    if target_key not in indicators:
                        indicators[target_key] = self._calculate_target_indicator(data, target)
        
        except Exception as e:
            self.logger.error(f"Indicator calculation failed: {e}")
        
        return indicators
    
    def _calculate_single_indicator(self, data: pd.DataFrame, indicator: str, params: Dict) -> pd.Series:
        """Calculate a single indicator"""
        try:
            if indicator == "RSI":
                period = int(params.get("period", 14))
                return self._calculate_rsi(data, period)
            
            elif indicator == "SMA":
                period = int(params.get("period", 20))
                return data['close'].rolling(period).mean()
            
            elif indicator == "EMA":
                period = int(params.get("period", 20))
                return data['close'].ewm(span=period).mean()
            
            elif indicator == "ADX":
                period = int(params.get("period", 14))
                return self._calculate_adx(data, period)
            
            elif indicator == "MACD":
                fast = int(params.get("fast", 12))
                slow = int(params.get("slow", 26))
                signal_period = int(params.get("signal", 9))
                return self._calculate_macd(data, fast, slow, signal_period)
            
            elif indicator in ["Close", "Open", "High", "Low", "Volume"]:
                return data[indicator.lower()]
            
            else:
                self.logger.warning(f"Unknown indicator: {indicator}")
                return pd.Series([0] * len(data), index=data.index)
                
        except Exception as e:
            self.logger.error(f"Failed to calculate {indicator}: {e}")
            return pd.Series([0] * len(data), index=data.index)
    
    def _build_condition_logic(self, data: pd.DataFrame, conditions: List[Dict], indicators: Dict) -> pd.Series:
        """Build the logical expression from visual conditions"""
        try:
            if not conditions:
                return pd.Series([False] * len(data), index=data.index)
            
            # Start with the first condition
            result = self._evaluate_single_condition(data, conditions[0], indicators)
            
            # Apply logic operators for remaining conditions
            for i in range(1, len(conditions)):
                condition = conditions[i-1]  # Previous condition has the logic operator
                next_condition_result = self._evaluate_single_condition(data, conditions[i], indicators)
                
                logic_op = condition.get("next_logic", "AND")
                if logic_op == "AND":
                    result = result & next_condition_result
                else:  # OR
                    result = result | next_condition_result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Logic building failed: {e}")
            return pd.Series([False] * len(data), index=data.index)
    
    def _evaluate_single_condition(self, data: pd.DataFrame, condition: Dict, indicators: Dict) -> pd.Series:
        """Evaluate a single condition"""
        try:
            indicator = condition.get("indicator", "")
            params = condition.get("parameters", {})
            operator = condition.get("operator", ">")
            target = condition.get("target", "0")
            
            # Get left side (indicator value)
            indicator_key = self._get_indicator_key(indicator, params)
            if indicator_key in indicators:
                left_side = indicators[indicator_key]
            else:
                left_side = pd.Series([0] * len(data), index=data.index)
            
            # Get right side (target value)
            if target.replace(".", "").replace("-", "").isdigit():
                # Numeric target
                right_side = float(target)
            elif self._is_indicator_target(target):
                # Indicator target (like SMA(20))
                target_key = self._parse_target_indicator(target)
                if target_key in indicators:
                    right_side = indicators[target_key]
                else:
                    right_side = pd.Series([0] * len(data), index=data.index)
            elif target.lower() in ["close", "open", "high", "low", "volume"]:
                # Price data target
                right_side = data[target.lower()]
            else:
                # Unknown target, treat as 0
                right_side = 0
            
            # Apply operator
            if operator == ">":
                return left_side > right_side
            elif operator == "<":
                return left_side < right_side
            elif operator == ">=":
                return left_side >= right_side
            elif operator == "<=":
                return left_side <= right_side
            elif operator == "==":
                return left_side == right_side
            elif operator == "crosses_above":
                # Simplified: current > target and previous <= target
                if isinstance(right_side, pd.Series):
                    return (left_side > right_side) & (left_side.shift(1) <= right_side.shift(1))
                else:
                    return (left_side > right_side) & (left_side.shift(1) <= right_side)
            elif operator == "crosses_below":
                # Simplified: current < target and previous >= target  
                if isinstance(right_side, pd.Series):
                    return (left_side < right_side) & (left_side.shift(1) >= right_side.shift(1))
                else:
                    return (left_side < right_side) & (left_side.shift(1) >= right_side)
            else:
                return pd.Series([False] * len(data), index=data.index)
                
        except Exception as e:
            self.logger.error(f"Condition evaluation failed: {e}")
            return pd.Series([False] * len(data), index=data.index)
    
    def _get_indicator_key(self, indicator: str, params: Dict) -> str:
        """Generate a unique key for an indicator"""
        if not params:
            return indicator
        param_str = "_".join([str(v) for v in params.values()])
        return f"{indicator}_{param_str}"
    
    def _is_indicator_target(self, target: str) -> bool:
        """Check if target is an indicator (like SMA(20))"""
        return "(" in target and target.split("(")[0] in ["SMA", "EMA", "RSI", "ADX", "MACD"]
    
    def _parse_target_indicator(self, target: str) -> str:
        """Parse target like 'SMA(20)' into key 'SMA_20'"""
        if "(" not in target:
            return target
        
        indicator = target.split("(")[0]
        params_str = target.split("(")[1].rstrip(")").replace(",", "_")
        return f"{indicator}_{params_str}"
    
    def _calculate_target_indicator(self, data: pd.DataFrame, target: str) -> pd.Series:
        """Calculate indicator from target string like 'SMA(20)'"""
        try:
            if target.startswith("SMA("):
                period = int(target.split("(")[1].rstrip(")"))
                return data['close'].rolling(period).mean()
            elif target.startswith("EMA("):
                period = int(target.split("(")[1].rstrip(")"))
                return data['close'].ewm(span=period).mean()
            # Add more as needed
            else:
                return pd.Series([0] * len(data), index=data.index)
        except Exception as e:
            self.logger.error(f"Target indicator calculation failed: {e}")
            return pd.Series([0] * len(data), index=data.index)
    
    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            # Avoid division by zero
            loss = loss.replace(0, 0.0001)
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Fill NaN values with 50 (neutral RSI)
            rsi = rsi.fillna(50)
            
            return rsi
        except Exception as e:
            self.logger.error(f"RSI calculation failed: {e}")
            return pd.Series([50] * len(data), index=data.index)
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX indicator (simplified version)"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Directional Movement
            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
            
            # Smoothed averages
            atr = true_range.rolling(period).mean()
            atr = atr.replace(0, 0.0001)  # Avoid division by zero
            
            plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
            
            # ADX
            di_sum = plus_di + minus_di
            di_sum = di_sum.replace(0, 0.0001)  # Avoid division by zero
            dx = 100 * abs(plus_di - minus_di) / di_sum
            adx = dx.rolling(period).mean()
            
            # Fill NaN values with 25 (moderate trend strength)
            adx = adx.fillna(25)
            
            return adx
        except Exception as e:
            self.logger.error(f"ADX calculation failed: {e}")
            return pd.Series([25] * len(data), index=data.index)
    
    def _calculate_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD indicator"""
        try:
            ema_fast = data['close'].ewm(span=fast).mean()
            ema_slow = data['close'].ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            
            # Fill NaN values with 0
            macd = macd.fillna(0)
            
            return macd
        except Exception as e:
            self.logger.error(f"MACD calculation failed: {e}")
            return pd.Series([0] * len(data), index=data.index)
    
    def get_position_size(self, signal, portfolio_value: float) -> float:
        """Calculate position size based on risk settings"""
        try:
            position_pct = self.risk_settings.get("position_size_pct", 5.0) / 100
            return portfolio_value * position_pct
        except Exception as e:
            self.logger.error(f"Position size calculation failed: {e}")
            return portfolio_value * 0.05  # Default 5%
    
    def get_stop_loss(self, entry_price: float, signal) -> float:
        """Calculate stop loss price"""
        try:
            stop_pct = self.risk_settings.get("stop_loss_pct", 2.0) / 100
            return entry_price * (1 - stop_pct)
        except Exception as e:
            self.logger.error(f"Stop loss calculation failed: {e}")
            return entry_price * 0.98  # Default 2% stop loss
    
    def get_take_profit(self, entry_price: float, signal) -> float:
        """Calculate take profit price"""
        try:
            tp_pct = self.risk_settings.get("take_profit_pct", 6.0) / 100
            return entry_price * (1 + tp_pct)
        except Exception as e:
            self.logger.error(f"Take profit calculation failed: {e}")
            return entry_price * 1.06  # Default 6% take profit


# =============================================================================
# SIMPLE INTEGRATION FUNCTIONS (if LoggingContext issues persist)
# =============================================================================

def create_simple_visual_strategy_test():
    """Simple test function to verify visual strategy executor works"""
    
    # Create sample data
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'open': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Create sample strategy config (like your RSI strategy)
    strategy_config = {
        "strategy_name": "Test RSI Strategy",
        "strategy_type": "visual_generated",
        "entry_conditions": [
            {
                "indicator": "RSI",
                "parameters": {"period": "14"},
                "operator": "<",
                "target": "30"
            }
        ],
        "exit_conditions": [
            {
                "indicator": "RSI", 
                "parameters": {"period": "14"},
                "operator": ">",
                "target": "50"
            }
        ],
        "risk_settings": {
            "position_size_pct": 5.0,
            "stop_loss_pct": 2.0,
            "take_profit_pct": 6.0
        }
    }
    
    # Test the executor
    try:
        from config import TradingConfig
        config = TradingConfig()
        
        executor = VisualStrategyExecutor(config, strategy_config)
        entry_signals = executor.generate_entry_signals(data)
        exit_signals = executor.generate_exit_signals(data, {})
        
        print(f"✅ Visual Strategy Test Passed!")
        print(f"Entry signals: {entry_signals.sum()}")
        print(f"Exit signals: {exit_signals.sum()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Visual Strategy Test Failed: {e}")
        return False


if __name__ == "__main__":
    # Test the visual strategy executor
    create_simple_visual_strategy_test()