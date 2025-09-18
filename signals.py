"""
Signal Generation System for Trading Strategies

Handles all signal logic that was embedded in your backtest function.
Clean separation of strategy logic from execution logic.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import logging

from config import TradingConfig
from logger import get_logger, LoggingContext


class SignalType(Enum):
    """Types of trading signals"""
    ENTRY_LONG = "ENTRY_LONG"
    EXIT_LONG = "EXIT_LONG"
    ENTRY_SHORT = "ENTRY_SHORT"
    EXIT_SHORT = "EXIT_SHORT"
    NO_SIGNAL = "NO_SIGNAL"


class SignalStrength(Enum):
    """Signal strength levels"""
    WEAK = 0.3
    MODERATE = 0.6
    STRONG = 0.8
    VERY_STRONG = 1.0


@dataclass
class TradingSignal:
    """Container for trading signal information"""
    symbol: str
    signal_type: SignalType
    timestamp: pd.Timestamp
    strength: float  # 0.0 to 1.0
    
    # Price levels
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Signal context
    reason: str = ""
    confidence: float = 0.0
    
    # Strategy-specific data
    strategy_data: Dict = None
    
    def __post_init__(self):
        if self.strategy_data is None:
            self.strategy_data = {}
    
    def is_entry_signal(self) -> bool:
        """Check if this is an entry signal"""
        return self.signal_type in [SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT]
    
    def is_exit_signal(self) -> bool:
        """Check if this is an exit signal"""
        return self.signal_type in [SignalType.EXIT_LONG, SignalType.EXIT_SHORT]
    
    def is_long_signal(self) -> bool:
        """Check if this is a long signal"""
        return self.signal_type in [SignalType.ENTRY_LONG, SignalType.EXIT_LONG]


class BaseSignalGenerator(ABC):
    """Abstract base class for signal generators"""
    
    def __init__(self, config: TradingConfig, name: str = "BaseStrategy"):
        self.config = config
        self.name = name
        self.logger = get_logger(f"signals_{name.lower()}")
        
        # Strategy parameters (can be overridden by subclasses)
        self.min_bars_required = max(config.ADX_LENGTH, config.CHANNEL_LENGTH) + 10
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, symbol: str, 
                        current_position: Optional[dict] = None) -> List[TradingSignal]:
        """Generate trading signals from data"""
        pass
    
    def is_data_sufficient(self, data: pd.DataFrame) -> bool:
        """Check if data is sufficient for signal generation"""
        if data.empty or len(data) < self.min_bars_required:
            self.logger.debug(f"Insufficient data", symbol=data.get('symbol', [''])[0] if len(data) > 0 else '', 
                            bars=len(data), required=self.min_bars_required)
            return False
        return True
    
    def validate_indicators(self, data: pd.DataFrame, required_cols: List[str]) -> bool:
        """Validate that required indicators are present and valid"""
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            self.logger.warning(f"Missing indicators", missing=missing_cols)
            return False
        
        # Check for sufficient non-NaN values
        latest_row = data.iloc[-1]
        for col in required_cols:
            if pd.isna(latest_row[col]):
                self.logger.debug(f"NaN indicator value", indicator=col)
                return False
        
        return True


class ADXBreakoutSignalGenerator(BaseSignalGenerator):
    """
    ADX Breakout strategy signal generator
    Implements your existing ADX + Donchian channel logic
    """
    
    def __init__(self, config: TradingConfig):
        super().__init__(config, "ADXBreakout")
        
        # Strategy-specific parameters
        self.adx_threshold = config.ADX_THRESHOLD
        self.channel_length = config.CHANNEL_LENGTH
        self.adx_length = config.ADX_LENGTH
        
        self.required_indicators = ["ADX", "HH", "LL", "open", "high", "low", "close"]
        
        self.logger.info(f"ADX Breakout strategy initialized", 
                        adx_threshold=self.adx_threshold,
                        channel_length=self.channel_length)
    
    def generate_signals(self, data: pd.DataFrame, symbol: str, 
                        current_position: Optional[dict] = None) -> List[TradingSignal]:
        """Generate ADX breakout signals"""
        
        if not self.is_data_sufficient(data):
            return []
        
        if not self.validate_indicators(data, self.required_indicators):
            return []
        
        signals = []
        
        with LoggingContext(self.logger, f"Generating signals for {symbol}", level=logging.DEBUG):
            
            # Get current and previous bar data
            current_bar = data.iloc[-1]
            
            if current_position is None:
                # Look for entry signals
                entry_signal = self._check_entry_conditions(data, symbol, current_bar)
                if entry_signal:
                    signals.append(entry_signal)
            else:
                # Look for exit signals
                exit_signal = self._check_exit_conditions(data, symbol, current_bar, current_position)
                if exit_signal:
                    signals.append(exit_signal)
        
        return signals
    
    def _check_entry_conditions(self, data: pd.DataFrame, symbol: str, current_bar: pd.Series) -> Optional[TradingSignal]:
        """Check for entry signal conditions"""
        
        # ADX filter: only trade when ADX < threshold (low volatility/consolidation)
        current_adx = current_bar["ADX"]
        if pd.isna(current_adx) or current_adx >= self.adx_threshold:
            self.logger.debug(f"ADX filter failed", symbol=symbol, adx=current_adx, threshold=self.adx_threshold)
            return None
        
        # Breakout setup: we have valid channel levels
        highest_high = current_bar["HH"]
        lowest_low = current_bar["LL"]
        
        if pd.isna(highest_high) or pd.isna(lowest_low):
            self.logger.debug(f"Invalid channel levels", symbol=symbol)
            return None
        
        # Calculate signal strength based on ADX level and channel width
        strength = self._calculate_entry_strength(current_bar, data)
        confidence = self._calculate_confidence(current_bar, data)
        
        # Create entry signal (buy stop at highest high)
        signal = TradingSignal(
            symbol=symbol,
            signal_type=SignalType.ENTRY_LONG,
            timestamp=current_bar.name,
            strength=strength,
            entry_price=highest_high,
            stop_loss=lowest_low,
            confidence=confidence,
            reason=f"ADX breakout setup: ADX={current_adx:.1f} < {self.adx_threshold}, HH={highest_high:.2f}",
            strategy_data={
                "adx": current_adx,
                "highest_high": highest_high,
                "lowest_low": lowest_low,
                "channel_width": highest_high - lowest_low,
                "entry_type": "breakout"
            }
        )
        
        self.logger.info(f"Entry signal generated", symbol=symbol, 
                        entry_price=highest_high, stop_loss=lowest_low,
                        strength=strength, confidence=confidence)
        
        return signal
    
    def _check_exit_conditions(self, data: pd.DataFrame, symbol: str, 
                              current_bar: pd.Series, position: dict) -> Optional[TradingSignal]:
        """Check for exit signal conditions"""
        
        # Stop loss exit at lowest low
        stop_loss_level = current_bar["LL"]
        
        if pd.isna(stop_loss_level):
            self.logger.debug(f"Invalid stop loss level", symbol=symbol)
            return None
        
        # Calculate exit strength and confidence
        strength = self._calculate_exit_strength(current_bar, data, position)
        
        signal = TradingSignal(
            symbol=symbol,
            signal_type=SignalType.EXIT_LONG,
            timestamp=current_bar.name,
            strength=strength,
            entry_price=stop_loss_level,  # Stop level
            confidence=0.9,  # High confidence in stop loss exits
            reason=f"Stop loss exit at LL={stop_loss_level:.2f}",
            strategy_data={
                "exit_type": "stop_loss",
                "stop_level": stop_loss_level,
                "position_entry": position.get("entry_price", 0)
            }
        )
        
        self.logger.info(f"Exit signal generated", symbol=symbol, 
                        stop_level=stop_loss_level, exit_type="stop_loss")
        
        return signal
    
    def _calculate_entry_strength(self, current_bar: pd.Series, data: pd.DataFrame) -> float:
        """Calculate signal strength for entry (0.0 to 1.0)"""
        
        adx = current_bar["ADX"]
        highest_high = current_bar["HH"]
        lowest_low = current_bar["LL"]
        
        # Stronger signal when ADX is lower (more consolidation)
        adx_factor = max(0.0, (self.adx_threshold - adx) / self.adx_threshold)
        
        # Stronger signal when channel is wider (more room for breakout)
        channel_width = highest_high - lowest_low
        close_price = current_bar["close"]
        width_factor = min(1.0, (channel_width / close_price) * 10)  # Normalize
        
        # Volume factor (if available)
        volume_factor = 0.5
        if "volume" in current_bar and "Volume_SMA" in current_bar:
            current_volume = current_bar["volume"]
            avg_volume = current_bar["Volume_SMA"]
            if pd.notna(current_volume) and pd.notna(avg_volume) and avg_volume > 0:
                volume_factor = min(1.0, current_volume / avg_volume)
        
        # Combine factors
        strength = (adx_factor * 0.4 + width_factor * 0.4 + volume_factor * 0.2)
        return max(0.1, min(1.0, strength))
    
    def _calculate_exit_strength(self, current_bar: pd.Series, data: pd.DataFrame, position: dict) -> float:
        """Calculate signal strength for exit"""
        # For stop losses, strength is typically high
        return 0.9
    
    def _calculate_confidence(self, current_bar: pd.Series, data: pd.DataFrame) -> float:
        """Calculate signal confidence based on market context"""
        
        # Look at recent price action
        recent_data = data.tail(5)
        
        # Higher confidence if we're near support/resistance
        confidence = 0.6  # Base confidence
        
        # Check if price has been consolidating
        if len(recent_data) >= 5:
            recent_highs = recent_data["high"].values
            recent_lows = recent_data["low"].values
            
            # Consolidation factor (lower volatility = higher confidence)
            high_range = np.max(recent_highs) - np.min(recent_highs)
            low_range = np.max(recent_lows) - np.min(recent_lows)
            avg_price = current_bar["close"]
            
            consolidation_factor = 1.0 - min(0.5, (high_range + low_range) / (2 * avg_price))
            confidence += consolidation_factor * 0.3
        
        return max(0.1, min(1.0, confidence))


class SignalManager:
    """Manages multiple signal generators and aggregates signals"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = get_logger("signal_manager")
        
        # Register signal generators
        self.generators: Dict[str, BaseSignalGenerator] = {
            "adx_breakout": ADXBreakoutSignalGenerator(config)
        }
        
        # Signal history for analysis
        self.signal_history: List[TradingSignal] = []
        
        self.logger.info(f"Signal manager initialized with {len(self.generators)} strategies")
    
    def add_generator(self, name: str, generator: BaseSignalGenerator):
        """Add a new signal generator"""
        self.generators[name] = generator
        self.logger.info(f"Added signal generator: {name}")
    
    def generate_signals_for_symbol(self, symbol: str, data: pd.DataFrame, 
                                   current_position: Optional[dict] = None,
                                   strategy_filter: Optional[List[str]] = None) -> List[TradingSignal]:
        """Generate signals for a symbol using all or filtered strategies"""
        
        all_signals = []
        
        # Determine which generators to use
        generators_to_use = self.generators
        if strategy_filter:
            generators_to_use = {k: v for k, v in self.generators.items() if k in strategy_filter}
        
        for strategy_name, generator in generators_to_use.items():
            try:
                signals = generator.generate_signals(data, symbol, current_position)
                
                # Tag signals with strategy name
                for signal in signals:
                    signal.strategy_data["strategy_name"] = strategy_name
                
                all_signals.extend(signals)
                
                if signals:
                    self.logger.debug(f"Generated {len(signals)} signals", 
                                    strategy=strategy_name, symbol=symbol)
                
            except Exception as e:
                self.logger.error(f"Signal generation failed", 
                                strategy=strategy_name, symbol=symbol, error=str(e))
        
        # Store signals in history
        self.signal_history.extend(all_signals)
        
        # Keep history manageable
        if len(self.signal_history) > 10000:
            self.signal_history = self.signal_history[-5000:]  # Keep last 5000
        
        return all_signals
    
    def generate_signals_for_portfolio(self, data_by_symbol: Dict[str, pd.DataFrame],
                                     current_positions: Dict[str, dict] = None,
                                     strategy_filter: Optional[List[str]] = None) -> Dict[str, List[TradingSignal]]:
        """Generate signals for entire portfolio"""
        
        if current_positions is None:
            current_positions = {}
        
        all_signals = {}
        
        with LoggingContext(self.logger, f"Generating portfolio signals for {len(data_by_symbol)} symbols"):
            
            for symbol, data in data_by_symbol.items():
                current_position = current_positions.get(symbol)
                
                signals = self.generate_signals_for_symbol(
                    symbol, data, current_position, strategy_filter
                )
                
                if signals:
                    all_signals[symbol] = signals
        
        total_signals = sum(len(signals) for signals in all_signals.values())
        self.logger.info(f"Generated {total_signals} total signals across {len(all_signals)} symbols")
        
        return all_signals
    
    def filter_signals(self, signals: List[TradingSignal], 
                      min_strength: float = 0.0,
                      min_confidence: float = 0.0,
                      signal_types: Optional[List[SignalType]] = None) -> List[TradingSignal]:
        """Filter signals based on criteria"""
        
        filtered = signals
        
        # Filter by strength
        if min_strength > 0:
            filtered = [s for s in filtered if s.strength >= min_strength]
        
        # Filter by confidence
        if min_confidence > 0:
            filtered = [s for s in filtered if s.confidence >= min_confidence]
        
        # Filter by signal type
        if signal_types:
            filtered = [s for s in filtered if s.signal_type in signal_types]
        
        self.logger.debug(f"Filtered signals", original=len(signals), filtered=len(filtered))
        return filtered
    
    def rank_signals(self, signals: List[TradingSignal], 
                    ranking_method: str = "composite") -> List[TradingSignal]:
        """Rank signals by priority"""
        
        if not signals:
            return signals
        
        if ranking_method == "composite":
            # Rank by combination of strength and confidence
            return sorted(signals, 
                         key=lambda s: (s.strength * 0.6 + s.confidence * 0.4), 
                         reverse=True)
        
        elif ranking_method == "strength":
            return sorted(signals, key=lambda s: s.strength, reverse=True)
        
        elif ranking_method == "confidence":
            return sorted(signals, key=lambda s: s.confidence, reverse=True)
        
        else:
            self.logger.warning(f"Unknown ranking method: {ranking_method}")
            return signals
    
    def get_signal_statistics(self, days_back: int = 30) -> Dict[str, any]:
        """Get statistics about recent signals"""
        
        if not self.signal_history:
            return {}
        
        # Filter to recent signals
        cutoff_time = pd.Timestamp.now() - pd.Timedelta(days=days_back)
        recent_signals = [s for s in self.signal_history if s.timestamp >= cutoff_time]
        
        if not recent_signals:
            return {"message": "No recent signals"}
        
        # Calculate statistics
        stats = {
            "total_signals": len(recent_signals),
            "entry_signals": len([s for s in recent_signals if s.is_entry_signal()]),
            "exit_signals": len([s for s in recent_signals if s.is_exit_signal()]),
            "avg_strength": np.mean([s.strength for s in recent_signals]),
            "avg_confidence": np.mean([s.confidence for s in recent_signals]),
            "symbols_with_signals": len(set(s.symbol for s in recent_signals)),
            "strategies_active": len(set(s.strategy_data.get("strategy_name") for s in recent_signals))
        }
        
        # Signal distribution by strategy
        strategy_counts = {}
        for signal in recent_signals:
            strategy = signal.strategy_data.get("strategy_name", "unknown")
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        stats["signals_by_strategy"] = strategy_counts
        
        return stats


class CustomSignalGenerator(BaseSignalGenerator):
    """
    Template for custom signal generators
    Users can extend this class to create their own strategies
    """
    
    def __init__(self, config: TradingConfig, name: str = "Custom"):
        super().__init__(config, name)
        
        # Define your custom parameters here
        self.custom_param1 = 20
        self.custom_param2 = 0.5
        
        # Define required indicators
        self.required_indicators = ["close", "volume"]  # Minimum required
    
    def generate_signals(self, data: pd.DataFrame, symbol: str, 
                        current_position: Optional[dict] = None) -> List[TradingSignal]:
        """
        Implement your custom signal logic here
        
        Example structure:
        1. Check data sufficiency
        2. Validate indicators
        3. Apply your strategy logic
        4. Return list of TradingSignal objects
        """
        
        if not self.is_data_sufficient(data):
            return []
        
        if not self.validate_indicators(data, self.required_indicators):
            return []
        
        signals = []
        current_bar = data.iloc[-1]
        
        # Example: Simple moving average crossover
        if "SMA_20" in data.columns and "SMA_50" in data.columns:
            sma_20_current = current_bar["SMA_20"]
            sma_50_current = current_bar["SMA_50"]
            
            # Previous values for crossover detection
            if len(data) >= 2:
                prev_bar = data.iloc[-2]
                sma_20_prev = prev_bar["SMA_20"]
                sma_50_prev = prev_bar["SMA_50"]
                
                # Bullish crossover
                if (sma_20_prev <= sma_50_prev and sma_20_current > sma_50_current and 
                    current_position is None):
                    
                    signal = TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.ENTRY_LONG,
                        timestamp=current_bar.name,
                        strength=0.7,
                        entry_price=current_bar["close"],
                        confidence=0.6,
                        reason="SMA bullish crossover",
                        strategy_data={"entry_type": "sma_crossover"}
                    )
                    signals.append(signal)
        
        return signals


# ===== SIGNAL UTILITIES =====

def signals_to_dataframe(signals: List[TradingSignal]) -> pd.DataFrame:
    """Convert list of signals to DataFrame for analysis"""
    
    if not signals:
        return pd.DataFrame()
    
    data = []
    for signal in signals:
        row = {
            "symbol": signal.symbol,
            "signal_type": signal.signal_type.value,
            "timestamp": signal.timestamp,
            "strength": signal.strength,
            "confidence": signal.confidence,
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "reason": signal.reason,
            "strategy": signal.strategy_data.get("strategy_name", "unknown")
        }
        data.append(row)
    
    return pd.DataFrame(data)


def analyze_signal_performance(signals: List[TradingSignal], 
                             price_data: Dict[str, pd.DataFrame],
                             holding_period_days: int = 10) -> pd.DataFrame:
    """
    Analyze how signals would have performed
    This is a simplified version - full analysis would need actual trade execution
    """
    
    results = []
    
    for signal in signals:
        if not signal.is_entry_signal() or signal.symbol not in price_data:
            continue
        
        symbol_data = price_data[signal.symbol]
        
        # Find the signal date in the data
        signal_date = signal.timestamp
        
        try:
            # Get entry price (next day's open after signal)
            entry_idx = symbol_data.index.get_indexer([signal_date], method='bfill')[0]
            if entry_idx >= len(symbol_data) - 1:
                continue
            
            entry_price = symbol_data.iloc[entry_idx + 1]["open"]
            
            # Get exit price (after holding period or stop loss)
            exit_idx = min(entry_idx + holding_period_days, len(symbol_data) - 1)
            exit_price = symbol_data.iloc[exit_idx]["close"]
            
            # Check if stop loss was hit
            if signal.stop_loss:
                period_data = symbol_data.iloc[entry_idx:exit_idx + 1]
                min_low = period_data["low"].min()
                if min_low <= signal.stop_loss:
                    exit_price = signal.stop_loss
            
            # Calculate return
            return_pct = (exit_price - entry_price) / entry_price * 100
            
            results.append({
                "symbol": signal.symbol,
                "signal_date": signal_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return_pct": return_pct,
                "strength": signal.strength,
                "confidence": signal.confidence,
                "strategy": signal.strategy_data.get("strategy_name", "unknown")
            })
            
        except (IndexError, KeyError):
            continue
    
    return pd.DataFrame(results)


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    # Test the signal system
    print("=== Testing Signal System ===")
    
    from config import TradingConfig
    import logging
    
    # Setup
    config = TradingConfig()
    signal_manager = SignalManager(config)
    
    # Create test data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    test_data = pd.DataFrame({
        'symbol': ['TEST'] * 100,
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100),
        'ADX': np.random.uniform(10, 30, 100),
        'HH': np.random.randn(100).cumsum() + 105,
        'LL': np.random.randn(100).cumsum() + 95,
        'Volume_SMA': np.random.randint(5000, 8000, 100)
    }, index=dates)
    
    # Test signal generation
    signals = signal_manager.generate_signals_for_symbol("TEST", test_data)
    print(f"Generated {len(signals)} signals")
    
    if signals:
        for signal in signals[:3]:  # Show first 3
            print(f"Signal: {signal.signal_type.value} for {signal.symbol} at {signal.entry_price:.2f}")
            print(f"  Strength: {signal.strength:.2f}, Confidence: {signal.confidence:.2f}")
            print(f"  Reason: {signal.reason}")
            print()
    
    # Test signal filtering and ranking
    if signals:
        strong_signals = signal_manager.filter_signals(signals, min_strength=0.5)
        ranked_signals = signal_manager.rank_signals(strong_signals)
        print(f"Filtered to {len(strong_signals)} strong signals")
    
    # Test signal statistics
    stats = signal_manager.get_signal_statistics()
    print(f"Signal statistics: {stats}")
    
    # Test DataFrame conversion
    if signals:
        signals_df = signals_to_dataframe(signals)
        print(f"Signals DataFrame shape: {signals_df.shape}")
        print(signals_df.head())
    
    print("\n=== Signal System Test Complete ===")


# ===== INTEGRATION NOTES =====
"""
To integrate this with your existing code:

1. Replace signal logic in backtest with:
   signal_manager = SignalManager(CONFIG)
   signals = signal_manager.generate_signals_for_symbol(symbol, df, current_position)

2. Replace entry/exit conditions with:
   entry_signals = [s for s in signals if s.is_entry_signal()]
   exit_signals = [s for s in signals if s.is_exit_signal()]

3. Use signal data for order placement:
   for signal in entry_signals:
       place_order(signal.symbol, signal.entry_price, signal.stop_loss)

4. Add custom strategies by extending CustomSignalGenerator:
   class MyStrategy(CustomSignalGenerator):
       def generate_signals(self, data, symbol, position):
           # Your custom logic here
           pass

5. Monitor signal performance:
   stats = signal_manager.get_signal_statistics()
   performance = analyze_signal_performance(signals, price_data)
"""