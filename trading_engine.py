"""
Main Trading Engine

Orchestrates all components: data, signals, portfolio, and order management.
Replaces the scattered logic from your original main function.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import time
import threading
from enum import Enum
from dataclasses import dataclass

from config import TradingConfig, get_config_for_mode
from logger import get_logger, LoggingContext, PerformanceLogger
from data_manager import DataManager
from signals import SignalManager, TradingSignal, SignalType
from portfolio import PortfolioManager


class TradingMode(Enum):
    """Trading modes"""
    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    LIVE = "LIVE"


class EngineState(Enum):
    """Engine state"""
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    ERROR = "ERROR"


@dataclass
class TradingSession:
    """Trading session information"""
    mode: TradingMode
    start_time: datetime
    end_time: Optional[datetime] = None
    total_signals: int = 0
    total_trades: int = 0
    session_pnl: float = 0.0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class TradingEngine:
    """
    Main trading engine that orchestrates all components
    """
    
    def __init__(self, config: TradingConfig, mode: TradingMode = TradingMode.PAPER):
        self.config = config
        self.mode = mode
        self.state = EngineState.STOPPED
        
        # Setup logging
        self.logger = get_logger("trading_engine")
        self.perf_logger = PerformanceLogger(self.logger)
        
        # Initialize components
        self.data_manager: Optional[DataManager] = None
        self.signal_manager: Optional[SignalManager] = None
        self.portfolio_manager: Optional[PortfolioManager] = None
        self.order_manager = None  # Live order manager for paper/live modes
        self.ib_app = None  # Will be set when connecting
        
        # Session tracking
        self.current_session: Optional[TradingSession] = None
        self.session_history: List[TradingSession] = []
        
        # Threading for live modes
        self._stop_event = threading.Event()
        self._engine_thread: Optional[threading.Thread] = None
        
        # Performance monitoring
        self._last_equity_update = None
        self._execution_stats = {
            "signals_processed": 0,
            "orders_placed": 0,
            "orders_failed": 0,
            "data_updates": 0
        }
        
        self.logger.info(f"Trading engine initialized", mode=mode.value)
    
    def initialize_components(self, ib_app=None):
        """Initialize all trading components"""
        with LoggingContext(self.logger, "Initializing trading components"):
            try:
                # CRITICAL FIX: Store IB app connection
                if ib_app:
                    self.ib_app = ib_app  # ← This was missing or not working!
                    self.logger.info("IB app connection established for trading engine")
                else:
                    self.logger.warning("No IB app provided - will use mock data")
                    self.ib_app = None
                
                # Initialize data manager with IB connection
                self.data_manager = DataManager(self.config, self.ib_app)
                if self.ib_app:
                    self.data_manager.set_ib_app(self.ib_app)
                
                # Initialize signal manager
                self.signal_manager = SignalManager(self.config)
                
                # Initialize portfolio manager  
                self.portfolio_manager = PortfolioManager(self.config)
                
                # Initialize performance logger
                self.perf_logger = PerformanceLogger(self.logger)
                
                self.logger.info("All components initialized successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to initialize components: {e}")
                self.state = EngineState.ERROR
                return False
    
    def start_session(self, session_name: str = None) -> bool:
        """Start a new trading session"""
        
        if self.state != EngineState.STOPPED:
            self.logger.error("Cannot start session: engine not stopped")
            return False
        
        try:
            self.state = EngineState.STARTING
            
            # End previous session if exists
            if self.current_session and not self.current_session.end_time:
                self._end_current_session()
            
            # Start new session
            self.current_session = TradingSession(
                mode=self.mode,
                start_time=datetime.now(timezone.utc)
            )
            
            # Initialize components if not done
            if not self.data_manager:
                self.initialize_components()
            
            self.state = EngineState.RUNNING
            self.logger.info(f"Trading session started", mode=self.mode.value)
            
            return True
            
        except Exception as e:
            self.state = EngineState.ERROR
            self.logger.error(f"Failed to start session", error=str(e))
            return False
    
    def stop_session(self):
        """Stop the current trading session"""
        
        if self.state not in [EngineState.RUNNING, EngineState.ERROR]:
            return
        
        self.state = EngineState.STOPPING
        self.logger.info("Stopping trading session...")
        
        # Signal stop to live trading thread
        self._stop_event.set()
        
        # Wait for thread to finish
        if self._engine_thread and self._engine_thread.is_alive():
            self._engine_thread.join(timeout=10)
            if self._engine_thread.is_alive():
                self.logger.warning("Trading thread did not stop cleanly")
        
        # End current session
        self._end_current_session()
        
        # Clean up components
        if self.data_manager:
            self.data_manager.cleanup_old_data(0)  # Clean all cached data
        
        self.state = EngineState.STOPPED
        self.logger.info("Trading session stopped")
        
        # Force garbage collection
        import gc
        gc.collect()
    
    def _end_current_session(self):
        """End the current trading session"""
        if self.current_session:
            self.current_session.end_time = datetime.now(timezone.utc)
            
            # Calculate session P&L
            if self.portfolio_manager:
                current_equity = self.portfolio_manager.current_equity
                initial_equity = self.portfolio_manager.initial_equity
                self.current_session.session_pnl = current_equity - initial_equity
            
            # Add to session history
            self.session_history.append(self.current_session)
            
            # Keep history manageable
            if len(self.session_history) > 100:
                self.session_history = self.session_history[-50:]
    
    def run_backtest(self, symbols: List[str] = None, 
                start_date: str = None, end_date: str = None) -> Dict[str, any]:
        """
        Run historical backtest
        """
        
        if not self.start_session("backtest"):
            return {"error": "Failed to start backtest session"}
        
        try:
            with LoggingContext(self.logger, "Running backtest"):
                
                # Use config symbols if not provided
                if not symbols:
                    symbols = self.config.SYMBOLS
                
                self.logger.info(f"Backtest parameters: symbols={symbols}, start={start_date}, end={end_date}")
                
                # CRITICAL: Log IB connection status
                if self.ib_app:
                    self.logger.info("✅ USING REAL IB DATA - IB connection available")
                else:
                    self.logger.warning("⚠️ USING MOCK DATA - No IB connection")
                
                # Fetch historical data
                self.perf_logger.start_timer("data_fetch")
                
                if self.ib_app:
                    # Try to fetch from IB
                    self.logger.info("Attempting to fetch real market data from IB...")
                    try:
                        data_by_symbol = self.data_manager.get_multiple_symbols(symbols)
                        
                        # Validate we got real data
                        if data_by_symbol and any(not df.empty for df in data_by_symbol.values()):
                            self.logger.info("✅ Successfully fetched REAL IB data")
                            # Log data details
                            for symbol, df in data_by_symbol.items():
                                if not df.empty:
                                    start_actual = df.index.min().strftime('%Y-%m-%d')
                                    end_actual = df.index.max().strftime('%Y-%m-%d')
                                    self.logger.info(f"  {symbol}: {len(df)} bars from {start_actual} to {end_actual}")
                        else:
                            self.logger.error("❌ IB data fetch returned empty - falling back to mock")
                            data_by_symbol = self._generate_mock_data(symbols, start_date, end_date)
                            
                    except Exception as e:
                        self.logger.error(f"❌ IB data fetch failed: {e} - falling back to mock")
                        data_by_symbol = self._generate_mock_data(symbols, start_date, end_date)
                else:
                    # Generate mock data
                    self.logger.warning("⚠️ Generating MOCK data (not real market data)")
                    data_by_symbol = self._generate_mock_data(symbols, start_date, end_date)
                
                self.perf_logger.end_timer("data_fetch")
                
                if not data_by_symbol:
                    return {"error": "No data available for backtest"}
                
                # Run backtest simulation
                results = self._run_backtest_simulation(data_by_symbol, start_date, end_date)
                
                # Add data source info to results
                results["data_source"] = "IB_REAL" if self.ib_app else "MOCK"
                results["data_warning"] = None if self.ib_app else "⚠️ Using mock data - results not realistic"
                
                # Export results
                if self.portfolio_manager:
                    self.portfolio_manager.export_to_csv(f"{self.config.OUTPUT_DIR}/backtest")
                
                return results
                
        except Exception as e:
            self.logger.error(f"Backtest failed", error=str(e))
            return {"error": str(e)}
        
        finally:
            self.stop_session()

    
    def _run_backtest_simulation(self, data_by_symbol: Dict[str, pd.DataFrame], 
                                start_date: str = None, end_date: str = None) -> Dict[str, any]:
        """Run the actual backtest simulation"""
        
        # Clean data preparation - NO FORWARD FILLING OF OHLC
        cleaned_data = self._prepare_backtest_data(data_by_symbol, start_date, end_date)
        
        if not cleaned_data:
            raise ValueError("No valid data after cleaning")
        
        # Get all trading dates from actual data (not forward-filled)
        all_dates = set()
        for df in cleaned_data.values():
            all_dates.update(df.index)
        
        common_dates = sorted(all_dates)
        
        if len(common_dates) < self.config.WARMUP_BARS:
            raise ValueError(f"Insufficient data: {len(common_dates)} bars < {self.config.WARMUP_BARS} required")
        
        # Skip warmup period
        trading_dates = common_dates[self.config.WARMUP_BARS:]
        
        self.logger.info(f"Backtesting over {len(trading_dates)} trading days")
        
        # Main backtest loop
        for i, current_date in enumerate(trading_dates):
            
            if i % 50 == 0:  # Progress logging
                self.logger.info(f"Backtest progress: {i}/{len(trading_dates)} ({i/len(trading_dates)*100:.1f}%)")
            
            # Get current prices - ONLY from symbols with actual data on this date
            current_prices = {}
            for symbol, df in cleaned_data.items():
                if current_date in df.index:
                    close_price = df.loc[current_date, "close"]
                    if not pd.isna(close_price):  # Use pd.isna() instead of .isna()
                        current_prices[symbol] = close_price
            
            # Update portfolio equity
            self.portfolio_manager.update_equity(current_prices)
            
            # Check for exit signals first
            self._process_exit_signals(cleaned_data, current_date, i, trading_dates)
            
            # Check for entry signals
            self._process_entry_signals(cleaned_data, current_date, i, trading_dates)
            
            # Update session stats
            if self.current_session:
                self.current_session.total_trades = len(self.portfolio_manager.trade_history)
        
        # Calculate final results
        return self._compile_backtest_results()
    
    def _prepare_backtest_data(self, data_by_symbol: Dict[str, pd.DataFrame], 
                              start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """Prepare data for backtest - NO SHARED CALENDAR, symbol-specific trading only"""
        
        cleaned_data = {}
        
        for symbol, df in data_by_symbol.items():
            if df.empty:
                continue
            
            # Clean the data
            clean_df = df.copy()
            
            # Filter date range if specified
            if start_date:
                start_ts = pd.Timestamp(start_date)
                clean_df = clean_df[clean_df.index >= start_ts]
            
            if end_date:
                end_ts = pd.Timestamp(end_date)
                clean_df = clean_df[clean_df.index <= end_ts]
            
            # Validate adjusted data (basic check for splits)
            if not self._validate_adjusted_data(clean_df, symbol):
                self.logger.warning(f"Skipping {symbol}: Failed adjusted data validation")
                continue
            
            # Only keep rows with valid OHLC data - NO REINDEXING TO SHARED CALENDAR
            clean_df = clean_df.dropna(subset=["open", "high", "low", "close"])
            
            if len(clean_df) < self.config.WARMUP_BARS:
                self.logger.warning(f"Skipping {symbol}: Insufficient data ({len(clean_df)} bars)")
                continue
            
            # Keep data in its original form - each symbol trades on its own dates
            cleaned_data[symbol] = clean_df
        
        self.logger.info(f"Prepared data for {len(cleaned_data)} symbols with symbol-specific calendars")
        return cleaned_data
    
    def _validate_adjusted_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """Basic validation for adjusted data - detect obvious splits"""
        
        if len(df) < 10:
            return True  # Too little data to validate
        
        # Check for suspicious price gaps that might indicate unadjusted data
        close_prices = df["close"].dropna()
        
        if len(close_prices) < 5:
            return True
        
        # Calculate day-to-day price changes
        price_changes = close_prices.pct_change().abs()
        
        # Flag if we see multiple large gaps (>30%) which might be splits
        large_gaps = (price_changes > 0.3).sum()
        
        if large_gaps > 2:  # More than 2 large gaps suggests unadjusted data
            self.logger.warning(f"{symbol}: Detected {large_gaps} large price gaps - possible unadjusted data")
            return False
        
        return True
    
    def _process_exit_signals(self, data_by_symbol: Dict[str, pd.DataFrame], current_date: pd.Timestamp, 
                             current_idx: int = None, trading_dates: List = None):
        """Process exit signals for current positions"""
        
        positions_to_close = []
        
        for symbol in list(self.portfolio_manager.positions.keys()):
            if symbol not in data_by_symbol:
                continue
            
            position = self.portfolio_manager.positions[symbol]
            df = data_by_symbol[symbol]
            
            # Get data up to current date with valid OHLC only
            historical_data = df[df.index <= current_date].dropna(subset=["open", "high", "low", "close"])
            if len(historical_data) < 2:
                continue
            
            # Check if next day's bar exists and is valid
            next_date_idx = list(df.index).index(current_date) + 1
            if next_date_idx >= len(df):
                continue  # No next day data
            
            next_bar = df.iloc[next_date_idx]
            if (pd.isna(next_bar["open"]) or pd.isna(next_bar["high"]) or 
                pd.isna(next_bar["low"]) or pd.isna(next_bar["close"])):
                self.logger.debug(f"Skipping {symbol} exit: Next day OHLC data missing")
                continue  # Skip if next day OHLC is missing
            
            # Generate exit signals
            position_dict = {
                "entry_price": position.entry_price,
                "shares": position.shares,
                "entry_date": position.entry_date
            }
            
            signals = self.signal_manager.generate_signals_for_symbol(
                symbol, historical_data, position_dict
            )
            
            exit_signals = [s for s in signals if s.is_exit_signal()]
            
            if exit_signals:
                # Use first exit signal
                exit_signal = exit_signals[0]
                
                # Simulate stop loss execution
                if exit_signal.entry_price:  # Stop price
                    exit_price = self._simulate_stop_exit(
                        next_bar, exit_signal.entry_price
                    )
                    if exit_price:
                        positions_to_close.append((symbol, exit_price))
                        self.logger.debug(f"Exit signal triggered", 
                                        symbol=symbol, exit_price=exit_price)
        
        # Close positions with proper exit dates
        for symbol, exit_price in positions_to_close:
            position = self.portfolio_manager.positions[symbol]
            
            # Calculate exit date (next trading day)
            next_date_idx = list(data_by_symbol[symbol].index).index(current_date) + 1
            if next_date_idx < len(data_by_symbol[symbol]):
                exit_date = data_by_symbol[symbol].index[next_date_idx]
            else:
                exit_date = current_date
            
            # Manual position closing for backtest with correct dates
            fees = position.shares * self.config.COMMISSION_PER_SHARE
            proceeds = position.shares * exit_price - fees
            
            # Close position manually
            position.close_position(exit_price, exit_date, fees)
            
            # Update portfolio
            self.portfolio_manager.cash += proceeds
            self.portfolio_manager.closed_positions.append(position)
            
            # Create enhanced trade record
            trade_record = {
                "symbol": symbol,
                "strategy": position.strategy_name,
                "side": "Long",
                "entry_date": position.entry_date.strftime('%Y-%m-%d'),
                "entry_time": "09:30:00",  # Market open
                "exit_date": exit_date.strftime('%Y-%m-%d'),
                "exit_time": "16:00:00",   # Market close
                "shares": position.shares,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "realized_pnl": position.realized_pnl,
                "return_pct": (exit_price - position.entry_price) / position.entry_price * 100,
                "holding_days": self._calculate_holding_days(position.entry_date, exit_date),
                "fees_paid": position.fees_paid,
                "reason": "Stop Loss",
                "signal_strength": position.signal_strength,
                "signal_confidence": position.signal_confidence,
                "equity_at_entry": position.entry_equity,
                "position_size_pct": (position.shares * position.entry_price) / position.entry_equity * 100,
                "trade_value": position.shares * position.entry_price
            }
            
            self.portfolio_manager.trade_history.append(trade_record)
            del self.portfolio_manager.positions[symbol]
            
            self._execution_stats["orders_placed"] += 1
    
    def _process_entry_signals(self, data_by_symbol: Dict[str, pd.DataFrame], current_date: pd.Timestamp,
                              current_idx: int = None, trading_dates: List = None):
        """Process entry signals for new positions"""
        
        # Check position limits
        if len(self.portfolio_manager.positions) >= self.config.MAX_POSITIONS:
            return
        
        entry_candidates = []
        
        for symbol, df in data_by_symbol.items():
            # Skip if already have position
            if symbol in self.portfolio_manager.positions:
                continue
            
            # Get data up to current date
            historical_data = df[df.index <= current_date].dropna(subset=["open", "high", "low", "close"])
            if len(historical_data) < self.config.WARMUP_BARS:
                continue
            
            # Check if next day's bar exists and is valid - CRITICAL CHANGE
            next_date_idx = list(df.index).index(current_date) + 1
            if next_date_idx >= len(df):
                continue  # No next day data
            
            next_bar = df.iloc[next_date_idx]
            if (pd.isna(next_bar["open"]) or pd.isna(next_bar["high"]) or 
                pd.isna(next_bar["low"]) or pd.isna(next_bar["close"])):
                self.logger.debug(f"Skipping {symbol} entry: Next day OHLC data missing")
                continue  # Skip if next day OHLC is missing
            
            # Generate entry signals
            signals = self.signal_manager.generate_signals_for_symbol(
                symbol, historical_data, None
            )
            
            entry_signals = [s for s in signals if s.is_entry_signal()]
            
            if entry_signals:
                # Use strongest signal
                best_signal = max(entry_signals, key=lambda s: s.strength)
                
                # Simulate stop entry execution
                if best_signal.entry_price:  # Stop price
                    entry_price = self._simulate_stop_entry(
                        next_bar, best_signal.entry_price
                    )
                    if entry_price:
                        entry_candidates.append((symbol, best_signal, entry_price))
        
        # Sort by signal strength and add positions
        entry_candidates.sort(key=lambda x: x[1].strength, reverse=True)
        
        for symbol, signal, entry_price in entry_candidates:
            # Check if we can still add positions
            if len(self.portfolio_manager.positions) >= self.config.MAX_POSITIONS:
                break
            
            # Calculate position size
            shares = self.portfolio_manager.position_sizer.calculate_position_size(
                symbol, entry_price, self.portfolio_manager.current_equity,
                {"signal_strength": signal.strength, "stop_loss": signal.stop_loss}
            )
            
            if shares > 0:
                # Check portfolio limits
                position_cost = shares * entry_price
                can_add, reason = self.portfolio_manager.risk_manager.check_position_limits(
                    self.portfolio_manager.positions, position_cost, 
                    self.portfolio_manager.current_equity
                )
                
                if can_add:
                    # Add position with proper date handling
                    strategy_data = {
                        "strategy_name": signal.strategy_data.get("strategy_name", "unknown"),
                        "signal_strength": signal.strength,
                        "signal_confidence": signal.confidence,
                        "stop_loss": signal.stop_loss
                    }
                    
                    # Create position with specific entry date
                    position_cost = shares * entry_price
                    fees = shares * self.config.COMMISSION_PER_SHARE
                    
                    # Manual position creation for backtest with correct dates
                    from portfolio import Position, PositionStatus
                    
                    # Use current date or next trading date for entry
                    entry_date = current_date
                    if trading_dates and current_idx is not None and current_idx + 1 < len(trading_dates):
                        entry_date = trading_dates[current_idx + 1]
                    
                    # Ensure entry_date is timezone-naive for consistency
                    if hasattr(entry_date, 'tz') and entry_date.tz is not None:
                        entry_date = entry_date.tz_localize(None)
                    
                    position = Position(
                        symbol=symbol,
                        shares=shares,
                        entry_price=entry_price,
                        entry_date=entry_date,
                        entry_equity=self.portfolio_manager.current_equity,
                        strategy_name=strategy_data["strategy_name"],
                        signal_strength=signal.strength,
                        signal_confidence=signal.confidence,
                        stop_loss=signal.stop_loss,
                        fees_paid=fees
                    )
                    
                    # Add to portfolio manually for backtest
                    self.portfolio_manager.positions[symbol] = position
                    self.portfolio_manager.cash -= position_cost + fees
                    
                    self._execution_stats["orders_placed"] += 1
                    self.logger.debug(f"Entry position added", 
                                    symbol=symbol, shares=shares, entry_price=entry_price,
                                    entry_date=position.entry_date.strftime('%Y-%m-%d'))
                else:
                    self._execution_stats["orders_failed"] += 1
    
    def _calculate_holding_days(self, entry_date, exit_date) -> int:
        """Calculate holding days handling timezone differences"""
        try:
            # Convert both to pandas Timestamp to normalize timezones
            entry_ts = pd.Timestamp(entry_date)
            exit_ts = pd.Timestamp(exit_date)
            
            # If they have different timezone info, make them both naive
            if entry_ts.tz is not None:
                entry_ts = entry_ts.tz_localize(None)
            if exit_ts.tz is not None:
                exit_ts = exit_ts.tz_localize(None)
                
            return (exit_ts - entry_ts).days
        except Exception as e:
            self.logger.warning(f"Error calculating holding days: {e}")
            return 1  # Default to 1 day
    
    def _simulate_stop_entry(self, next_bar: pd.Series, stop_price: float) -> Optional[float]:
        """Simulate stop entry order execution"""
        open_price = next_bar["open"]
        high_price = next_bar["high"]
        
        # Entry triggered if high >= stop price
        if high_price >= stop_price:
            # Fill at better of stop price or open
            fill_price = max(stop_price, open_price)
            # Add slippage
            return fill_price + self.config.SLIPPAGE_PER_SHARE
        
        return None
    
    def _simulate_stop_exit(self, next_bar: pd.Series, stop_price: float) -> Optional[float]:
        """Simulate stop exit order execution"""
        open_price = next_bar["open"]
        low_price = next_bar["low"]
        
        # Exit triggered if low <= stop price
        if low_price <= stop_price:
            # Fill at worse of stop price or open
            fill_price = min(stop_price, open_price)
            # Subtract slippage
            return fill_price - self.config.SLIPPAGE_PER_SHARE
        
        return None
    
    def _compile_backtest_results(self) -> Dict[str, any]:
        """Compile comprehensive backtest results"""
        
        if not self.portfolio_manager:
            return {"error": "No portfolio manager"}
        
        # Get portfolio summary and performance metrics
        summary = self.portfolio_manager.get_portfolio_summary()
        performance = self.portfolio_manager.get_performance_metrics()
        trades_df = self.portfolio_manager.get_trades_dataframe()
        
        # Calculate additional comprehensive statistics
        comprehensive_stats = self._calculate_comprehensive_stats(trades_df, summary, performance)
        
        # Compile results
        results = {
            "summary": summary,
            "performance": performance,
            "comprehensive_stats": comprehensive_stats,
            "execution_stats": self._execution_stats.copy(),
            "session_info": {
                "mode": self.mode.value,
                "start_time": self.current_session.start_time if self.current_session else None,
                "symbols_traded": len(set(trades_df["symbol"])) if not trades_df.empty else 0,
                "total_signals": self.current_session.total_signals if self.current_session else 0
            }
        }
        
        # Add trade details if requested
        if not trades_df.empty:
            results["trades"] = trades_df.to_dict("records")
        
        self.logger.info(f"Backtest completed", 
                        total_return_pct=summary["equity"]["total_return_pct"],
                        total_trades=summary["positions"]["closed"],
                        win_rate=summary["trades"].get("win_rate", 0))
        
        return results
    
    def _calculate_comprehensive_stats(self, trades_df: pd.DataFrame, summary: Dict, performance: Dict) -> Dict:
        """Calculate comprehensive trading statistics similar to RealTest format"""
        
        if trades_df.empty:
            return {}
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df["realized_pnl"] > 0])
        losing_trades = len(trades_df[trades_df["realized_pnl"] < 0])
        
        # Win/Loss statistics
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        wins_df = trades_df[trades_df["realized_pnl"] > 0]
        losses_df = trades_df[trades_df["realized_pnl"] < 0]
        
        avg_win_pct = wins_df["return_pct"].mean() if not wins_df.empty else 0
        avg_loss_pct = abs(losses_df["return_pct"].mean()) if not losses_df.empty else 0
        
        # Holding periods
        avg_win_days = wins_df["holding_days"].mean() if not wins_df.empty else 0
        avg_loss_days = losses_df["holding_days"].mean() if not losses_df.empty else 0
        
        # Profit factor
        gross_profit = wins_df["realized_pnl"].sum() if not wins_df.empty else 0
        gross_loss = abs(losses_df["realized_pnl"].sum()) if not losses_df.empty else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        
        # Expectancy
        expectancy = (win_rate / 100 * avg_win_pct) - ((100 - win_rate) / 100 * avg_loss_pct)
        
        # Exposure statistics
        if "equity_at_entry" in trades_df.columns and "trade_value" in trades_df.columns:
            exposure_pcts = (trades_df["trade_value"] / trades_df["equity_at_entry"] * 100)
            avg_exposure = exposure_pcts.mean()
            max_exposure = exposure_pcts.max()
        else:
            avg_exposure = 0
            max_exposure = 0
        
        # Calculate proper annualized ROR
        initial_equity = self.config.ACCOUNT_SIZE
        final_equity = summary["equity"]["current"]
        
        # Date range calculation
        if not trades_df.empty:
            start_date = pd.to_datetime(trades_df["entry_date"]).min()
            end_date = pd.to_datetime(trades_df["exit_date"]).max()
            date_range_str = f"{start_date.strftime('%m/%d/%y')} - {end_date.strftime('%m/%d/%y')}"
            
            # Calculate total time period in years
            total_days = (end_date - start_date).days
            years = total_days / 365.25  # Account for leap years
            
            # Compound Annual Growth Rate (CAGR) - this is the proper ROR
            if years > 0 and initial_equity > 0:
                ror_annual = ((final_equity / initial_equity) ** (1 / years) - 1) * 100
            else:
                ror_annual = 0
        else:
            date_range_str = "N/A"
            ror_annual = 0
            years = 0
        
        # MAR ratio (Annualized Return / Max Drawdown)
        max_drawdown = abs(performance.get("max_drawdown", 0)) * 100
        mar_ratio = (ror_annual / max_drawdown) if max_drawdown > 0 else float("inf")
        
        # Calculate periods (number of equity curve data points)
        equity_df = self.portfolio_manager.get_equity_curve()
        periods = len(equity_df) if not equity_df.empty else 0
        
        # Alternative periods calculation using actual trading days
        if not trades_df.empty:
            # Count unique trading dates
            all_dates = set()
            all_dates.update(pd.to_datetime(trades_df["entry_date"]))
            all_dates.update(pd.to_datetime(trades_df["exit_date"]))
            # Estimate trading days (roughly 252 trading days per year)
            trading_days = int(years * 252) if years > 0 else len(all_dates)
            periods = max(periods, trading_days)
        
        return {
            "test_name": "ADX Breakout Strategy",
            "date_range": date_range_str,
            "periods": periods,
            "net_profit": summary["equity"]["total_return"],
            "compound": True,  # Assuming compound returns
            "ror": ror_annual,  # This is now the proper annualized ROR
            "max_dd": max_drawdown,
            "mar": mar_ratio,
            "trades": total_trades,
            "pct_wins": win_rate,
            "expectancy": expectancy,
            "avg_win": avg_win_pct,
            "avg_loss": avg_loss_pct,
            "win_len": avg_win_days,
            "loss_len": avg_loss_days,
            "profit_factor": profit_factor,
            "sharpe": performance.get("sharpe_ratio", 0),
            "avg_exp": avg_exposure,
            "max_exp": max_exposure,
            "years": years,  # For debugging
            "total_return_pct": summary["equity"]["total_return_pct"]  # Total return (not annualized)
        }
    
    def run_live_trading(self, update_interval_sec: int = 300):
        """
        Run live trading (paper or real)
        
        Args:
            update_interval_sec: How often to check for signals (seconds)
        """
        
        if not self.start_session("live_trading"):
            return
        
        self.logger.info(f"Starting live trading", mode=self.mode.value, 
                        interval_sec=update_interval_sec)
        
        # Start trading thread
        self._stop_event.clear()
        self._engine_thread = threading.Thread(
            target=self._live_trading_loop,
            args=(update_interval_sec,),
            name="TradingEngine",
            daemon=True
        )
        self._engine_thread.start()
    
    def _live_trading_loop(self, update_interval_sec: int):
        """Main live trading loop"""
        
        try:
            while not self._stop_event.is_set() and self.state == EngineState.RUNNING:
                
                try:
                    # Update data
                    self._update_live_data()
                    
                    # Process signals and manage portfolio
                    self._process_live_signals()
                    
                    # Update portfolio and check alerts
                    self._update_portfolio_status()
                    
                    # Log status periodically
                    if self._execution_stats["data_updates"] % 10 == 0:
                        self._log_live_status()
                    
                except Exception as e:
                    self.logger.error(f"Error in trading loop", error=str(e))
                    if self.current_session:
                        self.current_session.errors.append(str(e))
                
                # Wait for next iteration
                if not self._stop_event.wait(timeout=update_interval_sec):
                    continue
                else:
                    break
                    
        except Exception as e:
            self.logger.error(f"Fatal error in trading loop", error=str(e))
            self.state = EngineState.ERROR
        
        finally:
            self.logger.info("Live trading loop ended")
    
    def _update_live_data(self):
        """Update data for live trading"""
        if not self.data_manager:
            return
        
        # Refresh data for all symbols
        symbols_to_update = self.config.SYMBOLS
        
        # Also update symbols we have positions in
        if self.portfolio_manager:
            symbols_to_update.extend(self.portfolio_manager.positions.keys())
        
        # Remove duplicates
        symbols_to_update = list(set(symbols_to_update))
        
        # Update data
        for symbol in symbols_to_update:
            try:
                self.data_manager.get_latest_data(symbol, force_refresh=True)
            except Exception as e:
                self.logger.warning(f"Failed to update data for {symbol}", error=str(e))
        
        self._execution_stats["data_updates"] += 1
    
    def _process_live_signals(self):
        """Process signals in live trading"""
        if not all([self.data_manager, self.signal_manager, self.portfolio_manager]):
            return
        
        # Get current data for all symbols
        data_by_symbol = {}
        for symbol in self.config.SYMBOLS:
            df = self.data_manager.cache.get(symbol)
            if df is not None and not df.empty:
                data_by_symbol[symbol] = df
        
        if not data_by_symbol:
            return
        
        # Generate signals for portfolio
        current_positions = {
            symbol: {"entry_price": pos.entry_price, "shares": pos.shares}
            for symbol, pos in self.portfolio_manager.positions.items()
        }
        
        all_signals = self.signal_manager.generate_signals_for_portfolio(
            data_by_symbol, current_positions
        )
        
        if all_signals:
            total_signals = sum(len(signals) for signals in all_signals.values())
            self.logger.info(f"Generated {total_signals} signals across {len(all_signals)} symbols")
            
            # Process signals (in live mode, this would place actual orders)
            self._execute_live_signals(all_signals)
            
            if self.current_session:
                self.current_session.total_signals += total_signals
    
    def _execute_live_signals(self, signals_by_symbol: Dict[str, List[TradingSignal]]):
        """Execute signals in live trading (placeholder for order placement)"""
        
        for symbol, signals in signals_by_symbol.items():
            for signal in signals:
                self.logger.info(f"Signal ready for execution", 
                               symbol=signal.symbol,
                               signal_type=signal.signal_type.value,
                               entry_price=signal.entry_price,
                               strength=signal.strength)
                
                # In live trading, this would interface with order manager
                # For now, just log the signal
                self._execution_stats["signals_processed"] += 1
    
    def _update_portfolio_status(self):
        """Update portfolio status and check alerts"""
        if not self.portfolio_manager:
            return
        
        # Get current prices (would come from real-time data in live mode)
        current_prices = {}
        current_data = {}
        
        for symbol in self.portfolio_manager.positions.keys():
            df = self.data_manager.cache.get(symbol)
            if df is not None and not df.empty:
                current_prices[symbol] = df.iloc[-1]["close"]
                current_data[symbol] = df
        
        # Update equity
        if current_prices:
            self.portfolio_manager.update_equity(current_prices)
        
        # Update trailing stops to match backtest behavior
        if self.order_manager and current_data:
            try:
                updates = self.order_manager.update_trailing_stops(current_data)
                if updates > 0:
                    self.logger.info(f"Updated {updates} trailing stops")
            except Exception as e:
                self.logger.error(f"Failed to update trailing stops: {e}")
        
        # Check alerts
        alerts = self.portfolio_manager.check_position_alerts(current_prices)
        for alert in alerts:
            self.logger.warning(f"Position alert", alert_type=alert["type"], 
                              symbol=alert["symbol"], message=alert["message"])
        
        # Update trailing stops in portfolio manager too
        if current_prices:
            self.portfolio_manager.rebalance_stops(current_prices)
    
    def _log_live_status(self):
        """Log current trading status"""
        if not self.portfolio_manager:
            return
        
        summary = self.portfolio_manager.get_portfolio_summary()
        
        self.logger.info(f"Trading status", 
                        equity=summary["equity"]["current"],
                        positions=summary["positions"]["open"],
                        cash_pct=summary["risk"].get("cash_pct", 0),
                        total_return_pct=summary["equity"]["total_return_pct"])
    
    def _generate_mock_data(
    self,
    symbols: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
        """Generate mock data for testing without IB connection"""
        # FIX: Use provided date range or reasonable defaults
        if not start_date:
            start_date = "2023-01-01"
        if not end_date:
            end_date = "2024-12-31"
        
        self.logger.info(f"Generating mock data for date range: {start_date} to {end_date}")
        
        data_by_symbol = {}
        
        for symbol in symbols[:5]:  # Limit to 5 symbols for testing
            # Generate date range from parameters
            dates = pd.date_range(start=start_date, end=end_date, freq="D")
            n = len(dates)
            
            if n == 0:
                self.logger.warning(f"Invalid date range: {start_date} to {end_date}")
                continue
            
            # Random walk for prices
            returns = np.random.normal(0.0005, 0.02, n)  # Slight positive drift
            prices = 100 * np.exp(np.cumsum(returns))
            
            # OHLC data
            df = pd.DataFrame({
                "symbol": symbol,
                "open": prices * (1 + np.random.normal(0, 0.01, n)),
                "high": prices * (1 + np.abs(np.random.normal(0, 0.015, n))),
                "low": prices * (1 - np.abs(np.random.normal(0, 0.015, n))),
                "close": prices,
                "volume": np.random.randint(100000, 1000000, n)
            }, index=dates)
            
            # Add indicators using the data manager's calculator
            df = self.data_manager.indicator_calc.add_all_indicators(df)
            
            data_by_symbol[symbol] = df
        
        self.logger.info(f"Generated mock data for {len(data_by_symbol)} symbols")
        return data_by_symbol
    
    def get_engine_status(self) -> Dict[str, any]:
        """Get current engine status"""
        
        status = {
            "state": self.state.value,
            "mode": self.mode.value,
            "components_initialized": all([
                self.data_manager is not None,
                self.signal_manager is not None,
                self.portfolio_manager is not None
            ]),
            "execution_stats": self._execution_stats.copy()
        }
        
        if self.current_session:
            status["current_session"] = {
                "start_time": self.current_session.start_time,
                "duration_minutes": (datetime.now(timezone.utc) - self.current_session.start_time).total_seconds() / 60,
                "total_signals": self.current_session.total_signals,
                "total_trades": self.current_session.total_trades,
                "errors": len(self.current_session.errors)
            }
        
        if self.portfolio_manager:
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            status["portfolio"] = {
                "equity": portfolio_summary["equity"]["current"],
                "return_pct": portfolio_summary["equity"]["total_return_pct"],
                "positions": portfolio_summary["positions"]["open"],
                "cash": portfolio_summary["equity"]["cash"]
            }
        
        return status


# ===== TRADING ENGINE FACTORY =====

def create_trading_engine(mode: str = "paper", config_overrides: Dict = None) -> TradingEngine:
    """
    Factory function to create a trading engine
    
    Args:
        mode: "paper", "live", or "backtest"
        config_overrides: Dictionary of config values to override
        
    Returns:
        Configured TradingEngine instance
    """
    
    # Get appropriate config for mode
    config = get_config_for_mode(mode)
    
    # Apply overrides
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Map string mode to enum
    mode_enum = {
        "paper": TradingMode.PAPER,
        "live": TradingMode.LIVE,
        "backtest": TradingMode.BACKTEST
    }.get(mode.lower(), TradingMode.PAPER)
    
    engine = TradingEngine(config, mode_enum)
    return engine


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    # Test the trading engine
    print("=== Testing Trading Engine ===")
    
    # Test backtest mode
    engine = create_trading_engine("backtest")
    engine.initialize_components()  # Without IB connection for testing
    
    # Run a simple backtest
    results = engine.run_backtest(symbols=["AAPL", "MSFT", "GOOGL"])
    
    if "error" not in results:
        print(f"Backtest Results:")
        print(f"  Total Return: {results['summary']['equity']['total_return_pct']:.2f}%")
        print(f"  Total Trades: {results['summary']['positions']['closed']}")
        print(f"  Win Rate: {results['summary']['trades'].get('win_rate', 0):.1f}%")
        print(f"  Max Drawdown: {results['performance'].get('max_drawdown', 0)*100:.2f}%")
        print(f"  Sharpe Ratio: {results['performance'].get('sharpe_ratio', 0):.2f}")
    else:
        print(f"Backtest failed: {results['error']}")
    
    # Test engine status
    status = engine.get_engine_status()
    print(f"\nEngine Status: {status['state']}")
    print(f"Components Initialized: {status['components_initialized']}")
    
    print("\n=== Trading Engine Test Complete ===")


# ===== INTEGRATION NOTES =====
"""
To integrate this with your existing code:

1. Replace your main execution logic with:
   engine = create_trading_engine("paper")
   engine.initialize_components(ib_app)
   results = engine.run_backtest()

2. For live trading:
   engine = create_trading_engine("paper")
   engine.initialize_components(ib_app)
   engine.run_live_trading(update_interval_sec=300)

3. Monitor engine status:
   status = engine.get_engine_status()
   if status['state'] == 'ERROR':
       handle_error()

4. Customize behavior with config overrides:
   engine = create_trading_engine("paper", {
       "TRADE_PERCENT": 10.0,
       "MAX_POSITIONS": 5,
       "SYMBOLS": ["AAPL", "MSFT"]
   })

5. Access components directly if needed:
   portfolio_summary = engine.portfolio_manager.get_portfolio_summary()
   signals = engine.signal_manager.get_signal_statistics()
   data_quality = engine.data_manager.validate_data_quality()

6. Handle different trading modes:
   # Backtest mode
   engine = create_trading_engine("backtest")
   results = engine.run_backtest(symbols=["AAPL", "MSFT"], start_date="2023-01-01")
   
   # Paper trading mode
   engine = create_trading_engine("paper")
   engine.initialize_components(ib_app)
   engine.run_live_trading(update_interval_sec=300)
   
   # Live trading mode (when ready)
   engine = create_trading_engine("live", {"MAX_POSITIONS": 3, "TRADE_PERCENT": 5.0})
   engine.initialize_components(ib_app)
   engine.run_live_trading(update_interval_sec=600)

7. Error handling and monitoring:
   try:
       engine = create_trading_engine("paper")
       engine.initialize_components(ib_app)
       engine.run_live_trading()
   except Exception as e:
       logger.error(f"Trading engine failed: {e}")
   finally:
       engine.stop_session()

8. Session management:
   engine.start_session("my_trading_session")
   # ... trading operations ...
   status = engine.get_engine_status()
   engine.stop_session()

The trading engine provides a clean, professional interface that:
- Maintains all your original trading logic
- Adds robust error handling and monitoring
- Supports multiple trading modes seamlessly
- Provides comprehensive logging and analytics
- Makes it easy to extend and customize behavior
"""