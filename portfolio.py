"""
Position and Portfolio Management System

Handles position tracking, risk management, and portfolio-level calculations.
Replaces the scattered position logic from your original backtest code.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from config import TradingConfig
from logger import get_logger, LoggingContext


class PositionStatus(Enum):
    """Position status types"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PENDING = "PENDING"


@dataclass
class Position:
    """Individual position tracking"""
    symbol: str
    shares: int
    entry_price: float
    entry_date: datetime
    entry_equity: float  # Portfolio equity when position was opened
    
    # Optional fields
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_price: Optional[float] = None
    exit_date: Optional[datetime] = None
    status: PositionStatus = PositionStatus.OPEN
    
    # Strategy information
    strategy_name: str = "unknown"
    signal_strength: float = 0.0
    signal_confidence: float = 0.0
    
    # P&L tracking
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    
    # Metadata
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure dates are timezone-naive for consistency"""
        if hasattr(self.entry_date, 'tz') and self.entry_date.tz is not None:
            self.entry_date = self.entry_date.replace(tzinfo=None)
        if self.exit_date and hasattr(self.exit_date, 'tz') and self.exit_date.tz is not None:
            self.exit_date = self.exit_date.replace(tzinfo=None)
    
    def get_current_value(self, current_price: float) -> float:
        """Get current market value of position"""
        return self.shares * current_price
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """Get unrealized P&L"""
        if self.status != PositionStatus.OPEN:
            return 0.0
        return (current_price - self.entry_price) * self.shares
    
    def get_unrealized_pnl_pct(self, current_price: float) -> float:
        """Get unrealized P&L percentage"""
        if self.status != PositionStatus.OPEN or self.entry_price == 0:
            return 0.0
        return (current_price - self.entry_price) / self.entry_price * 100
    
    def get_total_pnl(self, current_price: Optional[float] = None) -> float:
        """Get total P&L (realized + unrealized)"""
        total = self.realized_pnl
        if self.status == PositionStatus.OPEN and current_price:
            total += self.get_unrealized_pnl(current_price)
        return total
    
    def get_holding_days(self, current_date: Optional[datetime] = None) -> int:
        """Get number of days position has been held"""
        if current_date is None:
            current_date = datetime.now()
        
        # Ensure both dates are timezone-naive
        entry_dt = self.entry_date
        current_dt = current_date
        
        if hasattr(entry_dt, 'tz') and entry_dt.tz is not None:
            entry_dt = entry_dt.replace(tzinfo=None)
        if hasattr(current_dt, 'tz') and current_dt.tz is not None:
            current_dt = current_dt.replace(tzinfo=None)
            
        return (current_dt - entry_dt).days
    
    def close_position(self, exit_price: float, exit_date: datetime, fees: float = 0.0):
        """Close the position"""
        self.exit_price = exit_price
        self.exit_date = exit_date
        self.status = PositionStatus.CLOSED
        self.realized_pnl = (exit_price - self.entry_price) * self.shares - fees
        self.fees_paid += fees
    
    def update_stop_loss(self, new_stop: float):
        """Update stop loss level"""
        self.stop_loss = new_stop
        self.metadata["stop_updates"] = self.metadata.get("stop_updates", 0) + 1
    
    def to_dict(self) -> Dict:
        """Convert position to dictionary"""
        return {
            "symbol": self.symbol,
            "shares": self.shares,
            "entry_price": self.entry_price,
            "entry_date": self.entry_date,
            "exit_price": self.exit_price,
            "exit_date": self.exit_date,
            "status": self.status.value,
            "realized_pnl": self.realized_pnl,
            "fees_paid": self.fees_paid,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "strategy_name": self.strategy_name,
            "holding_days": self.get_holding_days()
        }


class PositionSizer:
    """Handles position sizing calculations"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = get_logger("position_sizer")
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                              current_equity: float, strategy_data: Dict = None) -> int:
        """
        Calculate position size based on strategy rules
        
        Args:
            symbol: Symbol to trade
            entry_price: Intended entry price
            current_equity: Current portfolio equity
            strategy_data: Additional strategy-specific data
            
        Returns:
            Number of shares to trade
        """
        
        if entry_price <= 0 or current_equity <= 0:
            return 0
        
        # Calculate dollar amount to risk
        dollar_target = current_equity * (self.config.TRADE_PERCENT / 100.0)
        
        # Calculate base position size
        base_shares = int(dollar_target // entry_price)
        
        # Apply any strategy-specific adjustments
        if strategy_data:
            adjusted_shares = self._apply_strategy_adjustments(
                base_shares, symbol, entry_price, strategy_data
            )
        else:
            adjusted_shares = base_shares
        
        # Apply risk management constraints
        final_shares = self._apply_risk_constraints(
            adjusted_shares, symbol, entry_price, current_equity
        )
        
        self.logger.debug(f"Position size calculated", 
                         symbol=symbol, entry_price=entry_price,
                         base_shares=base_shares, final_shares=final_shares)
        
        return final_shares
    
    def _apply_strategy_adjustments(self, base_shares: int, symbol: str, 
                                  entry_price: float, strategy_data: Dict) -> int:
        """Apply strategy-specific position size adjustments"""
        
        adjusted_shares = base_shares
        
        # Adjust based on signal strength
        signal_strength = strategy_data.get("signal_strength", 1.0)
        if signal_strength < 0.5:
            adjusted_shares = int(adjusted_shares * 0.5)  # Reduce size for weak signals
        elif signal_strength > 0.8:
            adjusted_shares = int(adjusted_shares * 1.2)  # Increase size for strong signals
        
        # Adjust based on volatility (if available)
        atr = strategy_data.get("atr")
        if atr and entry_price > 0:
            volatility_ratio = atr / entry_price
            if volatility_ratio > 0.05:  # High volatility
                adjusted_shares = int(adjusted_shares * 0.8)
            elif volatility_ratio < 0.02:  # Low volatility
                adjusted_shares = int(adjusted_shares * 1.1)
        
        return max(0, adjusted_shares)
    
    def _apply_risk_constraints(self, shares: int, symbol: str, 
                               entry_price: float, current_equity: float) -> int:
        """Apply portfolio-level risk constraints"""
        
        # Minimum position size
        min_dollar_position = 1000  # Minimum $1000 position
        min_shares = max(1, int(min_dollar_position // entry_price))
        
        # Maximum position size (% of equity)
        max_position_pct = 20.0  # Max 20% of equity in one position
        max_dollar_position = current_equity * (max_position_pct / 100.0)
        max_shares = int(max_dollar_position // entry_price)
        
        # Apply constraints
        constrained_shares = max(min_shares, min(shares, max_shares))
        
        # Ensure we don't exceed available cash (with some buffer)
        available_cash = current_equity * 0.95  # Keep 5% cash buffer
        max_affordable_shares = int(available_cash // entry_price)
        
        final_shares = min(constrained_shares, max_affordable_shares)
        
        return max(0, final_shares)


class RiskManager:
    """Handles portfolio-level risk management"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = get_logger("risk_manager")
    
    def check_position_limits(self, current_positions: Dict[str, Position], 
                            new_position_cost: float, current_equity: float) -> Tuple[bool, str]:
        """
        Check if new position would violate limits
        
        Returns:
            (can_add_position, reason_if_not)
        """
        
        # Check maximum number of positions
        if len(current_positions) >= self.config.MAX_POSITIONS:
            return False, f"Maximum positions limit reached ({self.config.MAX_POSITIONS})"
        
        # Check maximum exposure
        current_exposure = sum(pos.shares * pos.entry_price for pos in current_positions.values())
        total_exposure = current_exposure + new_position_cost
        exposure_pct = (total_exposure / current_equity) * 100
        
        if exposure_pct > self.config.MAX_EXPOSURE_PCT:
            return False, f"Maximum exposure limit would be exceeded ({exposure_pct:.1f}% > {self.config.MAX_EXPOSURE_PCT}%)"
        
        # Check available cash
        required_cash = new_position_cost * 1.05  # 5% buffer
        if required_cash > current_equity:
            return False, f"Insufficient cash for position"
        
        return True, "Position limits OK"
    
    def calculate_portfolio_risk_metrics(self, positions: Dict[str, Position], 
                                       current_prices: Dict[str, float],
                                       current_equity: float) -> Dict[str, float]:
        """Calculate portfolio-level risk metrics"""
        
        if not positions:
            return {"total_exposure": 0.0, "exposure_pct": 0.0, "concentration_risk": 0.0}
        
        # Calculate exposure
        total_exposure = 0.0
        position_values = {}
        
        for symbol, position in positions.items():
            if position.status == PositionStatus.OPEN:
                current_price = current_prices.get(symbol, position.entry_price)
                value = position.get_current_value(current_price)
                total_exposure += value
                position_values[symbol] = value
        
        exposure_pct = (total_exposure / current_equity) * 100 if current_equity > 0 else 0
        
        # Calculate concentration risk (largest position as % of portfolio)
        concentration_risk = 0.0
        if position_values:
            largest_position = max(position_values.values())
            concentration_risk = (largest_position / current_equity) * 100 if current_equity > 0 else 0
        
        # Calculate sector concentration (if metadata available)
        sector_exposure = {}
        for symbol, position in positions.items():
            if position.status == PositionStatus.OPEN:
                sector = position.metadata.get("sector", "Unknown")
                current_price = current_prices.get(symbol, position.entry_price)
                value = position.get_current_value(current_price)
                sector_exposure[sector] = sector_exposure.get(sector, 0) + value
        
        max_sector_exposure_pct = 0.0
        if sector_exposure and current_equity > 0:
            max_sector_exposure = max(sector_exposure.values())
            max_sector_exposure_pct = (max_sector_exposure / current_equity) * 100
        
        return {
            "total_exposure": total_exposure,
            "exposure_pct": exposure_pct,
            "concentration_risk": concentration_risk,
            "max_sector_exposure_pct": max_sector_exposure_pct,
            "num_positions": len([p for p in positions.values() if p.status == PositionStatus.OPEN]),
            "cash_pct": max(0, 100 - exposure_pct)
        }
    
    def check_stop_loss_triggers(self, positions: Dict[str, Position], 
                                current_prices: Dict[str, float]) -> List[Tuple[str, Position]]:
        """Check which positions should be stopped out"""
        
        stop_triggers = []
        
        for symbol, position in positions.items():
            if position.status != PositionStatus.OPEN:
                continue
            
            current_price = current_prices.get(symbol)
            if current_price is None:
                continue
            
            # Check stop loss
            if position.stop_loss and current_price <= position.stop_loss:
                stop_triggers.append((symbol, position))
                self.logger.info(f"Stop loss triggered", symbol=symbol, 
                               current_price=current_price, stop_loss=position.stop_loss)
        
        return stop_triggers


class PortfolioManager:
    """Main portfolio management class"""
    
    def __init__(self, config: TradingConfig, initial_equity: float = None):
        self.config = config
        self.logger = get_logger("portfolio_manager")
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.initial_equity = initial_equity or config.ACCOUNT_SIZE
        self.current_equity = self.initial_equity
        self.cash = self.initial_equity
        
        # Components
        self.position_sizer = PositionSizer(config)
        self.risk_manager = RiskManager(config)
        
        # Performance tracking
        self.equity_history: List[Dict] = []
        self.trade_history: List[Dict] = []
        
        self.logger.info(f"Portfolio manager initialized", initial_equity=self.initial_equity)
    
    def add_position(self, symbol: str, shares: int, entry_price: float, 
                    strategy_data: Dict = None, fees: float = None) -> bool:
        """
        Add a new position to the portfolio
        
        Returns:
            True if position was added successfully
        """
        
        if fees is None:
            fees = shares * self.config.COMMISSION_PER_SHARE
        
        total_cost = shares * entry_price + fees
        
        # Check if we can add this position
        can_add, reason = self.risk_manager.check_position_limits(
            self.positions, total_cost, self.current_equity
        )
        
        if not can_add:
            self.logger.warning(f"Cannot add position for {symbol}: {reason}")
            return False
        
        # Create position
        position = Position(
            symbol=symbol,
            shares=shares,
            entry_price=entry_price,
            entry_date=datetime.now(timezone.utc),
            entry_equity=self.current_equity,
            strategy_name=strategy_data.get("strategy_name", "unknown") if strategy_data else "unknown",
            signal_strength=strategy_data.get("signal_strength", 0.0) if strategy_data else 0.0,
            signal_confidence=strategy_data.get("signal_confidence", 0.0) if strategy_data else 0.0,
            stop_loss=strategy_data.get("stop_loss") if strategy_data else None,
            take_profit=strategy_data.get("take_profit") if strategy_data else None,
            fees_paid=fees,
            metadata=strategy_data.copy() if strategy_data else {}
        )
        
        # Update portfolio state
        self.positions[symbol] = position
        self.cash -= total_cost
        
        # Log the addition
        self.logger.info(f"Added position", symbol=symbol, shares=shares, 
                        entry_price=entry_price, total_cost=total_cost)
        
        return True
    
    def close_position(self, symbol: str, exit_price: float, fees: float = None) -> bool:
        """Close an existing position"""
        
        if symbol not in self.positions:
            self.logger.warning(f"Cannot close position: {symbol} not found")
            return False
        
        position = self.positions[symbol]
        
        if fees is None:
            fees = position.shares * self.config.COMMISSION_PER_SHARE
        
        # Close the position
        exit_date = datetime.now(timezone.utc)
        position.close_position(exit_price, exit_date, fees)
        
        # Update portfolio state
        proceeds = position.shares * exit_price - fees
        self.cash += proceeds
        
        # Move to closed positions
        self.closed_positions.append(position)
        del self.positions[symbol]
        
        # Record enhanced trade (RealTest format)
        trade_record = {
            # Basic trade info
            "symbol": symbol,
            "strategy": position.strategy_name,
            "side": "Long",  # Currently only long trades
            "entry_date": position.entry_date.strftime('%Y-%m-%d'),
            "entry_time": position.entry_date.strftime('%H:%M:%S'),
            "exit_date": exit_date.strftime('%Y-%m-%d'), 
            "exit_time": exit_date.strftime('%H:%M:%S'),
            
            # Quantities and prices
            "shares": position.shares,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            
            # P&L and performance
            "realized_pnl": position.realized_pnl,
            "return_pct": (exit_price - position.entry_price) / position.entry_price * 100,
            "holding_days": position.get_holding_days(exit_date),
            "fees_paid": position.fees_paid,
            
            # Additional metadata
            "reason": "Stop Loss",  # Default exit reason
            "signal_strength": position.signal_strength,
            "signal_confidence": position.signal_confidence,
            "equity_at_entry": position.entry_equity,
            
            # Position sizing info
            "position_size_pct": (position.shares * position.entry_price) / position.entry_equity * 100,
            "trade_value": position.shares * position.entry_price
        }
        self.trade_history.append(trade_record)
        
        self.logger.info(f"Closed position", symbol=symbol, 
                        realized_pnl=position.realized_pnl, 
                        return_pct=trade_record["return_pct"])
        
        return True
    
    def update_equity(self, current_prices: Dict[str, float]):
        """Update current equity based on market prices"""
        
        # Calculate market value of open positions
        position_value = 0.0
        for symbol, position in self.positions.items():
            if position.status == PositionStatus.OPEN:
                current_price = current_prices.get(symbol, position.entry_price)
                position_value += position.get_current_value(current_price)
        
        # Update equity
        self.current_equity = self.cash + position_value
        
        # Record equity history
        equity_record = {
            "timestamp": datetime.now(timezone.utc),
            "equity": self.current_equity,
            "cash": self.cash,
            "position_value": position_value,
            "num_positions": len(self.positions)
        }
        self.equity_history.append(equity_record)
        
        # Keep history manageable
        if len(self.equity_history) > 10000:
            self.equity_history = self.equity_history[-5000:]
    
    def get_portfolio_summary(self, current_prices: Dict[str, float] = None) -> Dict[str, any]:
        """Get comprehensive portfolio summary"""
        
        if current_prices:
            self.update_equity(current_prices)
        
        # Basic metrics
        total_return = self.current_equity - self.initial_equity
        total_return_pct = (total_return / self.initial_equity) * 100
        
        # Position summary
        open_positions = len(self.positions)
        total_positions_ever = len(self.closed_positions) + open_positions
        
        # Trade statistics
        trade_stats = self._calculate_trade_statistics()
        
        # Risk metrics
        risk_metrics = {}
        if current_prices:
            risk_metrics = self.risk_manager.calculate_portfolio_risk_metrics(
                self.positions, current_prices, self.current_equity
            )
        
        return {
            "equity": {
                "current": self.current_equity,
                "initial": self.initial_equity,
                "total_return": total_return,
                "total_return_pct": total_return_pct,
                "cash": self.cash,
                "invested": self.current_equity - self.cash
            },
            "positions": {
                "open": open_positions,
                "closed": len(self.closed_positions),
                "total_ever": total_positions_ever
            },
            "trades": trade_stats,
            "risk": risk_metrics
        }
    
    def _calculate_trade_statistics(self) -> Dict[str, any]:
        """Calculate trading performance statistics"""
        
        if not self.trade_history:
            return {"total_trades": 0}
        
        trades_df = pd.DataFrame(self.trade_history)
        
        # Basic stats
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df["realized_pnl"] > 0])
        losing_trades = len(trades_df[trades_df["realized_pnl"] < 0])
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L stats
        total_pnl = trades_df["realized_pnl"].sum()
        avg_win = trades_df[trades_df["realized_pnl"] > 0]["realized_pnl"].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df["realized_pnl"] < 0]["realized_pnl"].mean() if losing_trades > 0 else 0
        
        # Profit factor
        gross_profit = trades_df[trades_df["realized_pnl"] > 0]["realized_pnl"].sum()
        gross_loss = abs(trades_df[trades_df["realized_pnl"] < 0]["realized_pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Holding period stats
        avg_holding_days = trades_df["holding_days"].mean()
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "avg_holding_days": avg_holding_days,
            "best_trade": trades_df["realized_pnl"].max() if total_trades > 0 else 0,
            "worst_trade": trades_df["realized_pnl"].min() if total_trades > 0 else 0
        }
    
    def get_position_summary(self, current_prices: Dict[str, float] = None) -> pd.DataFrame:
        """Get summary of current positions"""
        
        if not self.positions:
            return pd.DataFrame()
        
        position_data = []
        for symbol, position in self.positions.items():
            current_price = current_prices.get(symbol, position.entry_price) if current_prices else position.entry_price
            
            data = {
                "symbol": symbol,
                "shares": position.shares,
                "entry_price": position.entry_price,
                "current_price": current_price,
                "market_value": position.get_current_value(current_price),
                "unrealized_pnl": position.get_unrealized_pnl(current_price),
                "unrealized_pnl_pct": position.get_unrealized_pnl_pct(current_price),
                "stop_loss": position.stop_loss,
                "days_held": position.get_holding_days(),
                "strategy": position.strategy_name
            }
            position_data.append(data)
        
        return pd.DataFrame(position_data)
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get all completed trades as DataFrame"""
        return pd.DataFrame(self.trade_history) if self.trade_history else pd.DataFrame()
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        return pd.DataFrame(self.equity_history) if self.equity_history else pd.DataFrame()
    
    def calculate_drawdown(self) -> pd.Series:
        """Calculate portfolio drawdown"""
        if not self.equity_history:
            return pd.Series()
        
        equity_df = self.get_equity_curve()
        equity_series = equity_df.set_index("timestamp")["equity"]
        
        # Calculate running maximum
        running_max = equity_series.cummax()
        
        # Calculate drawdown
        drawdown = (equity_series - running_max) / running_max
        
        return drawdown
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        if not self.equity_history:
            return {}
        
        equity_df = self.get_equity_curve()
        equity_series = equity_df.set_index("timestamp")["equity"]
        
        # Basic returns
        total_return = (self.current_equity - self.initial_equity) / self.initial_equity
        
        # Calculate daily returns
        daily_returns = equity_series.pct_change().dropna()
        
        if len(daily_returns) == 0:
            return {"total_return": total_return}
        
        # Risk metrics
        volatility = daily_returns.std() * np.sqrt(252)  # Annualized
        sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
        
        # Drawdown metrics
        drawdown = self.calculate_drawdown()
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        # Calmar ratio
        calmar_ratio = (total_return * 252 / len(daily_returns)) / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            "total_return": total_return,
            "annual_return": total_return * 252 / len(daily_returns) if len(daily_returns) > 0 else 0,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "num_observations": len(daily_returns)
        }
    
    def rebalance_stops(self, current_prices: Dict[str, float], 
                       trail_pct: float = 0.05):
        """Update trailing stops for profitable positions"""
        
        for symbol, position in self.positions.items():
            if position.status != PositionStatus.OPEN:
                continue
            
            current_price = current_prices.get(symbol)
            if not current_price:
                continue
            
            # Only trail stops for profitable positions
            unrealized_pnl_pct = position.get_unrealized_pnl_pct(current_price)
            if unrealized_pnl_pct <= 0:
                continue
            
            # Calculate new trailing stop
            new_stop = current_price * (1 - trail_pct)
            
            # Only update if new stop is higher than current stop
            if not position.stop_loss or new_stop > position.stop_loss:
                old_stop = position.stop_loss
                position.update_stop_loss(new_stop)
                
                self.logger.info(f"Updated trailing stop", symbol=symbol,
                               old_stop=old_stop, new_stop=new_stop,
                               current_price=current_price)
    
    def check_position_alerts(self, current_prices: Dict[str, float]) -> List[Dict]:
        """Check for position alerts (stop losses, profit targets, etc.)"""
        
        alerts = []
        
        # Check stop loss triggers
        stop_triggers = self.risk_manager.check_stop_loss_triggers(self.positions, current_prices)
        for symbol, position in stop_triggers:
            alerts.append({
                "type": "STOP_LOSS",
                "symbol": symbol,
                "message": f"Stop loss triggered at {current_prices[symbol]:.2f}",
                "current_price": current_prices[symbol],
                "stop_level": position.stop_loss,
                "position": position
            })
        
        # Check profit targets
        for symbol, position in self.positions.items():
            if position.status != PositionStatus.OPEN:
                continue
            
            current_price = current_prices.get(symbol)
            if not current_price:
                continue
            
            # Check take profit
            if position.take_profit and current_price >= position.take_profit:
                alerts.append({
                    "type": "TAKE_PROFIT",
                    "symbol": symbol,
                    "message": f"Take profit triggered at {current_price:.2f}",
                    "current_price": current_price,
                    "target_level": position.take_profit,
                    "position": position
                })
            
            # Check for large unrealized gains/losses
            unrealized_pnl_pct = position.get_unrealized_pnl_pct(current_price)
            if unrealized_pnl_pct > 20:  # >20% gain
                alerts.append({
                    "type": "LARGE_GAIN",
                    "symbol": symbol,
                    "message": f"Large unrealized gain: {unrealized_pnl_pct:.1f}%",
                    "unrealized_pnl_pct": unrealized_pnl_pct,
                    "position": position
                })
            elif unrealized_pnl_pct < -15:  # >15% loss
                alerts.append({
                    "type": "LARGE_LOSS",
                    "symbol": symbol,
                    "message": f"Large unrealized loss: {unrealized_pnl_pct:.1f}%",
                    "unrealized_pnl_pct": unrealized_pnl_pct,
                    "position": position
                })
        
        return alerts
    
    def export_to_csv(self, output_dir: str = "output"):
        """Export portfolio data to CSV files"""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Export trades in RealTest format
        trades_df = self.get_trades_dataframe()
        if not trades_df.empty:
            # Create RealTest-compatible trade export
            realtest_trades = self._create_realtest_format(trades_df)
            
            # Export both formats
            trades_path = os.path.join(output_dir, "portfolio_trades.csv")
            trades_df.to_csv(trades_path, index=False)
            self.logger.info(f"Exported trades to {trades_path}")
            
            realtest_path = os.path.join(output_dir, "trades_realtest_format.csv")
            realtest_trades.to_csv(realtest_path, index=False)
            self.logger.info(f"Exported RealTest format trades to {realtest_path}")
        
        # Export equity curve
        equity_df = self.get_equity_curve()
        if not equity_df.empty:
            equity_path = os.path.join(output_dir, "portfolio_equity.csv")
            equity_df.to_csv(equity_path, index=False)
            self.logger.info(f"Exported equity curve to {equity_path}")
        
        # Export current positions
        positions_df = self.get_position_summary()
        if not positions_df.empty:
            positions_path = os.path.join(output_dir, "current_positions.csv")
            positions_df.to_csv(positions_path, index=False)
            self.logger.info(f"Exported positions to {positions_path}")
        
        # Export performance summary
        summary = self.get_portfolio_summary()
        summary_df = pd.DataFrame([summary])
        summary_path = os.path.join(output_dir, "portfolio_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        self.logger.info(f"Exported summary to {summary_path}")
    
    def _create_realtest_format(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Convert trades to RealTest-compatible format"""
        
        if trades_df.empty:
            return pd.DataFrame()
        
        # Create RealTest format columns
        realtest_data = []
        
        for idx, trade in trades_df.iterrows():
            # Calculate additional metrics
            profit_str = f"${trade['realized_pnl']:.2f}"
            pct_gain_str = f"{trade['return_pct']:.2f}%"
            
            # Estimate position size as percentage of equity
            position_size_str = f"{trade.get('position_size_pct', 0):.1f}%"
            
            realtest_record = {
                # Core trade identification
                "Trade": idx + 1,
                "Strategy": trade['strategy'],
                "Symbol": trade['symbol'],
                "Side": trade.get('side', 'Long'),
                
                # Entry details
                "DateIn": trade['entry_date'],
                "TimeIn": trade.get('entry_time', '09:30:00'),
                "QtyIn": trade['shares'],
                "PriceIn": f"{trade['entry_price']:.2f}",
                
                # Exit details
                "DateOut": trade['exit_date'],
                "TimeOut": trade.get('exit_time', '16:00:00'),
                "QtyOut": trade['shares'],
                "PriceOut": f"{trade['exit_price']:.2f}",
                
                # Performance metrics
                "Reason": trade.get('reason', 'Stop Loss'),
                "Bars": trade['holding_days'],
                "PctGain": pct_gain_str,
                "Profit": profit_str,
                
                # Risk metrics (placeholder values - would need tick data for accurate calculation)
                "PctMFE": f"{max(0, trade['return_pct']):.2f}%",  # Approximate
                "PctMAE": f"{min(0, trade['return_pct']):.2f}%",  # Approximate
                
                # Position sizing
                "Fraction": "1.0",  # Full position (not fractional)
                "Size": position_size_str,
                "Dividends": "$0.00"  # Not tracking dividends currently
            }
            
            realtest_data.append(realtest_record)
        
        return pd.DataFrame(realtest_data)
    
    def export_trade_analysis(self, output_dir: str = "output"):
        """Export detailed trade analysis"""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        trades_df = self.get_trades_dataframe()
        if trades_df.empty:
            return
        
        # Monthly performance
        trades_df['entry_month'] = pd.to_datetime(trades_df['entry_date']).dt.to_period('M')
        monthly_stats = trades_df.groupby('entry_month').agg({
            'realized_pnl': ['count', 'sum', 'mean'],
            'return_pct': ['mean', 'std'],
            'holding_days': 'mean'
        }).round(2)
        
        monthly_path = os.path.join(output_dir, "monthly_performance.csv")
        monthly_stats.to_csv(monthly_path)
        
        # Symbol performance
        symbol_stats = trades_df.groupby('symbol').agg({
            'realized_pnl': ['count', 'sum', 'mean'],
            'return_pct': ['mean', 'std'],
            'holding_days': 'mean'
        }).round(2)
        
        symbol_path = os.path.join(output_dir, "symbol_performance.csv")
        symbol_stats.to_csv(symbol_path)
        
        # Strategy performance  
        if 'strategy' in trades_df.columns:
            strategy_stats = trades_df.groupby('strategy').agg({
                'realized_pnl': ['count', 'sum', 'mean'],
                'return_pct': ['mean', 'std'],
                'holding_days': 'mean'
            }).round(2)
            
            strategy_path = os.path.join(output_dir, "strategy_performance.csv")
            strategy_stats.to_csv(strategy_path)
        
        self.logger.info(f"Exported detailed trade analysis to {output_dir}")


# ===== PORTFOLIO ANALYTICS =====

class PortfolioAnalytics:
    """Advanced analytics for portfolio performance"""
    
    def __init__(self, portfolio_manager: PortfolioManager):
        self.portfolio = portfolio_manager
        self.logger = get_logger("portfolio_analytics")
    
    def analyze_strategy_performance(self) -> pd.DataFrame:
        """Analyze performance by strategy"""
        
        trades_df = self.portfolio.get_trades_dataframe()
        if trades_df.empty:
            return pd.DataFrame()
        
        strategy_stats = trades_df.groupby("strategy_name").agg({
            "realized_pnl": ["count", "sum", "mean"],
            "return_pct": ["mean", "std"],
            "holding_days": "mean"
        }).round(2)
        
        # Flatten column names
        strategy_stats.columns = ["_".join(col).strip() for col in strategy_stats.columns]
        
        # Calculate win rates
        win_rates = trades_df[trades_df["realized_pnl"] > 0].groupby("strategy_name").size() / trades_df.groupby("strategy_name").size() * 100
        strategy_stats["win_rate"] = win_rates.fillna(0).round(1)
        
        return strategy_stats.reset_index()
    
    def analyze_symbol_performance(self) -> pd.DataFrame:
        """Analyze performance by symbol"""
        
        trades_df = self.portfolio.get_trades_dataframe()
        if trades_df.empty:
            return pd.DataFrame()
        
        symbol_stats = trades_df.groupby("symbol").agg({
            "realized_pnl": ["count", "sum", "mean"],
            "return_pct": ["mean", "std"],
            "holding_days": "mean"
        }).round(2)
        
        # Flatten column names
        symbol_stats.columns = ["_".join(col).strip() for col in symbol_stats.columns]
        
        # Calculate win rates
        win_rates = trades_df[trades_df["realized_pnl"] > 0].groupby("symbol").size() / trades_df.groupby("symbol").size() * 100
        symbol_stats["win_rate"] = win_rates.fillna(0).round(1)
        
        return symbol_stats.reset_index()
    
    def calculate_monthly_returns(self) -> pd.DataFrame:
        """Calculate monthly returns"""
        
        equity_df = self.portfolio.get_equity_curve()
        if equity_df.empty:
            return pd.DataFrame()
        
        equity_df = equity_df.set_index("timestamp")
        equity_df["month"] = equity_df.index.to_period("M")
        
        monthly_equity = equity_df.groupby("month")["equity"].last()
        monthly_returns = monthly_equity.pct_change().dropna() * 100
        
        return monthly_returns.reset_index()


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    # Test the portfolio system
    print("=== Testing Portfolio System ===")
    
    from config import TradingConfig
    
    config = TradingConfig()
    portfolio = PortfolioManager(config, initial_equity=10000)
    
    # Test adding positions
    print("Testing position management...")
    
    # Add some test positions
    strategy_data = {
        "strategy_name": "adx_breakout",
        "signal_strength": 0.8,
        "signal_confidence": 0.7,
        "stop_loss": 95.0
    }
    
    success = portfolio.add_position("AAPL", 100, 150.0, strategy_data)
    print(f"Added AAPL position: {success}")
    
    success = portfolio.add_position("MSFT", 50, 200.0, strategy_data)
    print(f"Added MSFT position: {success}")
    
    # Test portfolio summary
    current_prices = {"AAPL": 155.0, "MSFT": 195.0}
    portfolio.update_equity(current_prices)
    
    summary = portfolio.get_portfolio_summary(current_prices)
    print(f"\nPortfolio Summary:")
    print(f"Current Equity: ${summary['equity']['current']:,.2f}")
    print(f"Total Return: {summary['equity']['total_return_pct']:.2f}%")
    print(f"Open Positions: {summary['positions']['open']}")
    
    # Test position summary
    positions_df = portfolio.get_position_summary(current_prices)
    print(f"\nCurrent Positions:")
    print(positions_df.to_string(index=False))
    
    # Test closing a position
    success = portfolio.close_position("AAPL", 155.0)
    print(f"\nClosed AAPL position: {success}")
    
    # Test trade statistics
    trades_df = portfolio.get_trades_dataframe()
    if not trades_df.empty:
        print(f"\nCompleted Trades:")
        print(trades_df.to_string(index=False))
    
    # Test alerts
    alerts = portfolio.check_position_alerts(current_prices)
    print(f"\nPosition Alerts: {len(alerts)}")
    for alert in alerts:
        print(f"  {alert['type']}: {alert['message']}")
    
    # Test analytics
    analytics = PortfolioAnalytics(portfolio)
    strategy_perf = analytics.analyze_strategy_performance()
    if not strategy_perf.empty:
        print(f"\nStrategy Performance:")
        print(strategy_perf.to_string(index=False))
    
    print("\n=== Portfolio System Test Complete ===")


# ===== INTEGRATION NOTES =====
"""
To integrate this with your existing code:

1. Replace position tracking in backtest with:
   portfolio = PortfolioManager(CONFIG)
   portfolio.add_position(symbol, shares, entry_price, strategy_data)
   portfolio.close_position(symbol, exit_price)

2. Replace portfolio calculations with:
   portfolio.update_equity(current_prices)
   summary = portfolio.get_portfolio_summary()

3. Use position sizing:
   sizer = PositionSizer(CONFIG)
   shares = sizer.calculate_position_size(symbol, entry_price, portfolio.current_equity)

4. Implement risk management:
   can_add, reason = portfolio.risk_manager.check_position_limits(
       portfolio.positions, position_cost, portfolio.current_equity
   )

5. Monitor positions:
   alerts = portfolio.check_position_alerts(current_prices)
   portfolio.rebalance_stops(current_prices)

6. Export results:
   portfolio.export_to_csv("output")
   
7. Advanced analytics:
   analytics = PortfolioAnalytics(portfolio)
   strategy_perf = analytics.analyze_strategy_performance()
"""