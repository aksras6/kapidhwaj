"""
Live Order Management with Trailing Stops

Implements daily cancel/replace logic for child SELL stops to match backtest behavior.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
import threading
import time

from config import TradingConfig
from logger import get_logger, LoggingContext


@dataclass
class LiveOrder:
    """Track live orders with parent/child relationships"""
    order_id: int
    symbol: str
    action: str  # BUY, SELL
    order_type: str  # STP, MKT, LMT
    quantity: int
    price: float
    parent_id: Optional[int] = None
    status: str = "Submitted"
    strategy: str = "adx_breakout"
    
    def is_parent(self) -> bool:
        return self.parent_id is None or self.parent_id == 0
    
    def is_child(self) -> bool:
        return not self.is_parent()


class LiveOrderManager:
    """Manages live orders with trailing stop updates"""
    
    def __init__(self, config: TradingConfig, ib_app=None):
        self.config = config
        self.ib_app = ib_app
        self.logger = get_logger("live_orders")
        
        # Order tracking
        self.active_orders: Dict[int, LiveOrder] = {}
        self.parent_child_map: Dict[int, List[int]] = {}  # parent_id -> [child_ids]
        self.symbol_parents: Dict[str, int] = {}  # symbol -> parent_order_id
        
        # Threading
        self._lock = threading.RLock()
        
        self.logger.info("Live order manager initialized")
    
    def refresh_open_orders(self) -> bool:
        """Refresh order state from IB"""
        if not self.ib_app:
            return False
        
        try:
            with LoggingContext(self.logger, "Refreshing open orders"):
                # Clear current state
                with self._lock:
                    self.active_orders.clear()
                    self.parent_child_map.clear()
                    self.symbol_parents.clear()
                
                # Get fresh orders from IB
                open_orders = self._fetch_open_orders_from_ib()
                
                # Process and categorize orders
                for order_id, contract, order in open_orders:
                    live_order = LiveOrder(
                        order_id=order_id,
                        symbol=getattr(contract, 'symbol', ''),
                        action=getattr(order, 'action', ''),
                        order_type=getattr(order, 'orderType', ''),
                        quantity=getattr(order, 'totalQuantity', 0),
                        price=getattr(order, 'auxPrice', 0.0),
                        parent_id=getattr(order, 'parentId', 0),
                        status="Active"
                    )
                    
                    with self._lock:
                        self.active_orders[order_id] = live_order
                        
                        # Track parent-child relationships
                        if live_order.is_parent():
                            if live_order.symbol:
                                self.symbol_parents[live_order.symbol] = order_id
                                self.parent_child_map[order_id] = []
                        else:
                            parent_id = live_order.parent_id
                            if parent_id in self.parent_child_map:
                                self.parent_child_map[parent_id].append(order_id)
                
                self.logger.info(f"Refreshed {len(self.active_orders)} active orders")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to refresh orders: {e}")
            return False
    
    def _fetch_open_orders_from_ib(self) -> List[Tuple]:
        """Fetch open orders from IB API"""
        if not self.ib_app:
            return []
        
        # Reset and fetch open orders
        self.ib_app._open_orders = []
        self.ib_app._open_orders_done_evt.clear()
        
        self.ib_app.reqOpenOrders()
        
        # Wait for response
        if self.ib_app._open_orders_done_evt.wait(timeout=10):
            return list(self.ib_app._open_orders)
        else:
            self.logger.warning("Timeout waiting for open orders")
            return []
    
    def update_trailing_stops(self, current_data: Dict[str, pd.DataFrame]) -> int:
        """
        Update trailing stops for all symbols to match backtest behavior
        
        Returns:
            Number of stops updated
        """
        updates_made = 0
        
        with LoggingContext(self.logger, "Updating trailing stops"):
            
            # Refresh order state first
            if not self.refresh_open_orders():
                return 0
            
            with self._lock:
                symbols_to_update = list(self.symbol_parents.keys())
            
            for symbol in symbols_to_update:
                if symbol not in current_data:
                    continue
                
                try:
                    updated = self._update_symbol_trailing_stop(symbol, current_data[symbol])
                    if updated:
                        updates_made += 1
                        
                except Exception as e:
                    self.logger.error(f"Failed to update trailing stop for {symbol}: {e}")
        
        if updates_made > 0:
            self.logger.info(f"Updated trailing stops for {updates_made} symbols")
        
        return updates_made
    
    def _update_symbol_trailing_stop(self, symbol: str, df: pd.DataFrame) -> bool:
        """Update trailing stop for a specific symbol"""
        
        with self._lock:
            parent_id = self.symbol_parents.get(symbol)
            if not parent_id or parent_id not in self.parent_child_map:
                return False
            
            child_ids = self.parent_child_map[parent_id]
            if not child_ids:
                return False
            
            # Find the SELL stop child order
            sell_stop_order = None
            for child_id in child_ids:
                if child_id in self.active_orders:
                    order = self.active_orders[child_id]
                    if order.action == "SELL" and order.order_type == "STP":
                        sell_stop_order = order
                        break
            
            if not sell_stop_order:
                self.logger.debug(f"No SELL stop found for {symbol}")
                return False
        
        # Calculate new trailing stop level (LL of last 20 completed bars)
        if len(df) < self.config.CHANNEL_LENGTH:
            return False
        
        # Use last 20 completed bars for LL calculation
        completed_bars = df.dropna(subset=["low"]).tail(self.config.CHANNEL_LENGTH)
        if len(completed_bars) < self.config.CHANNEL_LENGTH:
            return False
        
        new_stop_level = completed_bars["low"].min()
        current_stop_level = sell_stop_order.price
        
        # Only update if new stop is higher (trailing up)
        if new_stop_level <= current_stop_level:
            self.logger.debug(f"{symbol}: New stop {new_stop_level:.2f} not higher than current {current_stop_level:.2f}")
            return False
        
        # Check if update is significant enough
        if not self.config.prices_equal(new_stop_level, current_stop_level):
            return self._cancel_and_replace_stop(symbol, sell_stop_order, new_stop_level)
        
        return False
    
    def _cancel_and_replace_stop(self, symbol: str, old_order: LiveOrder, new_stop_level: float) -> bool:
        """Cancel old stop and place new one"""
        
        if not self.ib_app:
            return False
        
        try:
            # Cancel old order
            self.logger.info(f"Cancelling old stop for {symbol}: {old_order.order_id} @ {old_order.price:.2f}")
            self.ib_app.cancelOrder(old_order.order_id)
            
            # Wait a moment for cancellation
            time.sleep(0.5)
            
            # Create new stop order
            from ibapi_appv1 import create_stop_order, resolve_contract, order_contract
            
            # Get contract for symbol
            contract = resolve_contract(self.ib_app, symbol)
            if not contract:
                self.logger.error(f"Could not resolve contract for {symbol}")
                return False
            
            # Create new stop order
            new_order_id = self.ib_app.next_req_id()
            new_order = create_stop_order("SELL", old_order.quantity, new_stop_level, self.config.ORDER_TIF)
            new_order.orderId = new_order_id
            new_order.parentId = old_order.parent_id
            new_order.transmit = True
            
            # Place new order
            contract_for_order = order_contract(contract)
            self.ib_app.placeOrder(new_order_id, contract_for_order, new_order)
            
            self.logger.info(f"Placed new trailing stop for {symbol}: {new_order_id} @ {new_stop_level:.2f}")
            
            # Update our tracking
            with self._lock:
                # Remove old order
                self.active_orders.pop(old_order.order_id, None)
                
                # Add new order
                new_live_order = LiveOrder(
                    order_id=new_order_id,
                    symbol=symbol,
                    action="SELL",
                    order_type="STP",
                    quantity=old_order.quantity,
                    price=new_stop_level,
                    parent_id=old_order.parent_id,
                    status="Submitted"
                )
                self.active_orders[new_order_id] = new_live_order
                
                # Update parent-child mapping
                if old_order.parent_id in self.parent_child_map:
                    child_list = self.parent_child_map[old_order.parent_id]
                    if old_order.order_id in child_list:
                        child_list.remove(old_order.order_id)
                    child_list.append(new_order_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel/replace stop for {symbol}: {e}")
            return False
    
    def place_bracket_order(self, symbol: str, quantity: int, entry_stop: float, 
                           stop_loss: float, force_replace: bool = False) -> bool:
        """
        Place bracket order with idempotency check
        
        Args:
            symbol: Symbol to trade
            quantity: Number of shares
            entry_stop: Entry stop price
            stop_loss: Initial stop loss
            force_replace: If True, cancel existing and place new
            
        Returns:
            True if order placed successfully
        """
        
        # Check for existing orders
        with self._lock:
            existing_parent = self.symbol_parents.get(symbol)
            
            if existing_parent and not force_replace:
                existing_order = self.active_orders.get(existing_parent)
                if existing_order:
                    # Check if order is substantially the same
                    qty_match = existing_order.quantity == quantity
                    price_match = self.config.prices_equal(existing_order.price, entry_stop)
                    
                    if qty_match and price_match:
                        self.logger.info(f"Identical bracket order already exists for {symbol}")
                        return True
        
        # Cancel existing if force_replace
        if force_replace and existing_parent:
            self._cancel_bracket_for_symbol(symbol)
        
        # Place new bracket order
        return self._place_new_bracket(symbol, quantity, entry_stop, stop_loss)
    
    def _cancel_bracket_for_symbol(self, symbol: str):
        """Cancel entire bracket (parent + children) for symbol"""
        
        with self._lock:
            parent_id = self.symbol_parents.get(symbol)
            if not parent_id:
                return
            
            # Get all related orders
            orders_to_cancel = [parent_id]
            if parent_id in self.parent_child_map:
                orders_to_cancel.extend(self.parent_child_map[parent_id])
        
        # Cancel all orders
        for order_id in orders_to_cancel:
            try:
                self.logger.info(f"Cancelling order {order_id} for {symbol}")
                self.ib_app.cancelOrder(order_id)
            except Exception as e:
                self.logger.error(f"Failed to cancel order {order_id}: {e}")
        
        # Clean up tracking
        with self._lock:
            self.symbol_parents.pop(symbol, None)
            self.parent_child_map.pop(parent_id, None)
            for order_id in orders_to_cancel:
                self.active_orders.pop(order_id, None)
    
    def _place_new_bracket(self, symbol: str, quantity: int, entry_stop: float, stop_loss: float) -> bool:
        """Place new bracket order"""
        
        if not self.ib_app:
            return False
        
        try:
            from ibapi_appv1 import resolve_contract, create_bracket_buy, order_contract
            
            # Get contract
            contract = resolve_contract(self.ib_app, symbol)
            if not contract:
                self.logger.error(f"Could not resolve contract for {symbol}")
                return False
            
            # Create bracket orders
            parent_id = self.ib_app.next_req_id()
            bracket_orders = create_bracket_buy(parent_id, quantity, entry_stop, stop_loss, self.config.ORDER_TIF)
            
            contract_for_order = order_contract(contract)
            
            # Place orders
            for order in bracket_orders:
                self.ib_app.placeOrder(order.orderId, contract_for_order, order)
                
                # Track the order
                live_order = LiveOrder(
                    order_id=order.orderId,
                    symbol=symbol,
                    action=order.action,
                    order_type=order.orderType,
                    quantity=order.totalQuantity,
                    price=getattr(order, 'auxPrice', 0.0),
                    parent_id=getattr(order, 'parentId', 0),
                    status="Submitted"
                )
                
                with self._lock:
                    self.active_orders[order.orderId] = live_order
                    
                    if live_order.is_parent():
                        self.symbol_parents[symbol] = order.orderId
                        self.parent_child_map[order.orderId] = []
                    else:
                        if parent_id in self.parent_child_map:
                            self.parent_child_map[parent_id].append(order.orderId)
            
            self.logger.info(f"Placed bracket order for {symbol}: {quantity}@{entry_stop:.2f}, stop@{stop_loss:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to place bracket for {symbol}: {e}")
            return False
    
    def get_order_summary(self) -> Dict[str, any]:
        """Get summary of current orders"""
        
        with self._lock:
            summary = {
                "total_orders": len(self.active_orders),
                "parent_orders": len(self.symbol_parents),
                "symbols_with_orders": list(self.symbol_parents.keys()),
                "orders_by_type": {},
                "orders_by_status": {}
            }
            
            for order in self.active_orders.values():
                # Count by type
                order_type = f"{order.action}_{order.order_type}"
                summary["orders_by_type"][order_type] = summary["orders_by_type"].get(order_type, 0) + 1
                
                # Count by status
                summary["orders_by_status"][order.status] = summary["orders_by_status"].get(order.status, 0) + 1
        
        return summary


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    # Test the live order manager
    print("=== Testing Live Order Manager ===")
    
    from config import TradingConfig
    
    config = TradingConfig()
    order_manager = LiveOrderManager(config)
    
    # Test order tracking
    summary = order_manager.get_order_summary()
    print(f"Order summary: {summary}")
    
    print("\n=== Live Order Manager Test Complete ===")


# ===== INTEGRATION NOTES =====
"""
To integrate with your live trading:

1. Initialize in trading engine:
   self.order_manager = LiveOrderManager(self.config, self.ib_app)

2. Update stops daily after market close:
   current_data = self.data_manager.get_multiple_symbols(self.config.SYMBOLS)
   updates = self.order_manager.update_trailing_stops(current_data)

3. Place orders with idempotency:
   success = self.order_manager.place_bracket_order(
       symbol, shares, entry_stop, stop_loss, force_replace=False
   )

4. Monitor orders:
   summary = self.order_manager.get_order_summary()
   
This ensures your live trading matches the backtest behavior with proper trailing stops!
"""