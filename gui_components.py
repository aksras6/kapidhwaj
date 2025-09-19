"""
GUI Components for Trading System

Individual tab implementations and reusable UI components.
"""

import os
import threading
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from config import TradingConfig
from logger import get_logger
from visual_strategy_builder import EnhancedStrategyTab


class ConnectionTab:
    """IB Connection management tab."""

    def __init__(self, parent, config, connection_callback):
        self.parent = parent
        self.config = config
        self.connection_callback = connection_callback
        self.logger = get_logger("gui_connection")

        self.setup_ui()

    def setup_ui(self):
        """Setup connection tab UI."""
        # Main container
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Interactive Brokers Connection",
            style="Title.TLabel",
        )
        title_label.pack(anchor="w", pady=(0, 20))

        # Connection settings frame
        settings_frame = ttk.LabelFrame(
            main_frame,
            text="Connection Settings",
            padding=10,
        )
        settings_frame.pack(fill="x", pady=(0, 20))

        # Host setting
        ttk.Label(settings_frame, text="Host:").grid(row=0, column=0, sticky="w", pady=5)
        self.host_var = tk.StringVar(value=self.config.IB_HOST)
        host_entry = ttk.Entry(settings_frame, textvariable=self.host_var, width=20)
        host_entry.grid(row=0, column=1, sticky="w", padx=(10, 0), pady=5)

        # Port setting
        ttk.Label(settings_frame, text="Port:").grid(row=1, column=0, sticky="w", pady=5)
        self.port_var = tk.StringVar(value=str(self.config.IB_PAPER_PORT))
        port_entry = ttk.Entry(settings_frame, textvariable=self.port_var, width=20)
        port_entry.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=5)

        # Client ID setting
        ttk.Label(settings_frame, text="Client ID:").grid(row=2, column=0, sticky="w", pady=5)
        self.client_id_var = tk.StringVar(value=str(self.config.IB_CLIENT_ID))
        client_id_entry = ttk.Entry(settings_frame, textvariable=self.client_id_var, width=20)
        client_id_entry.grid(row=2, column=1, sticky="w", padx=(10, 0), pady=5)

        # Trading mode
        ttk.Label(settings_frame, text="Mode:").grid(row=3, column=0, sticky="w", pady=5)
        self.mode_var = tk.StringVar(value="Paper Trading")
        mode_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.mode_var,
            values=["Paper Trading", "Live Trading"],
            state="readonly",
            width=17,
        )
        mode_combo.grid(row=3, column=1, sticky="w", padx=(10, 0), pady=5)

        # Connection buttons
        button_frame = ttk.Frame(settings_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=(15, 0))

        self.connect_btn = ttk.Button(button_frame, text="Connect", command=self.connect)
        self.connect_btn.pack(side="left", padx=(0, 10))

        self.disconnect_btn = ttk.Button(
            button_frame,
            text="Disconnect",
            command=self.disconnect,
            state="disabled",
        )
        self.disconnect_btn.pack(side="left")

        # Status frame
        status_frame = ttk.LabelFrame(
            main_frame,
            text="Connection Status",
            padding=10,
        )
        status_frame.pack(fill="x", pady=(0, 20))

        self.status_label = ttk.Label(
            status_frame,
            text="⚫ Disconnected",
            style="Status.TLabel",
        )
        self.status_label.pack(anchor="w")

        # Info frame
        info_frame = ttk.LabelFrame(main_frame, text="Setup Instructions", padding=10)
        info_frame.pack(fill="both", expand=True)

        info_text = """
Setup Instructions:

1. Install and start Interactive Brokers TWS or Gateway
2. Enable API connections in TWS/Gateway:
   - Go to Edit → Global Configuration → API → Settings
   - Check "Enable ActiveX and Socket Clients"
   - Add your client ID to trusted clients
   - Set socket port (7497 for paper, 7496 for live)

3. Paper Trading Setup:
   - Use port 7497
   - Enable paper trading account in TWS

4. Live Trading Setup:
   - Use port 7496
   - Ensure live account is funded and active
   - WARNING: Live trading uses real money!

5. Click Connect to establish connection
        """.strip()

        info_display = tk.Text(info_frame, wrap="word", height=12, state="disabled")
        info_display.pack(fill="both", expand=True)
        info_display.config(state="normal")
        info_display.insert("1.0", info_text)
        info_display.config(state="disabled")

    def connect(self):
        """Handle connect button."""
        # Update config with current values
        self.config.IB_HOST = self.host_var.get()
        self.config.IB_PAPER_PORT = int(self.port_var.get())
        self.config.IB_CLIENT_ID = int(self.client_id_var.get())

        # Update UI
        self.connect_btn.config(state="disabled")
        self.status_label.config(text="🔄 Connecting...")

        # Trigger connection through callback
        self.connection_callback(True)

    # Enhanced disconnect method in ConnectionTab class in gui_components.py:

    def disconnect(self):
        """Handle disconnect button."""
        try:
            # Update UI immediately to show disconnecting state
            self.status_label.config(text="🔄 Disconnecting...")
            self.disconnect_btn.config(state="disabled")
            
            # Log the disconnect attempt
            self.logger.info("User initiated disconnect")
            
            # Trigger actual disconnection through callback
            self.connection_callback(False)
            
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")
            # Reset UI on error
            self.disconnect_btn.config(state="normal")
            self.status_label.config(text="🔴 Disconnect Error")

    def on_connection_change(self, connected: bool):
        """Update UI based on connection status."""
        if connected:
            self.status_label.config(text="🟢 Connected")
            self.connect_btn.config(state="disabled")
            self.disconnect_btn.config(state="normal")
            self.logger.info("Connection established")
        else:
            self.status_label.config(text="⚫ Disconnected")
            self.connect_btn.config(state="normal")
            self.disconnect_btn.config(state="disabled")
            self.logger.info("Connection terminated")


class StrategyTab:
    """Strategy editor tab."""

    def __init__(self, parent, config, strategy_callback):
        self.parent = parent
        self.config = config
        self.strategy_callback = strategy_callback
        self.logger = get_logger("gui_strategy")
        self.current_file = None

        self.setup_ui()
        self.load_default_strategy()

    def setup_ui(self):
        """Setup strategy tab UI."""
        # Main container
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Toolbar
        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill="x", pady=(0, 10))

        ttk.Button(toolbar, text="New", command=self.new_strategy).pack(side="left", padx=(0, 5))
        ttk.Button(toolbar, text="Open", command=self.open_strategy).pack(side="left", padx=(0, 5))
        ttk.Button(toolbar, text="Save", command=self.save_strategy).pack(side="left", padx=(0, 5))
        ttk.Button(toolbar, text="Save As", command=self.save_as_strategy).pack(side="left", padx=(0, 5))

        ttk.Separator(toolbar, orient="vertical").pack(side="left", padx=10, fill="y")

        ttk.Button(toolbar, text="Validate", command=self.validate_syntax).pack(side="left", padx=(0, 5))
        ttk.Button(toolbar, text="Test", command=self.test_strategy).pack(side="left", padx=(0, 5))

        # File info
        self.file_label = ttk.Label(toolbar, text="Untitled Strategy")
        self.file_label.pack(side="right")

        # Editor frame
        editor_frame = ttk.Frame(main_frame)
        editor_frame.pack(fill="both", expand=True)

        # Code editor with scrollbars
        self.text_editor = scrolledtext.ScrolledText(
            editor_frame,
            wrap="none",
            font=("Consolas", 10),
            undo=True,
            maxundo=50,
        )
        self.text_editor.pack(fill="both", expand=True)

        # Bind events
        self.text_editor.bind("<KeyRelease>", self.on_text_change)
        self.text_editor.bind("<Control-s>", lambda e: self.save_strategy())
        self.text_editor.bind("<Control-o>", lambda e: self.open_strategy())
        self.text_editor.bind("<Control-n>", lambda e: self.new_strategy())

    def load_default_strategy(self):
        """Load the default ADX strategy template."""
        template = '''"""
ADX Breakout Strategy - Customizable Template

This strategy implements the ADX breakout logic from your original system.
You can modify the parameters and logic below to customize the strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class ADXBreakoutStrategy:
    def __init__(self, config):
        self.config = config

        # Strategy parameters - modify these as needed
        self.channel_length = 20      # Donchian channel period
        self.adx_length = 15          # ADX calculation period
        self.adx_threshold = 20.0     # ADX threshold for trading

    def generate_entry_signal(self, df: pd.DataFrame, symbol: str) -> dict:
        """
        Generate entry signals.

        Args:
            df: DataFrame with OHLC data and indicators (ADX, HH, LL)
            symbol: Symbol being analyzed

        Returns:
            Signal dictionary or None
        """
        if len(df) < 2:
            return None

        current = df.iloc[-1]

        # Check ADX condition - only trade when ADX < threshold (consolidation)
        if pd.isna(current["ADX"]) or current["ADX"] >= self.adx_threshold:
            return None

        # Check for valid breakout setup
        if pd.isna(current["HH"]) or pd.isna(current["LL"]):
            return None

        # Entry signal: Buy stop at highest high of channel
        return {
            "action": "BUY",
            "symbol": symbol,
            "signal_type": "ENTRY",
            "entry_price": current["HH"],     # Stop entry price
            "stop_loss": current["LL"],       # Initial stop loss
            "confidence": self.calculate_confidence(df),
            "reason": f"ADX breakout setup: ADX={current['ADX']:.1f} < {self.adx_threshold}",
        }

    def generate_exit_signal(self, df: pd.DataFrame, symbol: str, position) -> dict:
        """
        Generate exit signals for existing positions.

        Args:
            df: DataFrame with OHLC data and indicators
            symbol: Symbol being analyzed
            position: Current position object

        Returns:
            Signal dictionary or None
        """
        if len(df) < 1:
            return None

        current = df.iloc[-1]

        # Exit on trailing stop (lowest low of channel)
        if pd.notna(current["LL"]):
            return {
                "action": "SELL",
                "symbol": symbol,
                "signal_type": "EXIT",
                "exit_price": current["LL"],   # Stop exit price
                "reason": f"Trailing stop at LL={current['LL']:.2f}",
            }

        return None

    def calculate_confidence(self, df: pd.DataFrame) -> float:
        """Calculate signal confidence (0-1)."""
        # Look at recent price action for confidence
        if len(df) < 5:
            return 0.6  # Default confidence

        recent_data = df.tail(5)

        # Higher confidence if price has been consolidating
        price_range = recent_data["high"].max() - recent_data["low"].min()
        avg_price = recent_data["close"].mean()

        # Lower volatility = higher confidence
        volatility_ratio = price_range / avg_price
        confidence = max(0.3, min(0.9, 1.0 - volatility_ratio * 10))

        return confidence


# Strategy validation
def validate_strategy():
    """Validate strategy implementation."""
    try:
        from config import TradingConfig
        config = TradingConfig()
        _ = ADXBreakoutStrategy(config)
        print("✅ Strategy validation passed")
        return True
    except Exception as e:
        print(f"❌ Strategy validation failed: {e}")
        return False


if __name__ == "__main__":
    validate_strategy()
'''
        self.text_editor.insert("1.0", template)
        self.text_editor.edit_reset()  # Reset undo history

    def on_text_change(self, event=None):
        """Handle text changes."""
        if self.current_file:
            # Mark as modified
            title = f"*{os.path.basename(self.current_file)}"
            self.file_label.config(text=title)

        # Notify callback
        content = self.text_editor.get("1.0", "end-1c")
        self.strategy_callback(content)

    def new_strategy(self):
        """Create new strategy."""
        if self.check_unsaved_changes():
            self.text_editor.delete("1.0", "end")
            self.load_default_strategy()
            self.current_file = None
            self.file_label.config(text="Untitled Strategy")

    def open_strategy(self):
        """Open existing strategy file."""
        if not self.check_unsaved_changes():
            return

        filename = filedialog.askopenfilename(
            title="Open Strategy",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
            initialdir=self.config.STRATEGY_DIR,
        )
        if filename:
            self.load_strategy(filename)

    def load_strategy(self, filename):
        """Load strategy from file."""
        try:
            with open(filename, "r", encoding="utf-8") as f:
                content = f.read()

            self.text_editor.delete("1.0", "end")
            self.text_editor.insert("1.0", content)
            self.text_editor.edit_reset()

            self.current_file = filename
            self.file_label.config(text=os.path.basename(filename))

            self.logger.info(f"Loaded strategy from {filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load strategy:\n{e}")

    def save_strategy(self):
        """Save current strategy."""
        if self.current_file:
            self._save_to_file(self.current_file)
        else:
            self.save_as_strategy()

    def save_as_strategy(self):
        """Save strategy as new file."""
        filename = filedialog.asksaveasfilename(
            title="Save Strategy",
            defaultextension=".py",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
            initialdir=self.config.STRATEGY_DIR,
        )
        if filename:
            self._save_to_file(filename)
            self.current_file = filename
            self.file_label.config(text=os.path.basename(filename))

    def _save_to_file(self, filename):
        """Save content to file."""
        try:
            content = self.text_editor.get("1.0", "end-1c")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)

            # Remove modified marker
            if self.current_file:
                self.file_label.config(text=os.path.basename(self.current_file))

            self.logger.info(f"Saved strategy to {filename}")
            messagebox.showinfo("Success", "Strategy saved successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save strategy:\n{e}")

    def validate_syntax(self):
        """Validate Python syntax."""
        content = self.text_editor.get("1.0", "end-1c")

        try:
            compile(content, "<strategy>", "exec")
            messagebox.showinfo("Validation", "✅ Syntax validation passed")
        except SyntaxError as e:
            messagebox.showerror(
                "Syntax Error",
                f"❌ Syntax error at line {e.lineno}:\n{e.msg}",
            )
        except Exception as e:
            messagebox.showerror("Error", f"❌ Validation error:\n{e}")

    def test_strategy(self):
        """Test strategy with sample data."""
        content = self.text_editor.get("1.0", "end-1c")

        def test_thread():
            try:
                # Execute strategy code
                namespace = {}
                exec(content, namespace)

                # Test if strategy class exists
                if "ADXBreakoutStrategy" in namespace:
                    # Create test data
                    dates = pd.date_range("2023-01-01", periods=100, freq="D")
                    test_data = pd.DataFrame(
                        {
                            "open": np.random.randn(100).cumsum() + 100,
                            "high": np.random.randn(100).cumsum() + 102,
                            "low": np.random.randn(100).cumsum() + 98,
                            "close": np.random.randn(100).cumsum() + 100,
                            "ADX": np.random.uniform(10, 30, 100),
                            "HH": np.random.randn(100).cumsum() + 105,
                            "LL": np.random.randn(100).cumsum() + 95,
                        },
                        index=dates,
                    )

                    # Test strategy
                    strategy = namespace["ADXBreakoutStrategy"](self.config)
                    signal = strategy.generate_entry_signal(test_data, "TEST")

                    result = "✅ Strategy test passed!\n"
                    if signal:
                        result += f"Generated signal: {signal}"
                    else:
                        result += "No signal generated (normal for test data)"

                    messagebox.showinfo("Test Result", result)
                else:
                    messagebox.showerror(
                        "Test Error",
                        "❌ ADXBreakoutStrategy class not found",
                    )

            except Exception as e:
                messagebox.showerror("Test Error", f"❌ Strategy test failed:\n{e}")

        # Run test in background
        threading.Thread(target=test_thread, daemon=True).start()

    def check_unsaved_changes(self) -> bool:
        """Check for unsaved changes."""
        if "*" in self.file_label.cget("text"):
            result = messagebox.askyesnocancel(
                "Unsaved Changes",
                "You have unsaved changes. Save before continuing?",
            )
            if result is True:
                self.save_strategy()
                return True
            elif result is False:
                return True
            else:
                return False
        return True

# class EnhancedStrategyTab:
#     """Enhanced Strategy Tab with Visual Builder"""
    
#     def __init__(self, parent, config, strategy_callback):
#         self.parent = parent
#         self.config = config
#         self.strategy_callback = strategy_callback
#         self.logger = get_logger("enhanced_strategy")
        
#         self.setup_ui()
    
#     def setup_ui(self):
#         """Setup enhanced strategy tab with mode switcher"""
        
#         # Main container
#         main_frame = ttk.Frame(self.parent)
#         main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
#         # Top toolbar with mode switcher
#         toolbar = ttk.Frame(main_frame)
#         toolbar.pack(fill="x", pady=(0, 10))
        
#         ttk.Label(toolbar, text="Strategy Editor Mode:", font=("Arial", 10, "bold")).pack(side="left", padx=(0, 10))
        
#         self.mode_var = tk.StringVar(value="visual")
#         mode_frame = ttk.Frame(toolbar)
#         mode_frame.pack(side="left")
        
#         ttk.Radiobutton(
#             mode_frame, 
#             text="🎨 Visual Builder", 
#             variable=self.mode_var, 
#             value="visual",
#             command=self.switch_mode
#         ).pack(side="left", padx=(0, 10))
        
#         ttk.Radiobutton(
#             mode_frame,
#             text="🐍 Code Editor", 
#             variable=self.mode_var,
#             value="code",
#             command=self.switch_mode
#         ).pack(side="left")
        
#         # Content area
#         self.content_frame = ttk.Frame(main_frame)
#         self.content_frame.pack(fill="both", expand=True)
        
#         # Initialize visual builder
#         self.visual_builder = VisualStrategyBuilder(
#             self.content_frame, 
#             self.config,
#             strategy_callback=self.test_visual_strategy
#         )
        
#         # Initialize code editor (hidden initially)
#         self.code_editor_frame = ttk.Frame(self.content_frame)
#         self.setup_code_editor()
        
#         # Start with visual mode
#         self.switch_mode()
    
#     def setup_code_editor(self):
#         """Setup traditional code editor"""
        
#         # Toolbar
#         toolbar = ttk.Frame(self.code_editor_frame)
#         toolbar.pack(fill="x", pady=(0, 10))
        
#         ttk.Button(toolbar, text="New", command=self.new_strategy).pack(side="left", padx=(0, 5))
#         ttk.Button(toolbar, text="Open", command=self.open_strategy).pack(side="left", padx=(0, 5))
#         ttk.Button(toolbar, text="Save", command=self.save_strategy).pack(side="left", padx=(0, 5))
#         ttk.Button(toolbar, text="Test", command=self.test_code_strategy).pack(side="left", padx=(0, 10))
        
#         # Code editor
#         self.code_editor = scrolledtext.ScrolledText(
#             self.code_editor_frame,
#             wrap="none",
#             font=("Consolas", 10),
#             undo=True,
#             maxundo=50,
#         )
#         self.code_editor.pack(fill="both", expand=True)
        
#         # Load default strategy template
#         self.load_default_strategy()
    
#     def switch_mode(self):
#         """Switch between visual and code modes"""
#         mode = self.mode_var.get()
        
#         # Hide all frames
#         for widget in self.content_frame.winfo_children():
#             widget.pack_forget()
        
#         if mode == "visual":
#             self.visual_builder.main_notebook.pack(fill="both", expand=True)
#         else:
#             self.code_editor_frame.pack(fill="both", expand=True)
        
#         self.logger.info(f"Switched to {mode} mode")
    
#     def test_visual_strategy(self, strategy_params):
#         """Test strategy created with visual builder"""
#         self.logger.info("Testing visual strategy...")
        
#         # Convert visual strategy to format expected by backtesting engine
#         backtest_params = {
#             "strategy_type": "visual_generated",
#             "strategy_name": strategy_params["strategy_name"],
#             "entry_conditions": strategy_params["entry_conditions"],
#             "exit_conditions": strategy_params["exit_conditions"],
#             "risk_settings": strategy_params["risk_settings"]
#         }
        
#         if self.strategy_callback:
#             self.strategy_callback(backtest_params)
    
#     def test_code_strategy(self):
#         """Test manually coded strategy"""
#         code = self.code_editor.get("1.0", "end-1c")
        
#         backtest_params = {
#             "strategy_type": "custom_code",
#             "strategy_code": code
#         }
        
#         if self.strategy_callback:
#             self.strategy_callback(backtest_params)
    
#     def new_strategy(self):
#         """Create new strategy"""
#         self.code_editor.delete("1.0", "end")
#         self.load_default_strategy()
    
#     def open_strategy(self):
#         """Open existing strategy"""
#         from tkinter import filedialog
        
#         filename = filedialog.askopenfilename(
#             title="Open Strategy",
#             filetypes=[("Python files", "*.py"), ("JSON templates", "*.json"), ("All files", "*.*")]
#         )
        
#         if filename:
#             try:
#                 with open(filename, 'r') as f:
#                     content = f.read()
                
#                 if filename.endswith('.json'):
#                     # Load as visual template
#                     self.mode_var.set("visual")
#                     self.switch_mode()
#                     self.visual_builder.load_template(filename)
#                 else:
#                     # Load as code
#                     self.mode_var.set("code")
#                     self.switch_mode()
#                     self.code_editor.delete("1.0", "end")
#                     self.code_editor.insert("1.0", content)
                
#             except Exception as e:
#                 messagebox.showerror("Error", f"Failed to open file: {e}")
    
#     def save_strategy(self):
#         """Save current strategy"""
#         from tkinter import filedialog
        
#         if self.mode_var.get() == "visual":
#             self.visual_builder.save_template()
#         else:
#             filename = filedialog.asksaveasfilename(
#                 title="Save Strategy",
#                 defaultextension=".py",
#                 filetypes=[("Python files", "*.py"), ("All files", "*.*")]
#             )
            
#             if filename:
#                 try:
#                     code = self.code_editor.get("1.0", "end-1c")
#                     with open(filename, 'w') as f:
#                         f.write(code)
#                     messagebox.showinfo("Saved", f"Strategy saved to {filename}")
#                 except Exception as e:
#                     messagebox.showerror("Error", f"Failed to save file: {e}")
    
#     def load_default_strategy(self):
#         """Load default strategy template"""
#         default_strategy = '''"""
# ADX Breakout Strategy Template

# Customize this strategy or switch to Visual Builder mode
# for drag-and-drop strategy creation.
# """

# import pandas as pd
# import numpy as np

# class CustomStrategy:
#     """Custom trading strategy"""
    
#     def __init__(self, config):
#         self.config = config
#         self.name = "Custom ADX Strategy"
    
#     def generate_entry_signals(self, data):
#         """Generate buy/sell entry signals"""
        
#         # Example: ADX breakout strategy
#         adx = self.calculate_adx(data, 14)
#         sma_20 = data['close'].rolling(20).mean()
        
#         # Entry conditions
#         entry_signals = (
#             (adx > 25) &  # Strong trend
#             (data['close'] > sma_20) &  # Above moving average
#             (data['volume'] > data['volume'].rolling(10).mean())  # Volume confirmation
#         )
        
#         return entry_signals
    
#     def generate_exit_signals(self, data, positions):
#         """Generate exit signals"""
        
#         sma_20 = data['close'].rolling(20).mean()
        
#         # Exit when price falls below moving average
#         exit_signals = data['close'] < sma_20
        
#         return exit_signals
    
#     def calculate_adx(self, data, period=14):
#         """Calculate ADX indicator"""
#         # Implement your ADX calculation here
#         # This is a placeholder
#         return pd.Series([25] * len(data), index=data.index)
    
#     def get_position_size(self, signal, portfolio_value):
#         """Calculate position size"""
#         return portfolio_value * 0.05  # 5% of portfolio
    
#     def get_stop_loss(self, entry_price, signal):
#         """Calculate stop loss"""
#         return entry_price * 0.98  # 2% stop loss
    
#     def get_take_profit(self, entry_price, signal):
#         """Calculate take profit"""
#         return entry_price * 1.06  # 6% take profit
# '''
#         self.code_editor.insert("1.0", default_strategy)

class BacktestTab:
    """Backtesting tab."""

    def __init__(self, parent, config, backtest_callback):
        self.parent = parent
        self.config = config
        self.backtest_callback = backtest_callback
        self.logger = get_logger("gui_backtest")

        self.setup_ui()

    def setup_ui(self):
        """Setup backtest tab UI."""
        # Main container with left/right split
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Left panel - Parameters
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side="left", fill="y", padx=(0, 10))

        # Right panel - Results
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True)

        self.setup_parameters_panel(left_frame)
        self.setup_results_panel(right_frame)

    def setup_parameters_panel(self, parent):
        """Setup backtest parameters panel."""
        # Title
        ttk.Label(parent, text="Backtest Parameters", style="Title.TLabel").pack(
            anchor="w", pady=(0, 10)
        )

        # Date range - UPDATED DEFAULTS
        date_frame = ttk.LabelFrame(parent, text="Date Range", padding=10)
        date_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(date_frame, text="Start Date:").grid(row=0, column=0, sticky="w", pady=2)
        # FIX: Use more recent start date
        self.start_date_var = tk.StringVar(value="2023-01-01")  
        start_entry = ttk.Entry(date_frame, textvariable=self.start_date_var, width=15)
        start_entry.grid(row=0, column=1, padx=(10, 0), pady=2)

        ttk.Label(date_frame, text="End Date:").grid(row=1, column=0, sticky="w", pady=2)
        # FIX: Use current date as default
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
        self.end_date_var = tk.StringVar(value=current_date)
        end_entry = ttk.Entry(date_frame, textvariable=self.end_date_var, width=15)
        end_entry.grid(row=1, column=1, padx=(10, 0), pady=2)

        # Capital
        capital_frame = ttk.LabelFrame(parent, text="Capital", padding=10)
        capital_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(capital_frame, text="Initial Capital:").grid(row=0, column=0, sticky="w", pady=2)
        self.capital_var = tk.StringVar(value=str(self.config.ACCOUNT_SIZE))
        ttk.Entry(capital_frame, textvariable=self.capital_var, width=15).grid(
            row=0, column=1, padx=(10, 0), pady=2
        )

        # Symbols
        symbols_frame = ttk.LabelFrame(parent, text="Symbols", padding=10)
        symbols_frame.pack(fill="x", pady=(0, 10))

        # Symbol selection
        self.symbols_text = tk.Text(symbols_frame, height=6, width=20)
        self.symbols_text.pack(fill="x")
        self.symbols_text.insert("1.0", ",".join(self.config.SYMBOLS))

        # Universe selection
        ttk.Label(symbols_frame, text="Or use universe:").pack(anchor="w", pady=(10, 2))
        self.universe_var = tk.StringVar(value="Custom")
        universe_combo = ttk.ComboBox = ttk.Combobox(
            symbols_frame,
            textvariable=self.universe_var,
            values=["Custom", "S&P 500 (2020)", "NASDAQ 100 (2020)", "Large Cap (2020)"],
            state="readonly",
        )
        universe_combo.pack(fill="x")
        universe_combo.bind("<<ComboboxSelected>>", self.on_universe_change)

        # Strategy parameters
        strategy_frame = ttk.LabelFrame(parent, text="Strategy", padding=10)
        strategy_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(strategy_frame, text="Channel Length:").grid(row=0, column=0, sticky="w", pady=2)
        self.channel_var = tk.StringVar(value=str(self.config.CHANNEL_LENGTH))
        ttk.Entry(strategy_frame, textvariable=self.channel_var, width=10).grid(
            row=0, column=1, padx=(10, 0), pady=2
        )

        ttk.Label(strategy_frame, text="ADX Length:").grid(row=1, column=0, sticky="w", pady=2)
        self.adx_len_var = tk.StringVar(value=str(self.config.ADX_LENGTH))
        ttk.Entry(strategy_frame, textvariable=self.adx_len_var, width=10).grid(
            row=1, column=1, padx=(10, 0), pady=2
        )

        ttk.Label(strategy_frame, text="ADX Threshold:").grid(row=2, column=0, sticky="w", pady=2)
        self.adx_thresh_var = tk.StringVar(value=str(self.config.ADX_THRESHOLD))
        ttk.Entry(strategy_frame, textvariable=self.adx_thresh_var, width=10).grid(
            row=2, column=1, padx=(10, 0), pady=2
        )

        # Run button
        self.run_btn = ttk.Button(parent, text="🚀 Run Backtest", command=self.run_backtest)
        self.run_btn.pack(pady=20)

        # Progress
        self.progress = ttk.Progressbar(parent, mode="indeterminate")
        self.progress.pack(fill="x", pady=(0, 10))

    def setup_results_panel(self, parent):
        """Setup results display panel."""
        # Title
        ttk.Label(parent, text="Backtest Results", style="Title.TLabel").pack(
            anchor="w", pady=(0, 10)
        )

        # Results notebook
        self.results_notebook = ttk.Notebook(parent)
        self.results_notebook.pack(fill="both", expand=True)

        # Summary tab
        self.summary_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.summary_frame, text="📊 Summary")

        # Trades tab
        self.trades_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.trades_frame, text="📋 Trades")

        # Charts tab
        self.charts_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.charts_frame, text="📈 Charts")

        # Setup individual result panels
        self.setup_summary_tab()
        self.setup_trades_tab()
        self.setup_charts_tab()

    def setup_summary_tab(self):
        """Setup summary results tab."""
        self.summary_text = scrolledtext.ScrolledText(
            self.summary_frame,
            wrap="word",
            font=("Consolas", 9),
            state="disabled",
        )
        self.summary_text.pack(fill="both", expand=True, padx=5, pady=5)

    def setup_trades_tab(self):
        """Setup trades results tab."""
        columns = (
            "Symbol",
            "Entry Date",
            "Exit Date",
            "Shares",
            "Entry Price",
            "Exit Price",
            "P&L",
            "Return %",
        )
        self.trades_tree = ttk.Treeview(
            self.trades_frame,
            columns=columns,
            show="headings",
            height=15,
        )

        for col in columns:
            self.trades_tree.heading(col, text=col)
            self.trades_tree.column(col, width=100)

        v_scrollbar = ttk.Scrollbar(
            self.trades_frame,
            orient="vertical",
            command=self.trades_tree.yview,
        )
        h_scrollbar = ttk.Scrollbar(
            self.trades_frame,
            orient="horizontal",
            command=self.trades_tree.xview,
        )
        self.trades_tree.configure(
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set,
        )

        self.trades_tree.pack(side="left", fill="both", expand=True)
        v_scrollbar.pack(side="right", fill="y")
        h_scrollbar.pack(side="bottom", fill="x")

    def setup_charts_tab(self):
        """Setup charts tab."""
        ttk.Label(
            self.charts_frame,
            text="📈 Charts will be displayed here\n(Implementation in Phase 8)",
            style="Heading.TLabel",
        ).pack(expand=True)

    def on_universe_change(self, event=None):
        """Handle universe selection change."""
        universe = self.universe_var.get()

        if universe == "S&P 500 (2020)":
            # Load S&P 500 symbols
            symbols = [
                "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG",
                "META", "TSLA", "NVDA", "BRK.A", "BRK.B",
            ]  # Sample
        elif universe == "NASDAQ 100 (2020)":
            symbols = [
                "AAPL", "MSFT", "AMZN", "GOOGL", "META",
                "TSLA", "NVDA", "NFLX", "ADBE", "INTC",
            ]  # Sample
        elif universe == "Large Cap (2020)":
            symbols = [
                "AAPL", "MSFT", "AMZN", "GOOGL", "JPM",
                "JNJ", "V", "PG", "UNH", "HD",
            ]  # Sample
        else:
            return  # Custom - don't change

        self.symbols_text.delete("1.0", "end")
        self.symbols_text.insert("1.0", ",".join(symbols))

    def run_backtest(self):
        """Run backtest with current parameters."""
        try:
            # Validate date inputs
            start_date = self.start_date_var.get().strip()
            end_date = self.end_date_var.get().strip()
            
            # Validate date format
            try:
                from datetime import datetime
                datetime.strptime(start_date, "%Y-%m-%d")
                datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                messagebox.showerror("Date Error", 
                                "Please use YYYY-MM-DD format for dates")
                return
            
            if start_date >= end_date:
                messagebox.showerror("Date Error", 
                                "Start date must be before end date")
                return
            
            symbols_text = self.symbols_text.get("1.0", "end-1c").strip()
            symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]

            if not symbols:
                messagebox.showerror("Error", "Please enter at least one symbol")
                return

            params = {
                "symbols": symbols,
                "start_date": start_date,  # Now properly passed
                "end_date": end_date,      # Now properly passed
                "initial_capital": float(self.capital_var.get()),
                "channel_length": int(self.channel_var.get()),
                "adx_length": int(self.adx_len_var.get()),
                "adx_threshold": float(self.adx_thresh_var.get()),
            }

            # Start progress
            self.run_btn.config(state="disabled", text="⏳ Running...")
            self.progress.start()

            # Clear previous results
            self.clear_results()

            # Call backtest
            self.backtest_callback(params)

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start backtest: {e}")

    def display_results(self, results):
        """Display backtest results with data source warnings."""
        try:
            # Stop progress and reset button
            self.progress.stop()
            self.run_btn.config(state="normal", text="🚀 Run Backtest")

            # Handle errors
            if "error" in results:
                messagebox.showerror("Backtest Error", results["error"])
                return

            # Check data source and prepare warning
            data_source = results.get("data_source", "UNKNOWN")
            data_warning = results.get("data_warning", "")
            
            # Clear previous results
            self.summary_text.config(state="normal")
            self.summary_text.delete("1.0", "end")
            
            # Clear trades tree
            for item in self.trades_tree.get_children():
                self.trades_tree.delete(item)

            # Add header with data source info
            header = "BACKTEST RESULTS\n" + "="*50 + "\n\n"
            
            if data_source == "MOCK":
                header += "🚨 WARNING: USING MOCK DATA (NOT REAL MARKET DATA)\n"
                header += "Connect to Interactive Brokers for real historical data\n"
                header += "Results below are artificial and not realistic!\n\n"
            elif data_source == "IB_REAL":
                header += "✅ Using real Interactive Brokers historical data\n\n"
            elif data_source == "UNKNOWN":
                header += "⚠️ Data source unknown - verify data quality\n\n"
            
            # Add data warning if present
            if data_warning:
                header += f"{data_warning}\n\n"
            
            self.summary_text.insert("1.0", header)

            # Display summary results
            self.display_summary(results)
            
            # Display trades in the tree
            self.display_trades(results)

            # Switch to summary tab to show results
            self.results_notebook.select(self.summary_frame)

        except Exception as e:
            self.logger.error(f"Error displaying results: {e}")
            messagebox.showerror("Display Error", f"Error displaying results: {e}")
            
            # Reset UI on error
            self.progress.stop()
            self.run_btn.config(state="normal", text="🚀 Run Backtest")


    def display_summary(self, results):
        """Display summary statistics in the summary text widget."""
        summary = results.get("summary", {})
        performance = results.get("performance", {})
        comp_stats = results.get("comprehensive_stats", {})

        # Build summary text
        summary_text = ""

        # Comprehensive stats if available (RealTest format)
        if comp_stats:
            summary_text += "🎯 COMPREHENSIVE STATISTICS\n" + "-" * 30 + "\n"
            summary_text += f"Test: {comp_stats.get('test_name', 'ADX Breakout Strategy')}\n"
            summary_text += f"Dates: {comp_stats.get('date_range', 'N/A')}\n"
            summary_text += f"Periods: {comp_stats.get('periods', 0)}\n"
            summary_text += f"NetProfit: ${comp_stats.get('net_profit', 0):,.0f}\n"
            summary_text += f"ROR: {comp_stats.get('ror', 0):.2f}%\n"
            summary_text += f"MaxDD: {comp_stats.get('max_dd', 0):.2f}%\n"
            summary_text += f"MAR: {comp_stats.get('mar', 0):.2f}\n"
            summary_text += f"Trades: {comp_stats.get('trades', 0)}\n"
            summary_text += f"PctWins: {comp_stats.get('pct_wins', 0):.2f}%\n"
            summary_text += f"Expectancy: {comp_stats.get('expectancy', 0):.2f}%\n"
            summary_text += f"AvgWin: {comp_stats.get('avg_win', 0):.2f}%\n"
            summary_text += f"AvgLoss: {comp_stats.get('avg_loss', 0):.2f}%\n"
            summary_text += f"ProfitFactor: {comp_stats.get('profit_factor', 0):.2f}\n"
            summary_text += f"Sharpe: {comp_stats.get('sharpe', 0):.2f}\n\n"

        # Portfolio performance
        if summary:
            equity = summary.get("equity", {})
            trades = summary.get("trades", {})

            summary_text += "📈 PORTFOLIO PERFORMANCE\n" + "-" * 30 + "\n"
            summary_text += f"Initial Equity: ${equity.get('initial', 0):,.2f}\n"
            summary_text += f"Final Equity: ${equity.get('current', 0):,.2f}\n"
            summary_text += f"Total Return: ${equity.get('total_return', 0):,.2f}\n"
            summary_text += f"Total Return %: {equity.get('total_return_pct', 0):.2f}%\n\n"

            summary_text += "🎯 TRADING STATISTICS\n" + "-" * 30 + "\n"
            summary_text += f"Total Trades: {trades.get('total_trades', 0)}\n"
            summary_text += f"Winning Trades: {trades.get('winning_trades', 0)}\n"
            summary_text += f"Win Rate: {trades.get('win_rate', 0):.1f}%\n"
            summary_text += f"Profit Factor: {trades.get('profit_factor', 0):.2f}\n"
            summary_text += f"Avg Holding Days: {trades.get('avg_holding_days', 0):.1f}\n"
            summary_text += f"Best Trade: ${trades.get('best_trade', 0):.2f}\n"
            summary_text += f"Worst Trade: ${trades.get('worst_trade', 0):.2f}\n\n"

        # Risk metrics
        if performance:
            summary_text += "📊 RISK METRICS\n" + "-" * 30 + "\n"
            summary_text += f"Max Drawdown: {performance.get('max_drawdown', 0) * 100:.2f}%\n"
            summary_text += f"Volatility: {performance.get('volatility', 0) * 100:.2f}%\n"
            summary_text += f"Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}\n"
            summary_text += f"Calmar Ratio: {performance.get('calmar_ratio', 0):.2f}\n\n"

        # Append to existing content in summary_text widget
        self.summary_text.insert("end", summary_text)
        self.summary_text.config(state="disabled")


    def display_trades(self, results):
        """Display individual trades in the trades tree."""
        # Clear existing trades
        for item in self.trades_tree.get_children():
            self.trades_tree.delete(item)

        trades = results.get("trades", [])
        
        if not trades:
            # Add a placeholder row if no trades
            self.trades_tree.insert("", "end", values=(
                "No trades", "", "", "", "", "", "", ""
            ))
            return

        for trade in trades:
            # Format trade data for display
            values = (
                trade.get("symbol", ""),
                trade.get("entry_date", ""),
                trade.get("exit_date", ""),
                trade.get("shares", 0),
                f"${trade.get('entry_price', 0):.2f}",
                f"${trade.get('exit_price', 0):.2f}",
                f"${trade.get('realized_pnl', 0):.2f}",
                f"{trade.get('return_pct', 0):.2f}%",
            )
            self.trades_tree.insert("", "end", values=values)


    def clear_results(self):
        """Clear previous results."""
        # Clear summary text
        self.summary_text.config(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.config(state="disabled")

        # Clear trades tree
        for item in self.trades_tree.get_children():
            self.trades_tree.delete(item)


class LiveTradingTab:
    """Live trading tab."""

    def __init__(self, parent, config, trading_callback):
        self.parent = parent
        self.config = config
        self.trading_callback = trading_callback
        self.logger = get_logger("gui_live")
        self.is_trading = False

        self.setup_ui()

    def setup_ui(self):
        """Setup live trading tab UI."""
        # Main container
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Trading Controls", padding=10)
        control_frame.pack(fill="x", pady=(0, 10))

        # Trading mode
        ttk.Label(control_frame, text="Mode:").grid(row=0, column=0, sticky="w", pady=5)
        self.mode_var = tk.StringVar(value="Paper Trading")
        mode_combo = ttk.Combobox(
            control_frame,
            textvariable=self.mode_var,
            values=["Paper Trading", "Live Trading"],
            state="readonly",
        )
        mode_combo.grid(row=0, column=1, sticky="w", padx=(10, 0), pady=5)

        # Update interval
        ttk.Label(control_frame, text="Update Interval:").grid(row=1, column=0, sticky="w", pady=5)
        self.interval_var = tk.StringVar(value="300")
        interval_combo = ttk.Combobox(
            control_frame,
            textvariable=self.interval_var,
            values=["60", "300", "600", "900"],
            state="readonly",
        )
        interval_combo.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=5)
        ttk.Label(control_frame, text="seconds").grid(row=1, column=2, sticky="w", padx=(5, 0), pady=5)

        # Capital
        ttk.Label(control_frame, text="Capital:").grid(row=2, column=0, sticky="w", pady=5)
        self.capital_var = tk.StringVar(value=str(self.config.ACCOUNT_SIZE))
        ttk.Entry(control_frame, textvariable=self.capital_var, width=15).grid(
            row=2, column=1, sticky="w", padx=(10, 0), pady=5
        )

        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=(15, 0))

        self.start_btn = ttk.Button(button_frame, text="🚀 Start Trading", command=self.start_trading)
        self.start_btn.pack(side="left", padx=(0, 10))

        self.stop_btn = ttk.Button(
            button_frame,
            text="🛑 Stop Trading",
            command=self.stop_trading,
            state="disabled",
        )
        self.stop_btn.pack(side="left")

        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Trading Status", padding=10)
        status_frame.pack(fill="x", pady=(0, 10))

        self.status_label = ttk.Label(status_frame, text="💤 Idle", style="Status.TLabel")
        self.status_label.pack(anchor="w")

        # Positions frame
        positions_frame = ttk.LabelFrame(main_frame, text="Current Positions", padding=5)
        positions_frame.pack(fill="both", expand=True, pady=(0, 10))

        # Positions table
        pos_columns = (
            "Symbol",
            "Shares",
            "Entry Price",
            "Current Price",
            "Market Value",
            "P&L",
            "P&L %",
        )
        self.positions_tree = ttk.Treeview(
            positions_frame,
            columns=pos_columns,
            show="headings",
            height=8,
        )

        for col in pos_columns:
            self.positions_tree.heading(col, text=col)
            self.positions_tree.column(col, width=80)

        # Scrollbar for positions
        pos_scrollbar = ttk.Scrollbar(positions_frame, orient="vertical", command=self.positions_tree.yview)
        self.positions_tree.configure(yscrollcommand=pos_scrollbar.set)

        self.positions_tree.pack(side="left", fill="both", expand=True)
        pos_scrollbar.pack(side="right", fill="y")

        # Orders frame
        orders_frame = ttk.LabelFrame(main_frame, text="Open Orders", padding=5)
        orders_frame.pack(fill="both", expand=True)

        # Orders table
        order_columns = ("Order ID", "Symbol", "Action", "Type", "Quantity", "Price", "Status")
        self.orders_tree = ttk.Treeview(
            orders_frame,
            columns=order_columns,
            show="headings",
            height=6,
        )

        for col in order_columns:
            self.orders_tree.heading(col, text=col)
            self.orders_tree.column(col, width=80)

        # Scrollbar for orders
        order_scrollbar = ttk.Scrollbar(orders_frame, orient="vertical", command=self.orders_tree.yview)
        self.orders_tree.configure(yscrollcommand=order_scrollbar.set)

        self.orders_tree.pack(side="left", fill="both", expand=True)
        order_scrollbar.pack(side="right", fill="y")

    def start_trading(self):
        """Start live trading."""
        if self.is_trading:
            return

        try:
            params = {
                "paper_mode": self.mode_var.get() == "Paper Trading",
                "update_interval": int(self.interval_var.get()),
                "capital": float(self.capital_var.get()),
                "symbols": self.config.SYMBOLS,
            }

            # Confirm live trading
            if not params["paper_mode"]:
                result = messagebox.askyesno(
                    "Live Trading Warning",
                    "⚠️ You are about to start LIVE TRADING with real money!\n\n"
                    "This will place actual orders in your brokerage account.\n"
                    "Are you sure you want to continue?",
                )
                if not result:
                    return

            # Update UI
            self.is_trading = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")

            mode_text = "📄 Paper Trading" if params["paper_mode"] else "💰 LIVE TRADING"
            self.status_label.config(text=f"🚀 Starting {mode_text}...")

            # Start trading
            self.trading_callback(params)

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start trading: {e}")
            self.reset_ui()

    def stop_trading(self):
        """Stop live trading."""
        if not self.is_trading:
            return

        result = messagebox.askyesno("Confirm Stop", "Stop trading and close all positions?")
        if result:
            self.reset_ui()
            # The actual stopping is handled by the main app

    def reset_ui(self):
        """Reset UI to idle state."""
        self.is_trading = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_label.config(text="💤 Idle")

    def on_connection_change(self, connected: bool):
        """Handle connection status changes."""
        if not connected and self.is_trading:
            self.reset_ui()
            messagebox.showwarning("Connection Lost", "Trading stopped due to connection loss")


class MonitoringTab:
    """Monitoring and analytics tab."""

    def __init__(self, parent, config, engine_callback):
        self.parent = parent
        self.config = config
        self.engine_callback = engine_callback
        self.logger = get_logger("gui_monitor")

        self.setup_ui()
        self.start_auto_refresh()

    def setup_ui(self):
        """Setup monitoring tab UI."""
        # Main container
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(control_frame, text="📊 System Monitoring", style="Title.TLabel").pack(side="left")

        ttk.Button(control_frame, text="🔄 Refresh", command=self.refresh_data).pack(side="right", padx=(0, 10))

        self.auto_refresh_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Auto Refresh", variable=self.auto_refresh_var).pack(side="right")

        # Notebook for different monitoring views
        self.monitor_notebook = ttk.Notebook(main_frame)
        self.monitor_notebook.pack(fill="both", expand=True)

        # Engine status tab
        self.engine_frame = ttk.Frame(self.monitor_notebook)
        self.monitor_notebook.add(self.engine_frame, text="🔧 Engine")

        # Performance tab
        self.performance_frame = ttk.Frame(self.monitor_notebook)
        self.monitor_notebook.add(self.performance_frame, text="📊 Performance")

        # Logs tab
        self.logs_frame = ttk.Frame(self.monitor_notebook)
        self.monitor_notebook.add(self.logs_frame, text="📝 Logs")

        # Setup individual tabs
        self.setup_engine_tab()
        self.setup_performance_tab()
        self.setup_logs_tab()

    def setup_engine_tab(self):
        """Setup engine status monitoring."""
        self.engine_text = scrolledtext.ScrolledText(
            self.engine_frame,
            wrap="word",
            font=("Consolas", 9),
            height=15,
            state="disabled",
        )
        self.engine_text.pack(fill="both", expand=True, padx=5, pady=5)

    def setup_performance_tab(self):
        """Setup performance monitoring."""
        self.performance_text = scrolledtext.ScrolledText(
            self.performance_frame,
            wrap="word",
            font=("Consolas", 9),
            height=15,
            state="disabled",
        )
        self.performance_text.pack(fill="both", expand=True, padx=5, pady=5)

    def setup_logs_tab(self):
        """Setup log monitoring."""
        self.logs_text = scrolledtext.ScrolledText(
            self.logs_frame,
            wrap="word",
            font=("Consolas", 8),
            height=20,
            state="disabled",
        )
        self.logs_text.pack(fill="both", expand=True, padx=5, pady=5)

    def start_auto_refresh(self):
        """Start automatic refresh timer."""

        def auto_refresh():
            if self.auto_refresh_var.get():
                self.refresh_data()
            # Schedule next refresh
            self.parent.after(5000, auto_refresh)  # Every 5 seconds

        # Start the refresh cycle
        self.parent.after(1000, auto_refresh)

    def refresh_data(self):
        """Refresh monitoring data."""
        try:
            engine = self.engine_callback()

            # Update engine status
            self.update_engine_status(engine)

            # Update performance
            self.update_performance_data(engine)

            # Update logs
            self.update_logs()

        except Exception as e:
            self.logger.error(f"Error refreshing monitoring data: {e}")

    def update_engine_status(self, engine):
        """Update engine status display."""
        status_text = "ENGINE STATUS\n" + "=" * 40 + "\n\n"

        if engine:
            status = engine.get_engine_status()

            status_text += f"State: {status.get('state', 'Unknown')}\n"
            status_text += f"Mode: {status.get('mode', 'Unknown')}\n"
            status_text += f"Components Initialized: {status.get('components_initialized', False)}\n\n"

            # Execution stats
            exec_stats = status.get("execution_stats", {})
            status_text += "EXECUTION STATISTICS\n" + "-" * 20 + "\n"
            status_text += f"Signals Processed: {exec_stats.get('signals_processed', 0)}\n"
            status_text += f"Orders Placed: {exec_stats.get('orders_placed', 0)}\n"
            status_text += f"Orders Failed: {exec_stats.get('orders_failed', 0)}\n"
            status_text += f"Data Updates: {exec_stats.get('data_updates', 0)}\n\n"

            # Session info
            session = status.get("current_session")
            if session:
                status_text += "CURRENT SESSION\n" + "-" * 15 + "\n"
                status_text += f"Start Time: {session.get('start_time', 'N/A')}\n"
                status_text += f"Duration: {session.get('duration_minutes', 0):.1f} minutes\n"
                status_text += f"Signals: {session.get('total_signals', 0)}\n"
                status_text += f"Trades: {session.get('total_trades', 0)}\n"
                status_text += f"Errors: {session.get('errors', 0)}\n\n"

            # Portfolio info
            portfolio = status.get("portfolio")
            if portfolio:
                status_text += "PORTFOLIO STATUS\n" + "-" * 16 + "\n"
                status_text += f"Equity: ${portfolio.get('equity', 0):,.2f}\n"
                status_text += f"Return: {portfolio.get('return_pct', 0):.2f}%\n"
                status_text += f"Positions: {portfolio.get('positions', 0)}\n"
                status_text += f"Cash: ${portfolio.get('cash', 0):,.2f}\n"
        else:
            status_text += "No trading engine active\n"

        # Update display
        self.engine_text.config(state="normal")
        self.engine_text.delete("1.0", "end")
        self.engine_text.insert("1.0", status_text)
        self.engine_text.config(state="disabled")

    def update_performance_data(self, engine):
        """Update performance metrics."""
        perf_text = "PERFORMANCE METRICS\n" + "=" * 40 + "\n\n"

        if engine and hasattr(engine, "portfolio_manager") and engine.portfolio_manager:
            try:
                summary = engine.portfolio_manager.get_portfolio_summary()

                # Portfolio performance
                equity = summary.get("equity", {})
                perf_text += "PORTFOLIO\n" + "-" * 10 + "\n"
                perf_text += f"Current Equity: ${equity.get('current', 0):,.2f}\n"
                perf_text += f"Total Return: {equity.get('total_return_pct', 0):.2f}%\n"
                perf_text += f"Cash: ${equity.get('cash', 0):,.2f}\n"
                perf_text += f"Invested: ${equity.get('invested', 0):,.2f}\n\n"

                # Positions
                positions = summary.get("positions", {})
                perf_text += "POSITIONS\n" + "-" * 10 + "\n"
                perf_text += f"Open: {positions.get('open', 0)}\n"
                perf_text += f"Closed: {positions.get('closed', 0)}\n"
                perf_text += f"Total Ever: {positions.get('total_ever', 0)}\n\n"

                # Trading stats
                trades = summary.get("trades", {})
                if trades:
                    perf_text += "TRADING\n" + "-" * 7 + "\n"
                    perf_text += f"Total Trades: {trades.get('total_trades', 0)}\n"
                    perf_text += f"Win Rate: {trades.get('win_rate', 0):.1f}%\n"
                    perf_text += f"Profit Factor: {trades.get('profit_factor', 0):.2f}\n"
                    perf_text += f"Best Trade: ${trades.get('best_trade', 0):.2f}\n"
                    perf_text += f"Worst Trade: ${trades.get('worst_trade', 0):.2f}\n\n"

                # Risk metrics
                risk = summary.get("risk", {})
                if risk:
                    perf_text += "RISK\n" + "-" * 4 + "\n"
                    perf_text += f"Exposure: {risk.get('exposure_pct', 0):.1f}%\n"
                    perf_text += f"Cash: {risk.get('cash_pct', 0):.1f}%\n"
                    perf_text += f"Positions: {risk.get('num_positions', 0)}\n"
                    perf_text += f"Concentration: {risk.get('concentration_risk', 0):.1f}%\n"

            except Exception as e:
                perf_text += f"Error retrieving performance data: {e}\n"
        else:
            perf_text += "No portfolio data available\n"

        # Update display
        self.performance_text.config(state="normal")
        self.performance_text.delete("1.0", "end")
        self.performance_text.insert("1.0", perf_text)
        self.performance_text.config(state="disabled")

    def update_logs(self):
        """Update log display with recent log entries."""
        try:
            log_file = os.path.join(
                self.config.LOG_DIR,
                f"trading_system_{datetime.now().strftime('%Y%m%d')}.log",
            )

            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                recent_lines = lines[-50:] if len(lines) > 50 else lines
                log_content = "".join(recent_lines)
            else:
                log_content = "No log file found for today\n"

            self.logs_text.config(state="normal")
            self.logs_text.delete("1.0", "end")
            self.logs_text.insert("1.0", log_content)
            self.logs_text.see("end")  # Scroll to bottom
            self.logs_text.config(state="disabled")

        except Exception as e:
            self.logger.error(f"Error reading logs: {e}")


class LogViewer:
    """Standalone log viewer window."""

    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("Log Viewer")
        self.window.geometry("800x600")
        self.window.transient(parent)

        self.setup_ui()
        self.load_logs()

    def setup_ui(self):
        """Setup log viewer UI."""
        toolbar = ttk.Frame(self.window)
        toolbar.pack(fill="x", padx=5, pady=5)

        ttk.Button(toolbar, text="Refresh", command=self.load_logs).pack(side="left", padx=(0, 5))
        ttk.Button(toolbar, text="Clear", command=self.clear_logs).pack(side="left", padx=(0, 5))
        ttk.Button(toolbar, text="Save As...", command=self.save_logs).pack(side="left", padx=(0, 5))

        self.log_text = scrolledtext.ScrolledText(
            self.window,
            wrap="none",
            font=("Consolas", 9),
        )
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

    def load_logs(self):
        """Load and display log files."""
        try:
            log_dir = "logs"
            if not os.path.exists(log_dir):
                self.log_text.insert("end", "No logs directory found\n")
                return

            log_files = [f for f in os.listdir(log_dir) if f.endswith(".log")]
            log_files.sort(reverse=True)  # Most recent first

            self.log_text.delete("1.0", "end")

            for log_file in log_files[:3]:  # Show last 3 days
                log_path = os.path.join(log_dir, log_file)
                self.log_text.insert("end", f"\n=== {log_file} ===\n")

                with open(log_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    self.log_text.insert("end", content)
                    self.log_text.insert("end", "\n")

        except Exception as e:
            self.log_text.insert("end", f"Error loading logs: {e}\n")

    def clear_logs(self):
        """Clear log display."""
        self.log_text.delete("1.0", "end")

    def save_logs(self):
        """Save logs to file."""
        filename = filedialog.asksaveasfilename(
            title="Save Logs",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if filename:
            try:
                content = self.log_text.get("1.0", "end-1c")
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(content)
                messagebox.showinfo("Success", "Logs saved successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save logs: {e}")
