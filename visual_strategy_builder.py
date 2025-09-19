"""
Visual Strategy Builder - Phase 3 Implementation
Enhanced Strategy Editor with Drag-and-Drop Interface
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

from logger import get_logger


class ConditionWidget:
    """A single condition widget (e.g., ADX > 25)"""
    
    def __init__(self, parent, condition_data=None, on_change_callback=None, on_delete_callback=None):
        self.parent = parent
        self.on_change = on_change_callback
        self.on_delete = on_delete_callback
        self.logger = get_logger("condition_widget")
        
        # Condition configuration
        self.indicators = {
            "Close": {"params": [], "description": "Closing price"},
            "Open": {"params": [], "description": "Opening price"}, 
            "High": {"params": [], "description": "High price"},
            "Low": {"params": [], "description": "Low price"},
            "Volume": {"params": [], "description": "Trading volume"},
            "SMA": {"params": ["period"], "description": "Simple Moving Average"},
            "EMA": {"params": ["period"], "description": "Exponential Moving Average"},
            "ADX": {"params": ["period"], "description": "Average Directional Index"},
            "RSI": {"params": ["period"], "description": "Relative Strength Index"},
            "MACD": {"params": ["fast", "slow", "signal"], "description": "MACD indicator"},
            "BB_Upper": {"params": ["period", "std"], "description": "Bollinger Band Upper"},
            "BB_Lower": {"params": ["period", "std"], "description": "Bollinger Band Lower"},
            "ATR": {"params": ["period"], "description": "Average True Range"},
        }
        
        self.operators = {
            ">": "greater than",
            "<": "less than", 
            ">=": "greater than or equal to",
            "<=": "less than or equal to",
            "==": "equals",
            "crosses_above": "crosses above",
            "crosses_below": "crosses below"
        }
        
        self.setup_widget(condition_data)
    
    def setup_widget(self, condition_data=None):
        """Create the condition widget UI"""
        
        # Main frame for this condition
        self.frame = ttk.Frame(self.parent)
        self.frame.pack(fill="x", padx=5, pady=2)
        
        # Left side indicator
        self.indicator_var = tk.StringVar(value=condition_data.get("indicator", "ADX") if condition_data else "ADX")
        self.indicator_combo = ttk.Combobox(
            self.frame, 
            textvariable=self.indicator_var, 
            values=list(self.indicators.keys()),
            state="readonly",
            width=12
        )
        self.indicator_combo.pack(side="left", padx=(0, 5))
        self.indicator_combo.bind("<<ComboboxSelected>>", self.on_indicator_change)
        
        # Parameters frame (dynamic based on indicator)
        self.params_frame = ttk.Frame(self.frame)
        self.params_frame.pack(side="left", padx=(0, 5))
        self.param_vars = {}
        
        # Operator
        self.operator_var = tk.StringVar(value=condition_data.get("operator", ">") if condition_data else ">")
        self.operator_combo = ttk.Combobox(
            self.frame,
            textvariable=self.operator_var,
            values=list(self.operators.keys()),
            state="readonly", 
            width=12
        )
        self.operator_combo.pack(side="left", padx=(0, 5))
        self.operator_combo.bind("<<ComboboxSelected>>", self.on_change_event)
        
        # Value/Target
        self.target_var = tk.StringVar(value=str(condition_data.get("target", "25")) if condition_data else "25")
        self.target_combo = ttk.Combobox(
            self.frame,
            textvariable=self.target_var,
            values=["Close", "Open", "High", "Low", "Volume", "SMA(20)", "EMA(20)"],
            width=15
        )
        self.target_combo.pack(side="left", padx=(0, 5))
        self.target_combo.bind("<<ComboboxSelected>>", self.on_change_event)
        self.target_combo.bind("<KeyRelease>", self.on_change_event)
        
        # Delete button
        self.delete_btn = ttk.Button(
            self.frame,
            text="×",
            width=3,
            command=self.delete_condition
        )
        self.delete_btn.pack(side="left", padx=(5, 0))
        
        # Initialize parameters
        self.update_parameters()
    
    def on_indicator_change(self, event=None):
        """Handle indicator selection change"""
        self.update_parameters()
        self.on_change_event()
    
    def update_parameters(self):
        """Update parameter inputs based on selected indicator"""
        # Clear existing parameter widgets
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        
        self.param_vars.clear()
        
        # Get parameters for selected indicator
        indicator = self.indicator_var.get()
        params = self.indicators.get(indicator, {}).get("params", [])
        
        # Create parameter inputs
        if params:
            ttk.Label(self.params_frame, text="(").pack(side="left")
            
            for i, param in enumerate(params):
                if i > 0:
                    ttk.Label(self.params_frame, text=",").pack(side="left", padx=(2, 2))
                
                # Default values for common parameters
                default_values = {
                    "period": "14", "fast": "12", "slow": "26", 
                    "signal": "9", "std": "2"
                }
                
                param_var = tk.StringVar(value=default_values.get(param, "14"))
                self.param_vars[param] = param_var
                
                param_entry = ttk.Entry(
                    self.params_frame, 
                    textvariable=param_var,
                    width=6
                )
                param_entry.pack(side="left", padx=(2, 0))
                param_entry.bind("<KeyRelease>", self.on_change_event)
            
            ttk.Label(self.params_frame, text=")").pack(side="left")
    
    def on_change_event(self, event=None):
        """Handle any change in the condition"""
        if self.on_change:
            self.on_change()
    
    def delete_condition(self):
        """Delete this condition"""
        if self.on_delete:
            self.on_delete(self)
        self.frame.destroy()
    
    def get_condition_data(self) -> Dict[str, Any]:
        """Get the current condition configuration"""
        params = {name: var.get() for name, var in self.param_vars.items()}
        
        return {
            "indicator": self.indicator_var.get(),
            "parameters": params,
            "operator": self.operator_var.get(),
            "target": self.target_var.get()
        }
    
    def get_description(self) -> str:
        """Get human-readable description of condition"""
        data = self.get_condition_data()
        indicator = data["indicator"]
        operator = self.operators.get(data["operator"], data["operator"])
        target = data["target"]
        
        # Format parameters
        if data["parameters"]:
            param_str = ",".join(data["parameters"].values())
            indicator_str = f"{indicator}({param_str})"
        else:
            indicator_str = indicator
        
        return f"{indicator_str} {operator} {target}"


class LogicWidget:
    """Widget for AND/OR logic between conditions"""
    
    def __init__(self, parent, logic="AND", on_change_callback=None):
        self.parent = parent
        self.on_change = on_change_callback
        
        self.frame = ttk.Frame(self.parent)
        self.frame.pack(fill="x", padx=20, pady=5)
        
        self.logic_var = tk.StringVar(value=logic)
        logic_combo = ttk.Combobox(
            self.frame,
            textvariable=self.logic_var,
            values=["AND", "OR"],
            state="readonly",
            width=8
        )
        logic_combo.pack()
        logic_combo.bind("<<ComboboxSelected>>", self.on_change_event)
    
    def on_change_event(self, event=None):
        if self.on_change:
            self.on_change()
    
    def get_logic(self) -> str:
        return self.logic_var.get()
    
    def destroy(self):
        self.frame.destroy()


class ConditionGroup:
    """Group of conditions with logic (for Entry or Exit)"""
    
    def __init__(self, parent, title="Conditions", on_change_callback=None):
        self.parent = parent
        self.title = title
        self.on_change = on_change_callback
        self.conditions = []
        self.logic_widgets = []
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the condition group UI"""
        # Main frame
        self.frame = ttk.LabelFrame(self.parent, text=self.title, padding=10)
        self.frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Conditions container
        self.conditions_frame = ttk.Frame(self.frame)
        self.conditions_frame.pack(fill="both", expand=True)
        
        # Add condition button
        self.add_btn = ttk.Button(
            self.frame,
            text="+ Add Condition",
            command=self.add_condition
        )
        self.add_btn.pack(pady=(10, 0))
        
        # Start with one condition
        self.add_condition()
    
    def add_condition(self):
        """Add a new condition"""
        # Add logic widget if not first condition
        if self.conditions:
            logic_widget = LogicWidget(
                self.conditions_frame, 
                on_change_callback=self.on_change_event
            )
            self.logic_widgets.append(logic_widget)
        
        # Add condition widget
        condition = ConditionWidget(
            self.conditions_frame,
            on_change_callback=self.on_change_event,
            on_delete_callback=self.delete_condition
        )
        self.conditions.append(condition)
        
        self.on_change_event()
    
    def delete_condition(self, condition_widget):
        """Delete a condition"""
        if len(self.conditions) <= 1:
            messagebox.showwarning("Cannot Delete", "At least one condition is required")
            return
        
        # Find and remove the condition
        try:
            index = self.conditions.index(condition_widget)
            self.conditions.pop(index)
            
            # Remove corresponding logic widget
            if index > 0 and index <= len(self.logic_widgets):
                self.logic_widgets[index-1].destroy()
                self.logic_widgets.pop(index-1)
            elif self.logic_widgets:
                self.logic_widgets[0].destroy()
                self.logic_widgets.pop(0)
            
            self.on_change_event()
            
        except ValueError:
            pass
    
    def on_change_event(self):
        """Handle condition group changes"""
        if self.on_change:
            self.on_change()
    
    def get_conditions_data(self) -> List[Dict]:
        """Get all conditions data with logic"""
        data = []
        for i, condition in enumerate(self.conditions):
            condition_data = condition.get_condition_data()
            if i < len(self.logic_widgets):
                condition_data["next_logic"] = self.logic_widgets[i].get_logic()
            data.append(condition_data)
        return data
    
    def get_description(self) -> str:
        """Get human-readable description"""
        if not self.conditions:
            return "No conditions"
        
        desc_parts = []
        for i, condition in enumerate(self.conditions):
            desc_parts.append(condition.get_description())
            if i < len(self.logic_widgets):
                desc_parts.append(f" {self.logic_widgets[i].get_logic()} ")
        
        return "".join(desc_parts)


class StrategyCodeGenerator:
    """Generates Python code from visual strategy definition"""
    
    def __init__(self):
        self.logger = get_logger("code_generator")
    
    def generate_strategy_code(self, strategy_name: str, entry_conditions: List[Dict], 
                              exit_conditions: List[Dict], risk_settings: Dict) -> str:
        """Generate complete Python strategy code"""
        
        code_parts = []
        
        # Header and imports
        code_parts.append(f'"""')
        code_parts.append(f'{strategy_name}')
        code_parts.append(f'Generated by Visual Strategy Builder on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        code_parts.append(f'"""')
        code_parts.append('')
        code_parts.append('import pandas as pd')
        code_parts.append('import numpy as np')
        code_parts.append('')
        
        # Strategy class
        code_parts.append(f'class {strategy_name.replace(" ", "")}Strategy:')
        code_parts.append('    """Auto-generated trading strategy"""')
        code_parts.append('')
        code_parts.append('    def __init__(self, config):')
        code_parts.append('        self.config = config')
        code_parts.append('        self.name = "{}"'.format(strategy_name))
        code_parts.append('')
        
        # Generate entry signals method
        code_parts.append('    def generate_entry_signals(self, data):')
        code_parts.append('        """Generate entry signals based on conditions"""')
        
        # Add indicator calculations
        indicators_used = self.get_indicators_used(entry_conditions)
        for indicator, params in indicators_used.items():
            code_parts.append(f'        {self.generate_indicator_code(indicator, params)}')
        
        code_parts.append('')
        code_parts.append('        # Entry logic')
        entry_logic = self.generate_condition_logic(entry_conditions)
        code_parts.append(f'        entry_signals = {entry_logic}')
        code_parts.append('        return entry_signals')
        code_parts.append('')
        
        # Generate exit signals method
        code_parts.append('    def generate_exit_signals(self, data, positions):')
        code_parts.append('        """Generate exit signals based on conditions"""')
        
        # Add indicator calculations for exits
        exit_indicators = self.get_indicators_used(exit_conditions)
        for indicator, params in exit_indicators.items():
            if indicator not in indicators_used:  # Don't duplicate
                code_parts.append(f'        {self.generate_indicator_code(indicator, params)}')
        
        code_parts.append('')
        code_parts.append('        # Exit logic')
        exit_logic = self.generate_condition_logic(exit_conditions)
        code_parts.append(f'        exit_signals = {exit_logic}')
        code_parts.append('        return exit_signals')
        code_parts.append('')
        
        # Risk management
        code_parts.append('    def get_position_size(self, signal, portfolio_value):')
        code_parts.append('        """Calculate position size based on risk settings"""')
        risk_pct = risk_settings.get('position_size_pct', 5.0)
        code_parts.append(f'        return portfolio_value * {risk_pct / 100}')
        code_parts.append('')
        
        # Stop loss and take profit
        code_parts.append('    def get_stop_loss(self, entry_price, signal):')
        code_parts.append('        """Calculate stop loss price"""')
        stop_pct = risk_settings.get('stop_loss_pct', 2.0)
        code_parts.append(f'        return entry_price * (1 - {stop_pct / 100})')
        code_parts.append('')
        
        code_parts.append('    def get_take_profit(self, entry_price, signal):')
        code_parts.append('        """Calculate take profit price"""')
        tp_pct = risk_settings.get('take_profit_pct', 6.0)
        code_parts.append(f'        return entry_price * (1 + {tp_pct / 100})')
        
        return '\n'.join(code_parts)
    
    def get_indicators_used(self, conditions: List[Dict]) -> Dict[str, Dict]:
        """Extract all indicators used in conditions"""
        indicators = {}
        for condition in conditions:
            indicator = condition["indicator"]
            params = condition["parameters"]
            
            if indicator not in ["Close", "Open", "High", "Low", "Volume"]:
                indicators[indicator] = params
                
            # Check target for indicators
            target = condition["target"]
            if target.startswith(("SMA", "EMA", "ADX", "RSI")):
                # Parse target like "SMA(20)"
                if "(" in target:
                    ind_name = target.split("(")[0]
                    ind_params = target.split("(")[1].rstrip(")").split(",")
                    param_dict = {"period": ind_params[0].strip()}
                    indicators[ind_name] = param_dict
        
        return indicators
    
    def generate_indicator_code(self, indicator: str, params: Dict) -> str:
        """Generate code for calculating an indicator"""
        if indicator == "SMA":
            period = params.get("period", "20")
            return f"sma_{period} = data['close'].rolling({period}).mean()"
        
        elif indicator == "EMA":
            period = params.get("period", "20") 
            return f"ema_{period} = data['close'].ewm(span={period}).mean()"
        
        elif indicator == "ADX":
            period = params.get("period", "14")
            return f"adx_{period} = calculate_adx(data, {period})  # Custom ADX function"
        
        elif indicator == "RSI":
            period = params.get("period", "14")
            return f"rsi_{period} = calculate_rsi(data, {period})  # Custom RSI function"
        
        elif indicator == "MACD":
            fast = params.get("fast", "12")
            slow = params.get("slow", "26")
            signal = params.get("signal", "9")
            return f"macd_{fast}_{slow}_{signal} = calculate_macd(data, {fast}, {slow}, {signal})"
        
        else:
            return f"# {indicator} calculation needed"
    
    def generate_condition_logic(self, conditions: List[Dict]) -> str:
        """Generate the logical expression for conditions"""
        if not conditions:
            return "pd.Series([False] * len(data))"
        
        logic_parts = []
        for condition in conditions:
            logic_parts.append(self.generate_single_condition(condition))
            
            # Add logic operator if not last condition
            if "next_logic" in condition:
                logic_op = " & " if condition["next_logic"] == "AND" else " | "
                logic_parts.append(logic_op)
        
        # Wrap in parentheses for safety
        return "(" + "".join(logic_parts) + ")"
    
    def generate_single_condition(self, condition: Dict) -> str:
        """Generate code for a single condition"""
        indicator = condition["indicator"]
        params = condition["parameters"]
        operator = condition["operator"]
        target = condition["target"]
        
        # Convert operator
        op_map = {
            ">": ">", "<": "<", ">=": ">=", "<=": "<=", "==": "==",
            "crosses_above": ">", "crosses_below": "<"  # Simplified for now
        }
        py_operator = op_map.get(operator, ">")
        
        # Generate left side (indicator)
        if indicator in ["Close", "Open", "High", "Low", "Volume"]:
            left_side = f"data['{indicator.lower()}']"
        else:
            # Use pre-calculated indicator
            if indicator == "SMA":
                period = params.get("period", "20")
                left_side = f"sma_{period}"
            elif indicator == "EMA":
                period = params.get("period", "20")
                left_side = f"ema_{period}"
            elif indicator == "ADX":
                period = params.get("period", "14")
                left_side = f"adx_{period}"
            elif indicator == "RSI":
                period = params.get("period", "14")
                left_side = f"rsi_{period}"
            else:
                left_side = f"{indicator.lower()}_indicator"
        
        # Generate right side (target)
        if target.replace(".", "").replace("-", "").isdigit():
            right_side = target
        elif target in ["Close", "Open", "High", "Low", "Volume"]:
            right_side = f"data['{target.lower()}']"
        else:
            # Handle targets like "SMA(20)"
            right_side = target.lower().replace("(", "_").replace(")", "").replace(",", "_")
        
        return f"({left_side} {py_operator} {right_side})"


class VisualStrategyBuilder:
    """Main Visual Strategy Builder interface"""
    
    def __init__(self, parent, config, strategy_callback=None):
        self.parent = parent
        self.config = config
        self.strategy_callback = strategy_callback
        self.logger = get_logger("visual_strategy_builder")
        
        self.code_generator = StrategyCodeGenerator()
        self.current_strategy_code = ""
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the main visual builder interface"""
        
        # Main container with notebook for different views
        self.main_notebook = ttk.Notebook(self.parent)
        self.main_notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Visual Builder tab
        self.builder_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.builder_frame, text="🎨 Visual Builder")
        
        # Generated Code tab
        self.code_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.code_frame, text="🐍 Generated Code")
        
        # Templates tab
        self.templates_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.templates_frame, text="📚 Templates")
        
        self.setup_builder_tab()
        self.setup_code_tab()
        self.setup_templates_tab()
    
    def setup_builder_tab(self):
        """Setup the visual strategy builder"""
        
        # Top toolbar
        toolbar = ttk.Frame(self.builder_frame)
        toolbar.pack(fill="x", padx=5, pady=5)
        
        # Strategy name
        ttk.Label(toolbar, text="Strategy Name:").pack(side="left", padx=(0, 5))
        self.strategy_name_var = tk.StringVar(value="My ADX Strategy")
        name_entry = ttk.Entry(toolbar, textvariable=self.strategy_name_var, width=20)
        name_entry.pack(side="left", padx=(0, 20))
        
        # Action buttons
        ttk.Button(toolbar, text="📝 Generate Code", command=self.generate_code).pack(side="left", padx=(0, 5))
        ttk.Button(toolbar, text="🧪 Test Strategy", command=self.test_strategy).pack(side="left", padx=(0, 5))
        ttk.Button(toolbar, text="💾 Save Template", command=self.save_template).pack(side="left", padx=(0, 5))
        
        # Main content area with left/right split
        content_frame = ttk.Frame(self.builder_frame)
        content_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Left side - Conditions
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Entry conditions
        self.entry_group = ConditionGroup(
            left_frame, 
            "🚀 Entry Conditions",
            on_change_callback=self.on_strategy_change
        )
        
        # Exit conditions  
        self.exit_group = ConditionGroup(
            left_frame,
            "🛑 Exit Conditions", 
            on_change_callback=self.on_strategy_change
        )
        
        # Right side - Settings & Preview
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side="right", fill="y", padx=(10, 0))
        
        self.setup_settings_panel(right_frame)
        self.setup_preview_panel(right_frame)

        # NOW call update_preview after all widgets exist:
        self.update_preview()  # ← MOVE this call to the END
    
    def setup_settings_panel(self, parent):
        """Setup risk management and strategy settings"""
        settings_frame = ttk.LabelFrame(parent, text="⚙️ Risk Management", padding=10)
        settings_frame.pack(fill="x", pady=(0, 10))
        
        # Position size
        ttk.Label(settings_frame, text="Position Size (%):").grid(row=0, column=0, sticky="w", pady=2)
        self.position_size_var = tk.StringVar(value="5.0")
        ttk.Entry(settings_frame, textvariable=self.position_size_var, width=10).grid(row=0, column=1, padx=(10, 0), pady=2)
        
        # Stop loss
        ttk.Label(settings_frame, text="Stop Loss (%):").grid(row=1, column=0, sticky="w", pady=2)
        self.stop_loss_var = tk.StringVar(value="2.0")
        ttk.Entry(settings_frame, textvariable=self.stop_loss_var, width=10).grid(row=1, column=1, padx=(10, 0), pady=2)
        
        # Take profit
        ttk.Label(settings_frame, text="Take Profit (%):").grid(row=2, column=0, sticky="w", pady=2)
        self.take_profit_var = tk.StringVar(value="6.0")
        ttk.Entry(settings_frame, textvariable=self.take_profit_var, width=10).grid(row=2, column=1, padx=(10, 0), pady=2)
        
        # Bind change events
        for var in [self.position_size_var, self.stop_loss_var, self.take_profit_var]:
            var.trace("w", lambda *args: self.on_strategy_change())
    
    def setup_preview_panel(self, parent):
        """Setup strategy preview panel"""
        preview_frame = ttk.LabelFrame(parent, text="📋 Strategy Preview", padding=10)
        preview_frame.pack(fill="both", expand=True)
        
        self.preview_text = scrolledtext.ScrolledText(
            preview_frame,
            wrap="word",
            height=15,
            width=40,
            font=("Consolas", 9),
            state="disabled"
        )
        self.preview_text.pack(fill="both", expand=True)
        
        # Initial preview
        self.update_preview()
    
    def setup_code_tab(self):
        """Setup generated code display"""
        # Toolbar
        code_toolbar = ttk.Frame(self.code_frame)
        code_toolbar.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(code_toolbar, text="🔄 Regenerate", command=self.generate_code).pack(side="left", padx=(0, 5))
        ttk.Button(code_toolbar, text="📋 Copy Code", command=self.copy_code).pack(side="left", padx=(0, 5))
        ttk.Button(code_toolbar, text="💾 Save as .py", command=self.save_code).pack(side="left", padx=(0, 5))
        
        # Code display
        self.code_text = scrolledtext.ScrolledText(
            self.code_frame,
            wrap="none",
            font=("Consolas", 10),
            state="disabled"
        )
        self.code_text.pack(fill="both", expand=True, padx=5, pady=5)
    
    def setup_templates_tab(self):
        """Setup strategy templates"""
        ttk.Label(
            self.templates_frame,
            text="🚧 Strategy Templates Coming Soon!\n\nWill include:\n• ADX Breakout\n• RSI Mean Reversion\n• Moving Average Cross\n• Bollinger Band Squeeze",
            font=("Arial", 12),
            justify="center"
        ).pack(expand=True)
    
    def on_strategy_change(self):
        """Handle any change in the strategy definition"""
        self.update_preview()
    
    def update_preview(self):
        """Update the strategy preview"""
        
        # Check if preview_text widget exists
        if not hasattr(self, 'preview_text') or not self.preview_text.winfo_exists():
            # Preview widget not ready yet, skip update
            return
        
        try:
            self.preview_text.config(state="normal")
            self.preview_text.delete("1.0", "end")
            
            preview_text = f"📊 {self.strategy_name_var.get()}\n"
            preview_text += "=" * 40 + "\n\n"
            
            # Entry conditions
            preview_text += "🚀 ENTRY CONDITIONS:\n"
            preview_text += f"   {self.entry_group.get_description()}\n\n"
            
            # Exit conditions  
            preview_text += "🛑 EXIT CONDITIONS:\n"
            preview_text += f"   {self.exit_group.get_description()}\n\n"
            
            # Risk settings
            preview_text += "⚙️ RISK MANAGEMENT:\n"
            preview_text += f"   Position Size: {self.position_size_var.get()}%\n"
            preview_text += f"   Stop Loss: {self.stop_loss_var.get()}%\n"
            preview_text += f"   Take Profit: {self.take_profit_var.get()}%\n"
            
            self.preview_text.insert("1.0", preview_text)
            self.preview_text.config(state="disabled")
            
        except tk.TclError as e:
            # Widget destroyed or not ready, ignore
            pass
        except Exception as e:
            # Log error but don't crash
            self.logger.error(f"Preview update failed: {e}")
    
    def generate_code(self):
        """Generate Python code from visual strategy"""
        try:
            strategy_name = self.strategy_name_var.get()
            entry_conditions = self.entry_group.get_conditions_data()
            exit_conditions = self.exit_group.get_conditions_data()
            
            risk_settings = {
                "position_size_pct": float(self.position_size_var.get()),
                "stop_loss_pct": float(self.stop_loss_var.get()),
                "take_profit_pct": float(self.take_profit_var.get())
            }
            
            # Generate the code
            self.current_strategy_code = self.code_generator.generate_strategy_code(
                strategy_name, entry_conditions, exit_conditions, risk_settings
            )
            
            # Display in code tab
            self.code_text.config(state="normal")
            self.code_text.delete("1.0", "end")
            self.code_text.insert("1.0", self.current_strategy_code)
            self.code_text.config(state="disabled")
            
            # Switch to code tab
            self.main_notebook.select(self.code_frame)
            
            self.logger.info(f"Generated code for strategy: {strategy_name}")
            messagebox.showinfo("Success", "Strategy code generated successfully!")
            
        except Exception as e:
            self.logger.error(f"Code generation failed: {e}")
            messagebox.showerror("Error", f"Failed to generate code: {e}")
    
    def test_strategy(self):
        """Test the strategy with a quick backtest"""
        try:
            if not self.current_strategy_code:
                self.generate_code()
            
            # Create a simple test configuration
            test_params = {
                "strategy_name": self.strategy_name_var.get(),
                "entry_conditions": self.entry_group.get_conditions_data(),
                "exit_conditions": self.exit_group.get_conditions_data(),
                "risk_settings": {
                    "position_size_pct": float(self.position_size_var.get()),
                    "stop_loss_pct": float(self.stop_loss_var.get()),
                    "take_profit_pct": float(self.take_profit_var.get())
                }
            }
            
            if self.strategy_callback:
                self.strategy_callback(test_params)
            
            self.logger.info("Strategy test initiated")
            
        except Exception as e:
            self.logger.error(f"Strategy test failed: {e}")
            messagebox.showerror("Error", f"Failed to test strategy: {e}")
    
    def copy_code(self):
        """Copy generated code to clipboard"""
        if self.current_strategy_code:
            self.parent.clipboard_clear()
            self.parent.clipboard_append(self.current_strategy_code)
            messagebox.showinfo("Copied", "Strategy code copied to clipboard!")
        else:
            messagebox.showwarning("No Code", "Please generate code first")
    
    def save_code(self):
        """Save generated code as Python file"""
        if not self.current_strategy_code:
            messagebox.showwarning("No Code", "Please generate code first")
            return
        
        from tkinter import filedialog
        
        filename = filedialog.asksaveasfilename(
            title="Save Strategy Code",
            defaultextension=".py",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.current_strategy_code)
                messagebox.showinfo("Saved", f"Strategy code saved to {filename}")
                self.logger.info(f"Strategy code saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {e}")
    
    def save_template(self):
        """Save current strategy as template"""
        try:
            template_data = {
                "name": self.strategy_name_var.get(),
                "entry_conditions": self.entry_group.get_conditions_data(),
                "exit_conditions": self.exit_group.get_conditions_data(),
                "risk_settings": {
                    "position_size_pct": float(self.position_size_var.get()),
                    "stop_loss_pct": float(self.stop_loss_var.get()),
                    "take_profit_pct": float(self.take_profit_var.get())
                },
                "created_date": datetime.now().isoformat()
            }
            
            from tkinter import filedialog
            
            filename = filedialog.asksaveasfilename(
                title="Save Strategy Template",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'w') as f:
                    json.dump(template_data, f, indent=2)
                messagebox.showinfo("Saved", f"Strategy template saved to {filename}")
                self.logger.info(f"Strategy template saved to {filename}")
                
        except Exception as e:
            self.logger.error(f"Failed to save template: {e}")
            messagebox.showerror("Error", f"Failed to save template: {e}")
    
    def load_template(self, filename):
        """Load strategy from template file"""
        try:
            with open(filename, 'r') as f:
                template_data = json.load(f)
            
            # Load strategy name
            self.strategy_name_var.set(template_data.get("name", "Loaded Strategy"))
            
            # Load risk settings
            risk_settings = template_data.get("risk_settings", {})
            self.position_size_var.set(str(risk_settings.get("position_size_pct", 5.0)))
            self.stop_loss_var.set(str(risk_settings.get("stop_loss_pct", 2.0)))
            self.take_profit_var.set(str(risk_settings.get("take_profit_pct", 6.0)))
            
            # TODO: Load conditions (requires more complex reconstruction)
            # This would involve recreating ConditionWidgets from saved data
            
            self.update_preview()
            messagebox.showinfo("Loaded", "Strategy template loaded successfully!")
            
        except Exception as e:
            self.logger.error(f"Failed to load template: {e}")
            messagebox.showerror("Error", f"Failed to load template: {e}")


# ============================================================================= 
# INTEGRATION WITH EXISTING GUI
# =============================================================================

class EnhancedStrategyTab:
    """Enhanced Strategy Tab with Visual Builder"""
    
    def __init__(self, parent, config, strategy_callback):
        self.parent = parent
        self.config = config
        self.strategy_callback = strategy_callback
        self.logger = get_logger("enhanced_strategy")
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup enhanced strategy tab with mode switcher"""
        
        # Main container
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Top toolbar with mode switcher
        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill="x", pady=(0, 10))
        
        ttk.Label(toolbar, text="Strategy Editor Mode:", font=("Arial", 10, "bold")).pack(side="left", padx=(0, 10))
        
        self.mode_var = tk.StringVar(value="visual")
        mode_frame = ttk.Frame(toolbar)
        mode_frame.pack(side="left")
        
        ttk.Radiobutton(
            mode_frame, 
            text="🎨 Visual Builder", 
            variable=self.mode_var, 
            value="visual",
            command=self.switch_mode
        ).pack(side="left", padx=(0, 10))
        
        ttk.Radiobutton(
            mode_frame,
            text="🐍 Code Editor", 
            variable=self.mode_var,
            value="code",
            command=self.switch_mode
        ).pack(side="left")
        
        # Content area
        self.content_frame = ttk.Frame(main_frame)
        self.content_frame.pack(fill="both", expand=True)
        
        # Initialize visual builder
        self.visual_builder = VisualStrategyBuilder(
            self.content_frame, 
            self.config,
            strategy_callback=self.test_visual_strategy
        )
        
        # Initialize code editor (hidden initially)
        self.code_editor_frame = ttk.Frame(self.content_frame)
        self.setup_code_editor()
        
        # Start with visual mode
        self.switch_mode()
    
    def setup_code_editor(self):
        """Setup traditional code editor"""
        
        # Toolbar
        toolbar = ttk.Frame(self.code_editor_frame)
        toolbar.pack(fill="x", pady=(0, 10))
        
        ttk.Button(toolbar, text="New", command=self.new_strategy).pack(side="left", padx=(0, 5))
        ttk.Button(toolbar, text="Open", command=self.open_strategy).pack(side="left", padx=(0, 5))
        ttk.Button(toolbar, text="Save", command=self.save_strategy).pack(side="left", padx=(0, 5))
        ttk.Button(toolbar, text="Test", command=self.test_code_strategy).pack(side="left", padx=(0, 10))
        
        # Code editor
        self.code_editor = scrolledtext.ScrolledText(
            self.code_editor_frame,
            wrap="none",
            font=("Consolas", 10),
            undo=True,
            maxundo=50,
        )
        self.code_editor.pack(fill="both", expand=True)
        
        # Load default strategy template
        self.load_default_strategy()
    
    def switch_mode(self):
        """Switch between visual and code modes"""
        mode = self.mode_var.get()
        
        # Hide all frames
        for widget in self.content_frame.winfo_children():
            widget.pack_forget()
        
        if mode == "visual":
            self.visual_builder.main_notebook.pack(fill="both", expand=True)
        else:
            self.code_editor_frame.pack(fill="both", expand=True)
        
        self.logger.info(f"Switched to {mode} mode")
    
    def test_visual_strategy(self, strategy_params):
        """Test strategy created with visual builder"""
        self.logger.info("Testing visual strategy...")
        
        # Convert visual strategy to format expected by backtesting engine
        backtest_params = {
            "strategy_type": "visual_generated",
            "strategy_name": strategy_params["strategy_name"],
            "entry_conditions": strategy_params["entry_conditions"],
            "exit_conditions": strategy_params["exit_conditions"],
            "risk_settings": strategy_params["risk_settings"]
        }
        
        if self.strategy_callback:
            self.strategy_callback(backtest_params)
    
    def test_code_strategy(self):
        """Test manually coded strategy"""
        code = self.code_editor.get("1.0", "end-1c")
        
        backtest_params = {
            "strategy_type": "custom_code",
            "strategy_code": code
        }
        
        if self.strategy_callback:
            self.strategy_callback(backtest_params)
    
    def new_strategy(self):
        """Create new strategy"""
        self.code_editor.delete("1.0", "end")
        self.load_default_strategy()
    
    def open_strategy(self):
        """Open existing strategy"""
        from tkinter import filedialog
        
        filename = filedialog.askopenfilename(
            title="Open Strategy",
            filetypes=[("Python files", "*.py"), ("JSON templates", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    content = f.read()
                
                if filename.endswith('.json'):
                    # Load as visual template
                    self.mode_var.set("visual")
                    self.switch_mode()
                    self.visual_builder.load_template(filename)
                else:
                    # Load as code
                    self.mode_var.set("code")
                    self.switch_mode()
                    self.code_editor.delete("1.0", "end")
                    self.code_editor.insert("1.0", content)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open file: {e}")
    
    def save_strategy(self):
        """Save current strategy"""
        from tkinter import filedialog
        
        if self.mode_var.get() == "visual":
            self.visual_builder.save_template()
        else:
            filename = filedialog.asksaveasfilename(
                title="Save Strategy",
                defaultextension=".py",
                filetypes=[("Python files", "*.py"), ("All files", "*.*")]
            )
            
            if filename:
                try:
                    code = self.code_editor.get("1.0", "end-1c")
                    with open(filename, 'w') as f:
                        f.write(code)
                    messagebox.showinfo("Saved", f"Strategy saved to {filename}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save file: {e}")
    
    def load_default_strategy(self):
        """Load default strategy template"""
        default_strategy = '''"""
ADX Breakout Strategy Template

Customize this strategy or switch to Visual Builder mode
for drag-and-drop strategy creation.
"""

import pandas as pd
import numpy as np

class CustomStrategy:
    """Custom trading strategy"""
    
    def __init__(self, config):
        self.config = config
        self.name = "Custom ADX Strategy"
    
    def generate_entry_signals(self, data):
        """Generate buy/sell entry signals"""
        
        # Example: ADX breakout strategy
        adx = self.calculate_adx(data, 14)
        sma_20 = data['close'].rolling(20).mean()
        
        # Entry conditions
        entry_signals = (
            (adx > 25) &  # Strong trend
            (data['close'] > sma_20) &  # Above moving average
            (data['volume'] > data['volume'].rolling(10).mean())  # Volume confirmation
        )
        
        return entry_signals
    
    def generate_exit_signals(self, data, positions):
        """Generate exit signals"""
        
        sma_20 = data['close'].rolling(20).mean()
        
        # Exit when price falls below moving average
        exit_signals = data['close'] < sma_20
        
        return exit_signals
    
    def calculate_adx(self, data, period=14):
        """Calculate ADX indicator"""
        # Implement your ADX calculation here
        # This is a placeholder
        return pd.Series([25] * len(data), index=data.index)
    
    def get_position_size(self, signal, portfolio_value):
        """Calculate position size"""
        return portfolio_value * 0.05  # 5% of portfolio
    
    def get_stop_loss(self, entry_price, signal):
        """Calculate stop loss"""
        return entry_price * 0.98  # 2% stop loss
    
    def get_take_profit(self, entry_price, signal):
        """Calculate take profit"""
        return entry_price * 1.06  # 6% take profit
'''
        self.code_editor.insert("1.0", default_strategy)


# =============================================================================
# USAGE EXAMPLE: How to integrate with existing GUI
# =============================================================================

"""
To integrate this Visual Strategy Builder with your existing GUI:

1. REPLACE your existing StrategyTab in gui_components.py with EnhancedStrategyTab

2. UPDATE gui_main.py initialize_tabs method:
   
   # Old:
   self.strategy_tab = StrategyTab(
       self.strategy_frame, 
       self.config,
       self.on_strategy_change
   )
   
   # New: 
   self.strategy_tab = EnhancedStrategyTab(
       self.strategy_frame,
       self.config, 
       self.test_visual_strategy_callback
   )

3. ADD callback method to gui_main.py:
   
   def test_visual_strategy_callback(self, strategy_params):
       \"\"\"Test strategy created with visual builder\"\"\"
       # Convert to backtest parameters and run
       backtest_params = {
           "symbols": self.config.SYMBOLS,
           "start_date": "2023-01-01",
           "end_date": "2024-12-31", 
           "initial_capital": self.config.ACCOUNT_SIZE,
           "strategy_config": strategy_params
       }
       
       # Run backtest with visual strategy
       self.run_backtest_callback(backtest_params)

4. UPDATE your trading engine to handle visual strategies:
   - Add strategy_config parameter parsing
   - Generate entry/exit signals based on visual conditions
   - Apply risk management settings
"""