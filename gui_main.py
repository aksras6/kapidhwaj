"""
ADX Trading System - Desktop GUI Application

Main entry point for the desktop trading application.
Built with tkinter for maximum compatibility and ease of deployment.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import sys
import os
from datetime import datetime
import webbrowser

# Import our backend components
from config import TradingConfig, get_config_for_mode
from logger import setup_logging, get_logger
from trading_engine import create_trading_engine, TradingMode
from gui_components import (
    ConnectionTab, EnhancedStrategyTab, BacktestTab, 
    LiveTradingTab, MonitoringTab, LogViewer
)

# Import IB components if available
try:
    from ibapi_appv1 import IbApp, IBGateway
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False


class TradingSystemGUI:
    """Main GUI application class"""
    
    def __init__(self):
        # Setup logging first
        setup_logging()
        self.logger = get_logger("gui_main")
        
        # Initialize main window
        self.root = tk.Tk()
        self.root.title("ADX Breakout Trading System v2.0")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 600)
        
        # Configure style
        self.setup_styles()
        
        # Initialize state
        self.config = TradingConfig()
        self.trading_engine = None
        self.ib_connection = None
        self.is_connected = False
        
        # Setup GUI components
        self.setup_menu()
        self.setup_main_layout()
        self.setup_status_bar()
        
        # Initialize tabs
        self.initialize_tabs()
        
        # Setup window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.logger.info("Desktop GUI initialized")
    
    def setup_styles(self):
        """Configure ttk styles for modern appearance"""
        style = ttk.Style()
        
        # Use modern theme if available
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')
        elif 'vista' in available_themes:
            style.theme_use('vista')
        
        # Custom styles
        style.configure('Title.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Heading.TLabel', font=('Arial', 10, 'bold'))
        style.configure('Status.TLabel', font=('Arial', 9))
        style.configure('Success.TLabel', foreground='green')
        style.configure('Error.TLabel', foreground='red')
        style.configure('Warning.TLabel', foreground='orange')
    
    def setup_menu(self):
        """Create main menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Strategy...", command=self.new_strategy)
        file_menu.add_command(label="Open Strategy...", command=self.open_strategy)
        file_menu.add_command(label="Save Strategy", command=self.save_strategy)
        file_menu.add_separator()
        file_menu.add_command(label="Load Config...", command=self.load_config)
        file_menu.add_command(label="Save Config...", command=self.save_config)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results...", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        
        # Trading menu
        trading_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Trading", menu=trading_menu)
        trading_menu.add_command(label="Connect to IB", command=self.connect_ib)
        trading_menu.add_command(label="Disconnect", command=self.disconnect_ib)
        trading_menu.add_separator()
        trading_menu.add_command(label="Run Backtest", command=self.run_backtest)
        trading_menu.add_command(label="Start Paper Trading", command=self.start_paper_trading)
        trading_menu.add_command(label="Stop Trading", command=self.stop_trading)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Validate System", command=self.validate_system)
        tools_menu.add_command(label="View Logs", command=self.view_logs)
        tools_menu.add_command(label="Open Output Folder", command=self.open_output_folder)
        tools_menu.add_separator()
        tools_menu.add_command(label="Settings...", command=self.open_settings)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Manual", command=self.open_help)
        help_menu.add_command(label="Strategy Guide", command=self.open_strategy_guide)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about)
    
    def setup_main_layout(self):
        """Setup main application layout"""
        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill='both', expand=True)
        
        # Create frames for each tab
        self.connection_frame = ttk.Frame(self.notebook)
        self.strategy_frame = ttk.Frame(self.notebook)
        self.backtest_frame = ttk.Frame(self.notebook)
        self.live_frame = ttk.Frame(self.notebook)
        self.monitor_frame = ttk.Frame(self.notebook)
        
        # Add tabs to notebook
        self.notebook.add(self.connection_frame, text="🔌 Connection")
        self.notebook.add(self.strategy_frame, text="📝 Strategy")
        self.notebook.add(self.backtest_frame, text="📊 Backtest")
        self.notebook.add(self.live_frame, text="📈 Live Trading")
        self.notebook.add(self.monitor_frame, text="👁️ Monitoring")
        
        # Bind tab change event
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

    def test_visual_strategy_callback(self, strategy_params):
        """Test strategy created with visual builder"""
        self.logger.info(f"🎨 Visual strategy test requested: {strategy_params['strategy_name']}")
        
        try:
            # Log the strategy configuration for debugging
            self.logger.info(f"Entry conditions: {len(strategy_params.get('entry_conditions', []))}")
            self.logger.info(f"Exit conditions: {len(strategy_params.get('exit_conditions', []))}")
            self.logger.info(f"Risk settings: {strategy_params.get('risk_settings', {})}")
            
            # Convert visual strategy to backtest parameters
            backtest_params = {
                "symbols": self.config.SYMBOLS,
                "start_date": "2023-01-01",
                "end_date": "2024-12-31",
                "initial_capital": self.config.ACCOUNT_SIZE,
                # CRITICAL: Add strategy_config to pass visual strategy
                "strategy_config": strategy_params  # ← This is the key addition!
            }
            
            # Show confirmation dialog with strategy details
            entry_desc = self._describe_conditions(strategy_params.get('entry_conditions', []))
            exit_desc = self._describe_conditions(strategy_params.get('exit_conditions', []))
            
            result = messagebox.askyesno(
                "Test Visual Strategy",
                f"🎨 Test strategy: '{strategy_params['strategy_name']}'?\n\n"
                f"📈 Entry: {entry_desc}\n"
                f"📉 Exit: {exit_desc}\n\n"
                f"Period: 2023-01-01 to 2024-12-31\n"
                f"Symbols: {', '.join(self.config.SYMBOLS[:3])}{'...' if len(self.config.SYMBOLS) > 3 else ''}"
            )
            
            if result:
                # IMPORTANT: Log that we're using visual strategy
                self.logger.info("🚀 Starting backtest with VISUAL STRATEGY")
                
                # Switch to backtest tab to show results
                self.notebook.select(self.backtest_frame)
                
                # Run backtest with visual strategy configuration
                self.run_backtest_callback(backtest_params)
            
        except Exception as e:
            self.logger.error(f"Visual strategy test failed: {e}")
            messagebox.showerror("Test Failed", f"Failed to test visual strategy:\n{e}")

    def _describe_conditions(self, conditions):
        """Helper method to describe conditions in human-readable format"""
        if not conditions:
            return "None"
        
        descriptions = []
        for condition in conditions:
            indicator = condition.get('indicator', '')
            operator = condition.get('operator', '')
            target = condition.get('target', '')
            params = condition.get('parameters', {})
            
            if params:
                param_str = ",".join(params.values())
                indicator_str = f"{indicator}({param_str})"
            else:
                indicator_str = indicator
                
            descriptions.append(f"{indicator_str} {operator} {target}")
        
        return " AND ".join(descriptions)
    
    def setup_status_bar(self):
        """Setup status bar at bottom"""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(side='bottom', fill='x', padx=5, pady=2)
        
        # Connection status
        self.connection_status = ttk.Label(
            self.status_frame, 
            text="⚫ Disconnected", 
            style='Status.TLabel'
        )
        self.connection_status.pack(side='left')
        
        # Separator
        ttk.Separator(self.status_frame, orient='vertical').pack(side='left', fill='y', padx=10)
        
        # Trading status
        self.trading_status = ttk.Label(
            self.status_frame, 
            text="💤 Idle", 
            style='Status.TLabel'
        )
        self.trading_status.pack(side='left')
        
        # Separator
        ttk.Separator(self.status_frame, orient='vertical').pack(side='left', fill='y', padx=10)
        
        # Clock
        self.clock_label = ttk.Label(
            self.status_frame, 
            text="", 
            style='Status.TLabel'
        )
        self.clock_label.pack(side='right')
        
        # Start clock update
        self.update_clock()
    
    def initialize_tabs(self):
        """Initialize all tab components"""
        try:
            # Connection tab
            self.connection_tab = ConnectionTab(
                self.connection_frame, 
                self.config, 
                self.on_connection_change
            )
            
            # # Strategy tab
            # self.strategy_tab = StrategyTab(
            #     self.strategy_frame, 
            #     self.config,
            #     self.on_strategy_change
            # )

            # New: 
            self.strategy_tab = EnhancedStrategyTab(
                self.strategy_frame,
                self.config, 
                self.test_visual_strategy_callback
            )
            
            
            # Backtest tab
            self.backtest_tab = BacktestTab(
                self.backtest_frame, 
                self.config,
                self.run_backtest_callback
            )
            
            # Live trading tab
            self.live_tab = LiveTradingTab(
                self.live_frame, 
                self.config,
                self.start_live_trading_callback
            )
            
            # Monitoring tab
            self.monitor_tab = MonitoringTab(
                self.monitor_frame, 
                self.config,
                self.get_trading_engine
            )
            
            self.logger.info("All tabs initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize tabs: {e}")
            messagebox.showerror("Initialization Error", f"Failed to initialize GUI components:\n{e}")
    
    def update_clock(self):
        """Update clock display"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.clock_label.config(text=current_time)
        self.root.after(1000, self.update_clock)  # Update every second
    
    def on_tab_changed(self, event):
        """Handle tab change events"""
        selected_tab = event.widget.select()
        tab_text = event.widget.tab(selected_tab, "text")
        self.logger.debug(f"Switched to tab: {tab_text}")
        
        # Refresh tab content if needed
        if "Monitoring" in tab_text and hasattr(self, 'monitor_tab'):
            self.monitor_tab.refresh_data()
    
    # ===== CONNECTION MANAGEMENT =====
    
    def connect_ib(self):
        """Connect to Interactive Brokers"""
        if not IB_AVAILABLE:
            messagebox.showerror("IB Not Available", 
                            "Interactive Brokers API is not available.\n"
                            "Please ensure ibapi_appv1.py is in the application directory.")
            return
            
        def connect_thread():
            try:
                self.update_status("🔄 Connecting to IB...")
                
                # Create IB connection
                app = IbApp()
                gateway = IBGateway(
                    app, 
                    self.config.IB_HOST, 
                    self.config.IB_PAPER_PORT, 
                    self.config.IB_CLIENT_ID
                )
                
                self.logger.info(f"Attempting IB connection: {self.config.IB_HOST}:{self.config.IB_PAPER_PORT}")
                gateway.start(wait_sec=10)
                
                # Store connection
                self.ib_connection = (app, gateway)
                self.is_connected = True
                
                self.logger.info("✅ IB connection established successfully")
                
                # Update UI on main thread
                self.root.after(0, self.on_connection_success)
                
            except Exception as e:
                self.logger.error(f"Connection failed: {e}")
                # Clear connection state on failure
                self.ib_connection = None
                self.is_connected = False
                self.root.after(0, lambda: self.on_connection_error(str(e)))
        
        # Start connection in background thread
        threading.Thread(target=connect_thread, daemon=True).start()
    
    def disconnect_ib(self):
        """Disconnect from Interactive Brokers"""
        if self.ib_connection:
            try:
                app, gateway = self.ib_connection
                gateway.stop()
                self.ib_connection = None
                self.is_connected = False
                self.update_connection_status("⚫ Disconnected", "red")
                self.update_status("💤 Idle")
                self.logger.info("Disconnected from IB")
                
            except Exception as e:
                self.logger.error(f"Disconnect error: {e}")
    
    def on_connection_success(self):
        """Handle successful IB connection"""
        self.update_connection_status("🟢 Connected", "green")
        self.update_status("📡 Connected to IB")
        self.logger.info("Successfully connected to IB")
        
        # Log connection details for debugging
        if self.ib_connection:
            app, gateway = self.ib_connection
            self.logger.info(f"IB connection stored: app={app is not None}, gateway={gateway is not None}")
            if app:
                self.logger.info(f"IB app connected: {app.isConnected()}")
        
        # Notify tabs
        if hasattr(self, 'connection_tab'):
            self.connection_tab.on_connection_change(True)
        if hasattr(self, 'live_tab'):
            self.live_tab.on_connection_change(True)
    
    def on_connection_error(self, error_msg):
        """Handle IB connection error"""
        self.update_connection_status("🔴 Error", "red")
        self.update_status("❌ Connection Failed")
        
        # Ensure connection state is cleared
        self.ib_connection = None
        self.is_connected = False
        
        # Notify tabs about failure
        if hasattr(self, 'connection_tab'):
            self.connection_tab.on_connection_change(False)
        if hasattr(self, 'live_tab'):
            self.live_tab.on_connection_change(False)
        
        messagebox.showerror("Connection Error", f"Failed to connect to IB:\n{error_msg}")
    
    # Find the on_connection_change method in gui_main.py and replace it with this fixed version:

    def on_connection_change(self, connected):
        """Callback for connection status changes"""
        if connected:
            # ACTUALLY START THE IB CONNECTION!
            self.logger.info("Connection requested - starting IB connection process")
            self.connect_ib()  # This will do the actual IB connection
        else:
            # Handle disconnection - actually disconnect from IB
            self.disconnect_ib()
            
            # Update UI status
            self.update_connection_status("⚫ Disconnected", "black")
            self.update_status("💤 Idle")
            
            # Notify tabs about disconnection
            if hasattr(self, 'connection_tab'):
                self.connection_tab.on_connection_change(False)
            if hasattr(self, 'live_tab'):
                self.live_tab.on_connection_change(False)
            
            self.logger.info("Disconnected via UI")
    
    # ===== TRADING OPERATIONS =====
    
    def run_backtest(self):
        """Run backtest via menu"""
        self.notebook.select(self.backtest_frame)
        if hasattr(self, 'backtest_tab'):
            self.backtest_tab.run_backtest()
    
    def run_backtest_callback(self, params):
        """Callback for backtest execution with connection validation"""
        
        # TEST IB CONNECTION FIRST
        if not self.test_ib_connection_before_backtest():
            self.update_status("❌ IB Connection Test Failed")
            if hasattr(self, 'backtest_tab'):
                self.backtest_tab.progress.stop()
                self.backtest_tab.run_btn.config(state="normal", text="🚀 Run Backtest")
            return
        
        # Check if this is a visual strategy
        strategy_config = params.get("strategy_config")
        if strategy_config:
            self.logger.info(f"🎨 Running backtest with VISUAL STRATEGY: {strategy_config.get('strategy_name')}")
        else:
            self.logger.info("📊 Running backtest with DEFAULT strategy")
        
        def backtest_thread():
            try:
                self.update_status("📊 Running backtest...")
                
                # Create trading engine
                engine_config = {
                    "SYMBOLS": params.get("symbols", self.config.SYMBOLS),
                    "START_DATE": params.get("start_date"),
                    "ACCOUNT_SIZE": params.get("initial_capital", self.config.ACCOUNT_SIZE)
                }
                
                engine = create_trading_engine("backtest", engine_config)
                
                # Initialize with IB connection
                ib_app = None
                if self.ib_connection:
                    ib_app, _ = self.ib_connection
                
                engine.initialize_components(ib_app)
                
                # CRITICAL: Pass strategy_config to run_backtest
                results = engine.run_backtest(
                    symbols=params.get("symbols"),
                    start_date=params.get("start_date"),
                    end_date=params.get("end_date"),
                    strategy_config=strategy_config  # ← Pass visual strategy config
                )
                
                # Update UI on main thread
                self.root.after(0, lambda: self.on_backtest_complete(results))
                
            except Exception as e:
                self.logger.error(f"Backtest failed: {e}")
                import traceback
                self.logger.error(f"Backtest traceback: {traceback.format_exc()}")
                self.root.after(0, lambda: self.on_backtest_error(str(e)))
        
        # Start backtest in background thread
        threading.Thread(target=backtest_thread, daemon=True).start()

    def debug_connection_status(self):
        """Debug method to check IB connection status"""
        print("=== IB CONNECTION DEBUG ===")
        print(f"is_connected: {self.is_connected}")
        print(f"ib_connection exists: {self.ib_connection is not None}")
        
        if self.ib_connection:
            print(f"ib_connection type: {type(self.ib_connection)}")
            if isinstance(self.ib_connection, tuple):
                app, gateway = self.ib_connection
                print(f"app exists: {app is not None}")
                print(f"gateway exists: {gateway is not None}")
                if app:
                    print(f"app.isConnected(): {app.isConnected()}")
                    print(f"app type: {type(app)}")
            else:
                print(f"Direct connection: {self.ib_connection}")
        else:
            print("ib_connection is None!")
        print("=== END DEBUG ===")

    def test_ib_connection_before_backtest(self):
        """Test IB connection before running backtest"""
        if not self.ib_connection:
            messagebox.showerror("No IB Connection", 
                            "IB connection object is None!\n"
                            "Try disconnecting and reconnecting.")
            return False
        
        try:
            if isinstance(self.ib_connection, tuple):
                app, gateway = self.ib_connection
                if not app or not app.isConnected():
                    messagebox.showerror("IB Connection Lost", 
                                    "IB app is not connected!\n" 
                                    "Try disconnecting and reconnecting.")
                    return False
            return True
        except Exception as e:
            messagebox.showerror("IB Connection Error", 
                            f"Error checking IB connection: {e}\n"
                            "Try disconnecting and reconnecting.")
            return False
    
    def on_backtest_complete(self, results):
        """Handle backtest completion"""
        self.update_status("✅ Backtest complete")
        if hasattr(self, 'backtest_tab'):
            self.backtest_tab.display_results(results)
    
    def on_backtest_error(self, error_msg):
        """Handle backtest error"""
        self.update_status("❌ Backtest failed")
        messagebox.showerror("Backtest Error", f"Backtest failed:\n{error_msg}")
    
    def start_paper_trading(self):
        """Start paper trading via menu"""
        if not self.is_connected:
            messagebox.showwarning("Not Connected", "Please connect to IB first")
            return
        
        self.notebook.select(self.live_frame)
        if hasattr(self, 'live_tab'):
            self.live_tab.start_trading()
    
    def start_live_trading_callback(self, params):
        """Callback for live trading"""
        if not self.is_connected:
            messagebox.showwarning("Not Connected", "Please connect to IB first")
            return
        
        def trading_thread():
            try:
                self.update_status("📈 Starting live trading...")
                
                # Create trading engine
                mode = "paper" if params.get("paper_mode", True) else "live"
                self.trading_engine = create_trading_engine(mode, {
                    "SYMBOLS": params.get("symbols", self.config.SYMBOLS),
                    "ACCOUNT_SIZE": params.get("capital", self.config.ACCOUNT_SIZE)
                })
                
                # Initialize with IB connection
                ib_app, _ = self.ib_connection
                self.trading_engine.initialize_components(ib_app)
                
                # Start live trading
                self.trading_engine.run_live_trading(
                    update_interval_sec=params.get("update_interval", 300)
                )
                
                # Update UI
                self.root.after(0, self.on_live_trading_started)
                
            except Exception as e:
                self.logger.error(f"Live trading failed: {e}")
                self.root.after(0, lambda: self.on_live_trading_error(str(e)))
        
        # Start trading in background thread
        threading.Thread(target=trading_thread, daemon=True).start()
    
    def on_live_trading_started(self):
        """Handle live trading start"""
        mode = "📄 Paper Trading" if self.trading_engine.mode == TradingMode.PAPER else "💰 Live Trading"
        self.update_trading_status(mode, "green")
        
    def on_live_trading_error(self, error_msg):
        """Handle live trading error"""
        self.update_status("❌ Trading failed")
        messagebox.showerror("Trading Error", f"Live trading failed:\n{error_msg}")
    
    def stop_trading(self):
        """Stop live trading"""
        if self.trading_engine:
            try:
                self.trading_engine.stop_session()
                self.trading_engine = None
                self.update_trading_status("💤 Idle", "black")
                self.update_status("🛑 Trading stopped")
                self.logger.info("Trading stopped")
            except Exception as e:
                self.logger.error(f"Error stopping trading: {e}")
    
    # ===== UTILITY METHODS =====
    
    def get_trading_engine(self):
        """Get current trading engine for monitoring"""
        return self.trading_engine
    
    def on_strategy_change(self, strategy_code):
        """Handle strategy changes"""
        self.logger.info("Strategy updated")
        # Could validate strategy here
    
    def update_status(self, message):
        """Update main status message"""
        # Could update a main status label if we had one
        self.logger.info(f"Status: {message}")
    
    def update_connection_status(self, text, color="black"):
        """Update connection status display"""
        self.connection_status.config(text=text)
        # Note: ttk labels don't support foreground color directly
    
    def update_trading_status(self, text, color="black"):
        """Update trading status display"""
        self.trading_status.config(text=text)
    
    # ===== FILE OPERATIONS =====
    
    def new_strategy(self):
        """Create new strategy"""
        if hasattr(self, 'strategy_tab'):
            self.strategy_tab.new_strategy()
    
    def open_strategy(self):
        """Open existing strategy"""
        filename = filedialog.askopenfilename(
            title="Open Strategy",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        if filename and hasattr(self, 'strategy_tab'):
            self.strategy_tab.load_strategy(filename)
    
    def save_strategy(self):
        """Save current strategy"""
        if hasattr(self, 'strategy_tab'):
            self.strategy_tab.save_strategy()
    
    def load_config(self):
        """Load configuration file"""
        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.config = TradingConfig.load_from_file(filename)
                self.logger.info(f"Loaded config from {filename}")
                messagebox.showinfo("Success", "Configuration loaded successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load config:\n{e}")
    
    def save_config(self):
        """Save current configuration"""
        filename = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.config.save_to_file(filename)
                messagebox.showinfo("Success", "Configuration saved successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save config:\n{e}")
    
    def export_results(self):
        """Export trading results"""
        folder = filedialog.askdirectory(title="Select Export Folder")
        if folder and self.trading_engine:
            try:
                # Export from portfolio manager
                if hasattr(self.trading_engine, 'portfolio_manager'):
                    self.trading_engine.portfolio_manager.export_to_csv(folder)
                    messagebox.showinfo("Success", f"Results exported to {folder}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results:\n{e}")
    
    # ===== TOOLS =====
    
    def validate_system(self):
        """Validate system setup"""
        from main import validate_setup
        if validate_setup():
            messagebox.showinfo("Validation", "System validation passed!")
        else:
            messagebox.showerror("Validation", "System validation failed. Check logs for details.")
    
    def view_logs(self):
        """Open log viewer"""
        LogViewer(self.root)
    
    def open_output_folder(self):
        """Open output folder in file explorer"""
        output_path = os.path.abspath(self.config.OUTPUT_DIR)
        if os.path.exists(output_path):
            if sys.platform == "win32":
                os.startfile(output_path)
            elif sys.platform == "darwin":
                os.system(f"open '{output_path}'")
            else:
                os.system(f"xdg-open '{output_path}'")
        else:
            messagebox.showwarning("Not Found", f"Output directory not found: {output_path}")
    
    def open_settings(self):
        """Open settings dialog"""
        SettingsDialog(self.root, self.config)
    
    def open_help(self):
        """Open help documentation"""
        help_url = "https://github.com/your-repo/trading-system/wiki"
        webbrowser.open(help_url)
    
    def open_strategy_guide(self):
        """Open strategy development guide"""
        guide_url = "https://github.com/your-repo/trading-system/wiki/Strategy-Guide"
        webbrowser.open(guide_url)
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
ADX Breakout Trading System v2.0

A professional algorithmic trading system featuring:
• Advanced ADX breakout strategy
• Comprehensive backtesting engine
• Live paper and real money trading
• Built-in strategy editor
• Professional risk management

Created with Python and Interactive Brokers API
        """
        messagebox.showinfo("About", about_text.strip())
    
    def on_closing(self):
        """Handle application closing"""
        if self.trading_engine:
            result = messagebox.askyesno(
                "Confirm Exit", 
                "Trading is active. Stop trading and exit?"
            )
            if result:
                self.stop_trading()
            else:
                return
        
        # Disconnect from IB
        if self.ib_connection:
            self.disconnect_ib()
        
        self.logger.info("Application closing")
        self.root.destroy()
    
    def run(self):
        """Start the GUI application"""
        try:
            self.logger.info("Starting desktop GUI application")
            self.root.mainloop()
        except Exception as e:
            self.logger.error(f"GUI error: {e}")
            messagebox.showerror("Application Error", f"An error occurred:\n{e}")


class SettingsDialog:
    """Settings configuration dialog"""
    
    def __init__(self, parent, config):
        self.config = config
        self.window = tk.Toplevel(parent)
        self.window.title("Settings")
        self.window.geometry("500x400")
        self.window.transient(parent)
        self.window.grab_set()
        
        # TODO: Implement settings dialog
        ttk.Label(self.window, text="Settings dialog - TODO").pack(pady=20)
        ttk.Button(self.window, text="Close", command=self.window.destroy).pack()


def main():
    """Main entry point for desktop GUI"""
    try:
        app = TradingSystemGUI()
        app.run()
    except Exception as e:
        print(f"Failed to start GUI: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())