"""
Main Application Entry Point

Clean interface to run different modes of the trading system.
Replaces the complex argument parsing from your original code.
"""

import sys
import argparse
import json
from typing import Dict, List, Optional
import time

# Import our refactored components
from config import TradingConfig, validate_environment, get_config_for_mode
from logger import setup_logging, get_logger
from trading_engine import create_trading_engine, TradingMode
from data_manager import DataManager
from signals import SignalManager
from portfolio import PortfolioManager

# For IB connection (using your existing code)
try:
    from ibapi_appv1 import IbApp, IBGateway
except ImportError:
    print("Warning: Could not import IB components. Running in test mode.")
    IbApp = None
    IBGateway = None


def setup_ib_connection(config: TradingConfig, timeout_sec: int = 10):
    """Setup IB connection using existing code with proper cleanup"""
    if not IbApp or not IBGateway:
        print("IB components not available. Using mock data.")
        return None
    
    try:
        app = IbApp()
        gateway = IBGateway(app, config.IB_HOST, config.IB_PAPER_PORT, config.IB_CLIENT_ID)
        gateway.start(timeout_sec)
        
        print(f"✓ Connected to IB at {config.IB_HOST}:{config.IB_PAPER_PORT}")
        
        # Return both app and gateway for proper cleanup
        return app, gateway
        
    except Exception as e:
        print(f"✗ Failed to connect to IB: {e}")
        return None


def cleanup_ib_connection(ib_connection):
    """Properly cleanup IB connection"""
    if ib_connection:
        try:
            if isinstance(ib_connection, tuple):
                app, gateway = ib_connection
                if gateway:
                    gateway.stop()
                if app and app.isConnected():
                    app.disconnect()
            else:
                # Handle single app object
                if ib_connection.isConnected():
                    ib_connection.disconnect()
            
            # Give time for cleanup
            import time
            time.sleep(1)
            print("✓ IB connection closed")
            
        except Exception as e:
            print(f"Warning: Error during IB cleanup: {e}")


def run_backtest(config: TradingConfig, symbols: Optional[List[str]] = None, 
                start_date: str = None, end_date: str = None) -> bool:
    """Run historical backtest"""
    
    logger = get_logger("main")
    logger.info("Starting backtest mode")
    
    # Setup IB connection for data
    ib_connection = setup_ib_connection(config)
    ib_app = None
    
    if ib_connection:
        if isinstance(ib_connection, tuple):
            ib_app, gateway = ib_connection
        else:
            ib_app = ib_connection
    
    try:
        # Create and run trading engine
        engine = create_trading_engine("backtest", {
            "SYMBOLS": symbols or config.SYMBOLS,
            "START_DATE": start_date or config.START_DATE
        })
        
        engine.initialize_components(ib_app)
        
        print("\n" + "="*60)
        print("🔄 RUNNING BACKTEST")
        print("="*60)
        
        # Run backtest
        results = engine.run_backtest(symbols, start_date, end_date)
        
        if "error" in results:
            print(f"❌ Backtest failed: {results['error']}")
            return False
        
        # Display results
        print("\n" + "="*60)
        print("📊 BACKTEST RESULTS")
        print("="*60)
        
        summary = results["summary"]
        performance = results["performance"]
        comp_stats = results.get("comprehensive_stats", {})
        
        # Display comprehensive statistics in RealTest format
        if comp_stats:
            print(f"\n🎯 COMPREHENSIVE STATISTICS")
            print("-" * 80)
            
            # Show calculation details for verification
            years = comp_stats.get('years', 0)
            total_return_pct = comp_stats.get('total_return_pct', 0)
            print(f"   Debug: Years={years:.2f}, Total Return={total_return_pct:.2f}%, Annualized ROR={comp_stats.get('ror', 0):.2f}%")
            
            # Format the stats in a single line like RealTest
            stats_line = (
                f"Test: {comp_stats.get('test_name', 'ADX Breakout Strategy')} | "
                f"Dates: {comp_stats.get('date_range', 'N/A')} | "
                f"Periods: {comp_stats.get('periods', 0)} | "
                f"NetProfit: ${comp_stats.get('net_profit', 0):,.0f} | "
                f"Comp: {comp_stats.get('compound', True)} | "
                f"ROR: {comp_stats.get('ror', 0):.2f}% | "
                f"MaxDD: {comp_stats.get('max_dd', 0):.2f}% | "
                f"MAR: {comp_stats.get('mar', 0):.2f}"
            )
            print(stats_line)
            
            stats_line2 = (
                f"Trades: {comp_stats.get('trades', 0)} | "
                f"PctWins: {comp_stats.get('pct_wins', 0):.2f}% | "
                f"Expectancy: {comp_stats.get('expectancy', 0):.2f}% | "
                f"AvgWin: {comp_stats.get('avg_win', 0):.2f}% | "
                f"AvgLoss: {comp_stats.get('avg_loss', 0):.2f}% | "
                f"WinLen: {comp_stats.get('win_len', 0):.0f} | "
                f"LossLen: {comp_stats.get('loss_len', 0):.2f}"
            )
            print(stats_line2)
            
            stats_line3 = (
                f"ProfitFactor: {comp_stats.get('profit_factor', 0):.2f} | "
                f"Sharpe: {comp_stats.get('sharpe', 0):.2f} | "
                f"AvgExp: {comp_stats.get('avg_exp', 0):.2f}% | "
                f"MaxExp: {comp_stats.get('max_exp', 0):.2f}%"
            )
            print(stats_line3)
        
        print(f"\n📈 Portfolio Performance:")
        print(f"   Initial Equity:     ${summary['equity']['initial']:,.2f}")
        print(f"   Final Equity:       ${summary['equity']['current']:,.2f}")
        print(f"   Total Return:       ${summary['equity']['total_return']:,.2f}")
        print(f"   Total Return %:     {summary['equity']['total_return_pct']:.2f}%")
        
        print(f"\n📊 Risk Metrics:")
        print(f"   Max Drawdown:       {performance.get('max_drawdown', 0)*100:.2f}%")
        print(f"   Volatility:         {performance.get('volatility', 0)*100:.2f}%")
        print(f"   Sharpe Ratio:       {performance.get('sharpe_ratio', 0):.2f}")
        
        print(f"\n🎯 Trading Statistics:")
        trades = summary["trades"]
        print(f"   Total Trades:       {trades.get('total_trades', 0)}")
        print(f"   Winning Trades:     {trades.get('winning_trades', 0)}")
        print(f"   Win Rate:           {trades.get('win_rate', 0):.1f}%")
        print(f"   Profit Factor:      {trades.get('profit_factor', 0):.2f}")
        print(f"   Avg Holding Days:   {trades.get('avg_holding_days', 0):.1f}")
        
        print(f"\n💰 Best/Worst Trades:")
        print(f"   Best Trade:         ${trades.get('best_trade', 0):.2f}")
        print(f"   Worst Trade:        ${trades.get('worst_trade', 0):.2f}")
        
        print(f"\n📁 Results exported to: {config.OUTPUT_DIR}/backtest/")
        
        # Show top performing symbols if available
        if "trades" in results and results["trades"]:
            print(f"\n🏆 Sample Trades:")
            trades_list = results["trades"]
            # Show first 5 trades
            for i, trade in enumerate(trades_list[:5]):
                print(f"   {i+1}. {trade['symbol']}: {trade['return_pct']:.1f}% "
                      f"(${trade['realized_pnl']:.2f}) - {trade['holding_days']} days")
        
        print("\n" + "="*60)
        logger.info("Backtest completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Backtest failed with exception: {e}")
        print(f"❌ Backtest failed: {e}")
        return False
        
    finally:
        # Always cleanup IB connection
        cleanup_ib_connection(ib_connection)


def run_paper_trading(config: TradingConfig, update_interval: int = 300):
    """Run paper trading"""
    
    logger = get_logger("main")
    logger.info("Starting paper trading mode")
    
    # Setup IB connection
    ib_connection = setup_ib_connection(config)
    if not ib_connection:
        print("❌ Paper trading requires IB connection")
        return False
    
    ib_app = ib_connection[0] if isinstance(ib_connection, tuple) else ib_connection
    
    try:
        # Create trading engine
        engine = create_trading_engine("paper")
        engine.initialize_components(ib_app)
        
        print("\n" + "="*60)
        print("📈 PAPER TRADING ACTIVE")
        print("="*60)
        print(f"🔄 Update Interval: {update_interval} seconds")
        print(f"📊 Tracking Symbols: {', '.join(config.SYMBOLS[:10])}")
        if len(config.SYMBOLS) > 10:
            print(f"    ... and {len(config.SYMBOLS) - 10} more")
        print(f"💰 Initial Capital: ${config.ACCOUNT_SIZE:,.2f}")
        print(f"🎯 Max Positions: {config.MAX_POSITIONS}")
        print("\nPress Ctrl+C to stop")
        print("="*60)
        
        # Start live trading
        engine.run_live_trading(update_interval)
        
        # Monitor and display status
        while engine.state.value == "RUNNING":
            time.sleep(30)  # Status update every 30 seconds
            
            status = engine.get_engine_status()
            
            if "portfolio" in status:
                portfolio = status["portfolio"]
                print(f"\n📊 Status Update:")
                print(f"   💰 Equity: ${portfolio['equity']:,.2f} ({portfolio['return_pct']:+.2f}%)")
                print(f"   📍 Positions: {portfolio['positions']}")
                print(f"   💵 Cash: ${portfolio['cash']:,.2f}")
                
            if "current_session" in status:
                session = status["current_session"]
                print(f"   ⏱️  Runtime: {session['duration_minutes']:.0f} minutes")
                print(f"   📡 Signals: {session['total_signals']}")
                print(f"   🔄 Trades: {session['total_trades']}")
    
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping paper trading...")
        engine.stop_session()
        
        # Final status
        status = engine.get_engine_status()
        if "portfolio" in status:
            portfolio = status["portfolio"]
            print(f"\n📊 Final Status:")
            print(f"   💰 Final Equity: ${portfolio['equity']:,.2f}")
            print(f"   📈 Total Return: {portfolio['return_pct']:+.2f}%")
            print(f"   📍 Final Positions: {portfolio['positions']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Paper trading failed: {e}")
        print(f"❌ Paper trading failed: {e}")
        return False
        
    finally:
        # Always cleanup IB connection
        cleanup_ib_connection(ib_connection)
        logger.info("Paper trading session ended")


def run_daily_roll(config: TradingConfig):
    """Run daily order management (after market close)"""
    
    logger = get_logger("main")
    logger.info("Starting daily roll")
    
    # Setup IB connection
    ib_connection = setup_ib_connection(config)
    if not ib_connection:
        print("❌ Daily roll requires IB connection")
        return False
    
    print("\n" + "="*60)
    print("🔄 DAILY ORDER ROLL")
    print("="*60)
    
    try:
        # Use your existing roll function
        from ibapi_appv1 import roll_daily_brackets_after_close
        
        result = roll_daily_brackets_after_close(
            config.SYMBOLS,
            host=config.IB_HOST,
            port=config.IB_PAPER_PORT,
            client_id=config.IB_CLIENT_ID + 10,  # Different client ID
            equity=config.ACCOUNT_SIZE,
            max_positions=config.MAX_POSITIONS,
            max_exposure_pct=config.MAX_EXPOSURE_PCT
        )
        
        print("✅ Daily roll completed successfully")
        logger.info("Daily roll completed")
        return True
        
    except Exception as e:
        print(f"❌ Daily roll failed: {e}")
        logger.error(f"Daily roll failed: {e}")
        return False
        
    finally:
        # Always cleanup IB connection
        cleanup_ib_connection(ib_connection)


def run_analysis(config: TradingConfig):
    """Run analysis on existing results"""
    
    logger = get_logger("main")
    logger.info("Starting analysis mode")
    
    print("\n" + "="*60)
    print("📊 PERFORMANCE ANALYSIS")
    print("="*60)
    
    try:
        import os
        import pandas as pd
        
        # Look for existing result files
        output_dir = config.OUTPUT_DIR
        
        # Check for trade files
        trade_files = []
        if os.path.exists(output_dir):
            for filename in os.listdir(output_dir):
                if "trades" in filename and filename.endswith(".csv"):
                    trade_files.append(os.path.join(output_dir, filename))
        
        if not trade_files:
            print("❌ No trade files found. Run backtest first.")
            return False
        
        print(f"📁 Found {len(trade_files)} trade files:")
        
        for trade_file in trade_files:
            print(f"\n📈 Analyzing: {os.path.basename(trade_file)}")
            
            try:
                trades_df = pd.read_csv(trade_file)
                
                if trades_df.empty:
                    print("   ⚠️  No trades in file")
                    continue
                
                # Basic statistics
                total_trades = len(trades_df)
                winning_trades = len(trades_df[trades_df["realized_pnl"] > 0])
                win_rate = (winning_trades / total_trades) * 100
                
                total_pnl = trades_df["realized_pnl"].sum()
                best_trade = trades_df["realized_pnl"].max()
                worst_trade = trades_df["realized_pnl"].min()
                
                print(f"   📊 {total_trades} trades, {win_rate:.1f}% win rate")
                print(f"   💰 Total P&L: ${total_pnl:,.2f}")
                print(f"   🏆 Best: ${best_trade:.2f}, Worst: ${worst_trade:.2f}")
                
                # Symbol breakdown
                if "symbol" in trades_df.columns:
                    symbol_pnl = trades_df.groupby("symbol")["realized_pnl"].sum().sort_values(ascending=False)
                    print(f"   🔝 Top symbols: {', '.join(symbol_pnl.head(3).index.tolist())}")
                
                # Strategy breakdown
                if "strategy_name" in trades_df.columns:
                    strategy_pnl = trades_df.groupby("strategy_name")["realized_pnl"].sum()
                    print(f"   🎯 Strategies: {', '.join(strategy_pnl.index.tolist())}")
                    
            except Exception as e:
                print(f"   ❌ Error analyzing {trade_file}: {e}")
        
        logger.info("Analysis completed")
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        logger.error(f"Analysis failed: {e}")
        return False


def validate_setup():
    """Validate the trading system setup"""
    
    print("\n" + "="*60)
    print("🔍 SYSTEM VALIDATION")
    print("="*60)
    
    # Check configuration
    print("📋 Checking configuration...")
    if not validate_environment():
        print("❌ Configuration validation failed")
        return False
    print("✅ Configuration OK")
    
    # Check directories
    print("📁 Checking directories...")
    config = TradingConfig()
    
    import os
    required_dirs = [config.OUTPUT_DIR, config.LOG_DIR, config.STRATEGY_DIR]
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}/ exists")
        else:
            print(f"🔄 Creating {directory}/")
            os.makedirs(directory, exist_ok=True)
    
    # Test imports
    print("🔌 Checking imports...")
    try:
        import pandas as pd
        import numpy as np
        print("✅ Core libraries (pandas, numpy)")
    except ImportError as e:
        print(f"❌ Missing core libraries: {e}")
        return False
    
    try:
        from ibapi_appv1 import IbApp
        print("✅ IB API components")
    except ImportError:
        print("⚠️  IB API components not available (test mode only)")
    
    # Test component initialization
    print("🧪 Testing components...")
    try:
        from data_manager import DataManager
        from signals import SignalManager
        from portfolio import PortfolioManager
        
        config = TradingConfig()
        data_mgr = DataManager(config)
        signal_mgr = SignalManager(config)
        portfolio_mgr = PortfolioManager(config)
        
        print("✅ All components initialize successfully")
        
    except Exception as e:
        print(f"❌ Component initialization failed: {e}")
        return False
    
    print("\n✅ System validation passed!")
    return True


def main():
    """Main application entry point"""
    
    parser = argparse.ArgumentParser(
        description="ADX Breakout Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py validate                    # Validate system setup
  python main.py backtest                    # Run backtest with default settings
  python main.py backtest --symbols AAPL,MSFT --start 2023-01-01
  python main.py paper                       # Start paper trading
  python main.py paper --interval 60         # Paper trading with 1-min updates
  python main.py roll                        # Daily order management
  python main.py analyze                     # Analyze existing results
        """
    )
    
    parser.add_argument("mode", choices=["validate", "backtest", "paper", "roll", "analyze"],
                       help="Trading mode to run")
    
    # Backtest options
    parser.add_argument("--symbols", type=str,
                       help="Comma-separated list of symbols (e.g., AAPL,MSFT,GOOGL)")
    parser.add_argument("--start", type=str,
                       help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument("--end", type=str,
                       help="End date for backtest (YYYY-MM-DD)")
    
    # Paper trading options
    parser.add_argument("--interval", type=int, default=300,
                       help="Update interval for paper trading (seconds)")
    
    # Configuration options
    parser.add_argument("--config", type=str,
                       help="Path to custom configuration file")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    import logging
    log_level = getattr(logging, args.log_level)
    setup_logging(level=log_level)
    
    logger = get_logger("main")
    logger.info(f"Starting trading system in {args.mode} mode")
    
    # Load configuration
    if args.config:
        try:
            config = TradingConfig.load_from_file(args.config)
            print(f"📋 Loaded custom config from {args.config}")
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            return 1
    else:
        config = get_config_for_mode(args.mode)
    
    # Parse symbols if provided
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
        print(f"🎯 Using custom symbols: {symbols}")
    
    # Route to appropriate function
    success = False
    
    if args.mode == "validate":
        success = validate_setup()
        
    elif args.mode == "backtest":
        success = run_backtest(config, symbols, args.start, args.end)
        
    elif args.mode == "paper":
        success = run_paper_trading(config, args.interval)
        
    elif args.mode == "roll":
        success = run_daily_roll(config)
        
    elif args.mode == "analyze":
        success = run_analysis(config)
    
    # Exit with appropriate code
    if success:
        print("\n✅ Operation completed successfully")
        logger.info(f"{args.mode} mode completed successfully")
        return 0
    else:
        print("\n❌ Operation failed")
        logger.error(f"{args.mode} mode failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())


# ===== USAGE EXAMPLES =====
"""
Command Line Usage Examples:

1. Validate system setup:
   python main.py validate

2. Run backtest with default settings:
   python main.py backtest

3. Run backtest for specific symbols and date range:
   python main.py backtest --symbols AAPL,MSFT,GOOGL --start 2023-01-01 --end 2024-01-01

4. Start paper trading with 5-minute updates:
   python main.py paper --interval 300

5. Run daily order management:
   python main.py roll

6. Analyze existing results:
   python main.py analyze

7. Use custom configuration:
   python main.py backtest --config my_config.json

8. Enable debug logging:
   python main.py paper --log-level DEBUG

Programmatic Usage:

```python
from main import run_backtest, run_paper_trading
from config import TradingConfig

# Custom configuration
config = TradingConfig()
config.TRADE_PERCENT = 10.0
config.MAX_POSITIONS = 5

# Run backtest
success = run_backtest(config, symbols=["AAPL", "MSFT"])

# Or start paper trading
success = run_paper_trading(config, update_interval=60)
```

Integration with your existing code:

1. Replace your main execution with:
   python main.py backtest

2. Replace your paper trading with:
   python main.py paper

3. Replace your daily roll with:
   python main.py roll

The new system provides:
- Clean command-line interface
- Proper error handling and logging
- Modular architecture
- Easy configuration management
- Comprehensive validation
- Better status reporting
"""