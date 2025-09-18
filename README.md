# ADX Breakout Trading System - Refactored

A clean, modular implementation of your ADX breakout strategy with backtesting, paper trading, and live trading capabilities.

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv trading_env
source trading_env/bin/activate  # Linux/Mac
# trading_env\Scripts\activate  # Windows

# Install dependencies
pip install pandas numpy ibapi matplotlib
```

### 2. Validate Setup

```bash
python main.py validate
```

### 3. Run Your First Backtest

```bash
# Default backtest
python main.py backtest

# Custom symbols and date range
python main.py backtest --symbols AAPL,MSFT,GOOGL --start 2023-01-01
```

## 📁 New File Structure

Your code is now organized into clean, focused modules:

```
📦 trading_system/
├── 📜 main.py              # Main application entry point
├── ⚙️  config.py           # Centralized configuration
├── 📝 logger.py            # Logging system
├── 📊 data_manager.py      # Data fetching & caching
├── 🎯 signals.py           # Signal generation
├── 💰 portfolio.py         # Position & portfolio management
├── 🔧 trading_engine.py    # Main orchestration engine
├── 📈 ibapi_appv1.py       # Your existing IB code (unchanged)
├── 📁 output/              # CSV exports
├── 📁 logs/                # Log files
└── 📁 strategies/          # Custom strategies (for future use)
```

## 🎮 Usage Examples

### Backtesting

```bash
# Basic backtest
python main.py backtest

# Specific symbols
python main.py backtest --symbols AAPL,MSFT,NVDA,GOOGL

# Date range
python main.py backtest --start 2023-01-01 --end 2024-01-01

# Custom configuration
python main.py backtest --config my_settings.json
```

### Paper Trading

```bash
# Start paper trading (5-minute updates)
python main.py paper

# More frequent updates (1-minute)
python main.py paper --interval 60

# With debug logging
python main.py paper --log-level DEBUG
```

### Daily Management

```bash
# Run after market close to update orders
python main.py roll
```

### Analysis

```bash
# Analyze existing backtest results
python main.py analyze
```

## ⚙️ Configuration

All parameters are now centralized in `config.py`. You can:

1. **Edit config.py directly** for permanent changes
2. **Create custom config files** and load with `--config`
3. **Override programmatically** when using as a library

### Key Settings

```python
# Strategy Parameters
CHANNEL_LENGTH: int = 20         # Donchian channel period
ADX_LENGTH: int = 15            # ADX calculation period
ADX_THRESHOLD: float = 20.0     # ADX threshold for trading

# Portfolio Management  
TRADE_PERCENT: float = 15.0     # % of equity per position
MAX_POSITIONS: int = 10         # Maximum concurrent positions
ACCOUNT_SIZE: float = 10_000.0  # Starting capital

# Interactive Brokers
IB_HOST: str = "127.0.0.1"     # TWS/Gateway host
IB_PAPER_PORT: int = 7497      # Paper trading port
IB_LIVE_PORT: int = 7496       # Live trading port
```

## 📊 What's Improved

### ✅ **Better Organization**
- **Single responsibility** - each module has one clear purpose
- **Clean interfaces** - easy to understand and modify
- **Modular design** - swap components without breaking others

### ✅ **Enhanced Logging**
- **Structured logging** with different levels
- **Daily log files** with timestamps
- **Trade event tracking** for analysis
- **Performance monitoring** with timing

### ✅ **Robust Data Management**
- **Intelligent caching** reduces API calls
- **Data quality validation** catches issues early
- **Automatic retries** for failed data requests
- **Multiple data source fallbacks**

### ✅ **Advanced Signal System**
- **Signal strength & confidence** for better decision making
- **Multiple strategy support** - easily add new strategies
- **Signal filtering & ranking** for portfolio optimization
- **Historical signal analysis**

### ✅ **Professional Portfolio Management**
- **Position sizing** with risk management
- **Portfolio-level limits** and exposure controls
- **Trailing stops** and alert system
- **Comprehensive performance analytics**

### ✅ **Clean Execution Engine**
- **Multiple modes** (backtest, paper, live) with shared logic
- **Session management** with error handling
- **Real-time monitoring** and status reporting
- **Graceful shutdown** and recovery

## 🔧 Development Workflow

### Adding a New Strategy

1. **Create strategy class** in `signals.py`:
```python
class MyCustomStrategy(BaseSignalGenerator):
    def generate_signals(self, data, symbol, position):
        # Your strategy logic here
        return [TradingSignal(...)]
```

2. **Register with signal manager**:
```python
signal_manager.add_generator("my_strategy", MyCustomStrategy(config))
```

3. **Test with backtest**:
```bash
python main.py backtest --symbols AAPL
```

### Customizing Configuration

1. **Create config file** `my_config.json`:
```json
{
  "TRADE_PERCENT": 10.0,
  "MAX_POSITIONS": 5,
  "SYMBOLS": ["AAPL", "MSFT", "GOOGL"],
  "ADX_THRESHOLD": 15.0
}
```

2. **Use with any mode**:
```bash
python main.py backtest --config my_config.json
```

### Using as a Library

```python
from trading_engine import create_trading_engine
from config import TradingConfig

# Create custom configuration
config = TradingConfig()
config.SYMBOLS = ["AAPL", "MSFT"]
config.TRADE_PERCENT = 10.0

# Run backtest
engine = create_trading_engine("backtest")
engine.config = config
engine.initialize_components()
results = engine.run_backtest()

print(f"Total Return: {results['summary']['equity']['total_return_pct']:.2f}%")
```

## 📈 Output & Results

### CSV Exports (in `output/` directory)

- **`trades.csv`** - Complete trade history
- **`equity.csv`** - Portfolio equity curve  
- **`positions.csv`** - Current positions (live modes)
- **`summary.csv`** - Performance summary
- **`kpis_by_symbol.csv`** - Per-symbol statistics

### Log Files (in `logs/` directory)

- **`trading_system_YYYYMMDD.log`** - Main application log
- **`data_manager_YYYYMMDD.log`** - Data operations
- **`signals_YYYYMMDD.log`** - Signal generation
- **`portfolio_manager_YYYYMMDD.log`** - Portfolio operations
- **`trade_events_YYYYMMDD.jsonl`** - Structured trade events

## 🔍 Troubleshooting

### Common Issues

1. **"IB components not available"**
   - Your original `ibapi_appv1.py` file needs to be in the same directory
   - Or run in test mode without IB connection

2. **"Configuration validation failed"**
   - Run `python main.py validate` to see specific issues
   - Check that required directories exist

3. **"No data available"**
   - Ensure TWS/IB Gateway is running
   - Check your IB paper trading permissions
   - Verify symbol names are correct

4. **Memory issues with large backtests**
   - Reduce the number of symbols
   - Shorten the date range
   - Increase the `WARMUP_BARS` to reduce data size

### Performance Tips

- **Cache data** between runs by not forcing refresh
- **Limit symbols** to 10-20 for faster backtests  
- **Use paper mode** for strategy development
- **Enable debug logging** only when troubleshooting

## 🚀 Next Steps

Now that you have a solid foundation, you can:

1. **Build the desktop GUI** (Phases 6-9 from our plan)
2. **Add more strategies** using the signal framework
3. **Implement live order management** with your broker
4. **Add more sophisticated risk management**
5. **Create custom analytics and reporting**

The modular design makes all of these extensions much easier than before!

## 📞 Migration from Original Code

Your original functions map to new components like this:

| Original Function | New Component |
|------------------|---------------|
| `backtest_portfolio()` | `TradingEngine.run_backtest()` |
| `place_paper_orders_now()` | `TradingEngine.run_live_trading()` |
| `roll_daily_brackets_after_close()` | `main.py roll` mode |
| `adx_wilder()` | `IndicatorCalculator.calculate_adx()` |
| Position tracking variables | `PortfolioManager.positions` |
| Signal logic in backtest | `ADXBreakoutSignalGenerator` |

The core logic is preserved, just much better organized!