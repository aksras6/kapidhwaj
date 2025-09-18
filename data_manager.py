"""
Data Manager for Trading System

Handles all data fetching, caching, and indicator calculations.
Replaces the scattered data logic from your original code.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
import threading
import time

from config import TradingConfig
from logger import get_logger, LoggingContext, PerformanceLogger


@dataclass
class DataQuality:
    """Data quality metrics"""
    symbol: str
    total_bars: int
    missing_bars: int
    data_start: pd.Timestamp
    data_end: pd.Timestamp
    has_gaps: bool
    quality_score: float  # 0-1, higher is better


class IndicatorCalculator:
    """Handles all indicator calculations"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = get_logger("indicators")
    
    def wilder_rma(self, values: pd.Series, n: int) -> pd.Series:
        """Wilder's RSI Moving Average (RMA)"""
        arr = values.to_numpy(dtype=float)
        out = np.full_like(arr, np.nan, dtype=float)
        
        if len(arr) == 0 or n <= 0:
            return pd.Series(out, index=values.index)
        if len(arr) < n:
            return pd.Series(out, index=values.index)
        
        # Initial SMA
        init = np.nanmean(arr[:n])
        out[n-1] = init
        
        # Wilder's smoothing
        for i in range(n, len(arr)):
            out[i] = (out[i-1] * (n-1) + arr[i]) / n
        
        return pd.Series(out, index=values.index)
    
    def calculate_adx(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """Calculate ADX indicator using Wilder's method"""
        if period is None:
            period = self.config.ADX_LENGTH
        
        h, l, c = df["high"], df["low"], df["close"]
        
        # Directional movement
        up_move = h.diff()
        down_move = -l.diff()
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # True Range
        prev_close = c.shift(1)
        tr = pd.concat([
            (h - l).abs(),
            (h - prev_close).abs(),
            (l - prev_close).abs()
        ], axis=1).max(axis=1)
        
        # Average True Range and Directional Indicators
        atr = self.wilder_rma(tr, period)
        plus_di = 100.0 * (self.wilder_rma(pd.Series(plus_dm, index=c.index), period) / atr)
        minus_di = 100.0 * (self.wilder_rma(pd.Series(minus_dm, index=c.index), period) / atr)
        
        # Directional Index
        dx = 100.0 * (plus_di.subtract(minus_di).abs() / (plus_di + minus_di))
        
        # ADX
        adx = self.wilder_rma(dx, period)
        
        self.logger.debug(f"Calculated ADX", period=period, valid_values=adx.notna().sum())
        return adx
    
    def calculate_donchian_channels(self, df: pd.DataFrame, period: int = None) -> Tuple[pd.Series, pd.Series]:
        """Calculate Donchian channel (highest high, lowest low)"""
        if period is None:
            period = self.config.CHANNEL_LENGTH
        
        hh = df["high"].rolling(period, min_periods=period).max()
        ll = df["low"].rolling(period, min_periods=period).min()
        
        self.logger.debug(f"Calculated Donchian channels", period=period, 
                         hh_valid=hh.notna().sum(), ll_valid=ll.notna().sum())
        return hh, ll
    
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all required indicators to dataframe"""
        if df.empty:
            return df
        
        result = df.copy()
        
        with LoggingContext(self.logger, f"Adding indicators to {result['symbol'].iloc[0]}"):
            # ADX
            result["ADX"] = self.calculate_adx(result)
            
            # Donchian Channels
            hh, ll = self.calculate_donchian_channels(result)
            result["HH"] = hh
            result["LL"] = ll
            
            # Additional useful indicators
            result["ATR"] = self.calculate_atr(result)
            result["Volume_SMA"] = result["volume"].rolling(20).mean()
        
        return result
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        h, l, c = df["high"], df["low"], df["close"]
        prev_close = c.shift(1)
        
        tr = pd.concat([
            (h - l).abs(),
            (h - prev_close).abs(),
            (l - prev_close).abs()
        ], axis=1).max(axis=1)
        
        return self.wilder_rma(tr, period)


class DataCache:
    """In-memory cache for market data"""
    
    def __init__(self, max_symbols: int = 100):
        self.max_symbols = max_symbols
        self._data: Dict[str, pd.DataFrame] = {}
        self._last_updated: Dict[str, datetime] = {}
        self._lock = threading.RLock()
        self.logger = get_logger("data_cache")
    
    def get(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get cached data for symbol"""
        with self._lock:
            return self._data.get(symbol)
    
    def put(self, symbol: str, data: pd.DataFrame):
        """Cache data for symbol"""
        with self._lock:
            # Remove oldest if at capacity
            if len(self._data) >= self.max_symbols and symbol not in self._data:
                oldest_symbol = min(self._last_updated.keys(), 
                                  key=lambda s: self._last_updated[s])
                self.remove(oldest_symbol)
            
            self._data[symbol] = data.copy()
            self._last_updated[symbol] = datetime.now()
            
            self.logger.debug(f"Cached data for {symbol}", bars=len(data))
    
    def remove(self, symbol: str):
        """Remove symbol from cache"""
        with self._lock:
            self._data.pop(symbol, None)
            self._last_updated.pop(symbol, None)
            self.logger.debug(f"Removed {symbol} from cache")
    
    def clear(self):
        """Clear all cached data"""
        with self._lock:
            self._data.clear()
            self._last_updated.clear()
            self.logger.info("Cache cleared")
    
    def get_cache_info(self) -> Dict[str, any]:
        """Get cache statistics"""
        with self._lock:
            total_bars = sum(len(df) for df in self._data.values())
            return {
                "symbols_cached": len(self._data),
                "total_bars": total_bars,
                "memory_usage_mb": sum(df.memory_usage(deep=True).sum() for df in self._data.values()) / 1024**2
            }


class DataManager:
    """Main data management class"""
    
    def __init__(self, config: TradingConfig, ib_app=None):
        self.config = config
        self.ib_app = ib_app
        self.logger = get_logger("data_manager")
        self.perf_logger = PerformanceLogger(self.logger)
        
        # Components
        self.cache = DataCache()
        self.indicator_calc = IndicatorCalculator(config)
        
        # Data quality tracking
        self.quality_metrics: Dict[str, DataQuality] = {}
        
        self.logger.info("DataManager initialized")
    
    def set_ib_app(self, ib_app):
        """Set IB application instance"""
        self.ib_app = ib_app
        self.logger.info("IB app connected to DataManager")
    
    def get_latest_data(self, symbol: str, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get latest data for symbol with indicators
        
        Args:
            symbol: Symbol to fetch
            force_refresh: Force refresh from IB even if cached
            
        Returns:
            DataFrame with OHLC data and indicators
        """
        with LoggingContext(self.logger, f"Getting data for {symbol}"):
            # Check cache first
            if not force_refresh:
                cached_data = self.cache.get(symbol)
                if cached_data is not None and not cached_data.empty:
                    self.logger.debug(f"Using cached data for {symbol}")
                    return cached_data
            
            # Fetch fresh data
            raw_data = self._fetch_raw_data(symbol)
            if raw_data.empty:
                self.logger.warning(f"No data available for {symbol}")
                return pd.DataFrame()
            
            # Add indicators
            data_with_indicators = self.indicator_calc.add_all_indicators(raw_data)
            
            # Cache the result
            self.cache.put(symbol, data_with_indicators)
            
            # Update quality metrics
            self._update_quality_metrics(symbol, data_with_indicators)
            
            return data_with_indicators
    
    def _fetch_raw_data(self, symbol: str) -> pd.DataFrame:
        """Fetch raw OHLC data from IB"""
        if self.ib_app is None:
            self.logger.error("IB app not connected")
            return pd.DataFrame()
        
        self.perf_logger.start_timer(f"fetch_{symbol}")
        
        try:
            # Use your existing data fetching logic
            from ibapi_appv1 import try_download_symbol
            data = try_download_symbol(self.ib_app, symbol)
            
            self.perf_logger.end_timer(f"fetch_{symbol}")
            
            if not data.empty:
                self.logger.info(f"Fetched {len(data)} bars for {symbol}")
            else:
                self.logger.warning(f"No data returned for {symbol}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {symbol}", error=str(e))
            self.perf_logger.end_timer(f"fetch_{symbol}")
            return pd.DataFrame()
    
    def _update_quality_metrics(self, symbol: str, data: pd.DataFrame):
        """Update data quality metrics for symbol"""
        if data.empty:
            return
        
        # Calculate quality metrics
        total_bars = len(data)
        missing_bars = data.isnull().sum().sum()
        
        # Check for gaps (simplified)
        if 'datetime' in data.index.names or isinstance(data.index, pd.DatetimeIndex):
            # Check for weekend gaps vs missing trading days
            has_gaps = False  # Simplified for now
        else:
            has_gaps = False
        
        # Quality score (0-1)
        completeness = 1.0 - (missing_bars / (total_bars * len(data.columns)))
        recency_score = 1.0  # Could check how recent the data is
        quality_score = (completeness + recency_score) / 2.0
        
        self.quality_metrics[symbol] = DataQuality(
            symbol=symbol,
            total_bars=total_bars,
            missing_bars=missing_bars,
            data_start=data.index[0] if len(data) > 0 else pd.NaT,
            data_end=data.index[-1] if len(data) > 0 else pd.NaT,
            has_gaps=has_gaps,
            quality_score=quality_score
        )
        
        self.logger.debug(f"Updated quality metrics for {symbol}", 
                         quality_score=round(quality_score, 3))
    
    def get_multiple_symbols(self, symbols: List[str], force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """Get data for multiple symbols efficiently"""
        with LoggingContext(self.logger, f"Fetching {len(symbols)} symbols from IB"):
            results = {}
            
            if not self.ib_app:
                self.logger.error("Cannot fetch from IB - no connection")
                return results
            
            for symbol in symbols:
                try:
                    self.logger.info(f"Fetching {symbol} from IB...")
                    data = self.get_latest_data(symbol, force_refresh)
                    
                    if not data.empty:
                        results[symbol] = data
                        start_date = data.index.min().strftime('%Y-%m-%d') 
                        end_date = data.index.max().strftime('%Y-%m-%d')
                        self.logger.info(f"✅ {symbol}: {len(data)} bars ({start_date} to {end_date})")
                    else:
                        self.logger.warning(f"❌ {symbol}: No data returned from IB")
                        
                except Exception as e:
                    self.logger.error(f"Failed to fetch {symbol}: {e}")
            
            self.logger.info(f"Fetched data for {len(results)}/{len(symbols)} symbols")
            return results
    
    def refresh_all_cached_data(self):
        """Refresh all cached data"""
        cached_symbols = list(self.cache._data.keys())
        self.logger.info(f"Refreshing {len(cached_symbols)} cached symbols")
        
        for symbol in cached_symbols:
            try:
                self.get_latest_data(symbol, force_refresh=True)
            except Exception as e:
                self.logger.error(f"Failed to refresh {symbol}", error=str(e))
    
    def get_data_summary(self) -> Dict[str, any]:
        """Get summary of all managed data"""
        cache_info = self.cache.get_cache_info()
        
        quality_summary = {}
        if self.quality_metrics:
            scores = [q.quality_score for q in self.quality_metrics.values()]
            quality_summary = {
                "avg_quality_score": np.mean(scores),
                "min_quality_score": np.min(scores),
                "symbols_with_issues": len([s for s, q in self.quality_metrics.items() 
                                          if q.quality_score < 0.9])
            }
        
        return {
            "cache_info": cache_info,
            "quality_summary": quality_summary,
            "total_symbols_tracked": len(self.quality_metrics)
        }
    
    def cleanup_old_data(self, days_old: int = 7):
        """Clean up old cached data"""
        cutoff_time = datetime.now() - pd.Timedelta(days=days_old)
        
        with self.cache._lock:
            symbols_to_remove = [
                symbol for symbol, last_updated in self.cache._last_updated.items()
                if last_updated < cutoff_time
            ]
        
        for symbol in symbols_to_remove:
            self.cache.remove(symbol)
            self.quality_metrics.pop(symbol, None)
        
        if symbols_to_remove:
            self.logger.info(f"Cleaned up {len(symbols_to_remove)} old symbols")
    
    def validate_data_quality(self, min_quality_score: float = 0.8) -> Dict[str, List[str]]:
        """Validate data quality and return issues"""
        issues = {
            "low_quality": [],
            "missing_data": [],
            "stale_data": []
        }
        
        for symbol, quality in self.quality_metrics.items():
            if quality.quality_score < min_quality_score:
                issues["low_quality"].append(symbol)
            
            if quality.missing_bars > quality.total_bars * 0.1:  # >10% missing
                issues["missing_data"].append(symbol)
            
            # Check if data is stale (older than 1 day for daily data)
            if pd.notna(quality.data_end):
                if (datetime.now() - quality.data_end.to_pydatetime()).days > 1:
                    issues["stale_data"].append(symbol)
        
        # Log issues
        for issue_type, symbols in issues.items():
            if symbols:
                self.logger.warning(f"Data quality issue: {issue_type}", 
                                   symbols=symbols[:5])  # Log first 5
        
        return issues


# ===== UTILITY FUNCTIONS =====

def bars_to_dataframe(symbol: str, bars) -> pd.DataFrame:
    """Convert IB bar data to DataFrame (from your original code)"""
    def _bar_to_dict(bar) -> dict:
        return {
            "date": getattr(bar, "date", None),
            "open": getattr(bar, "open", None),
            "high": getattr(bar, "high", None),
            "low": getattr(bar, "low", None),
            "close": getattr(bar, "close", None),
            "volume": getattr(bar, "volume", None),
            "barCount": getattr(bar, "barCount", None),
            "wap": getattr(bar, "average", getattr(bar, "wap", None)),
        }

    def _parse_ib_date(date_str: str) -> pd.Timestamp:
        if not isinstance(date_str, str):
            return pd.NaT
        if len(date_str) == 8:
            return pd.to_datetime(date_str, format="%Y%m%d", errors="coerce")
        if len(date_str) == 17:
            return pd.to_datetime(date_str, format="%Y%m%d  %H:%M:%S", errors="coerce")
        return pd.to_datetime(date_str, errors="coerce")

    rows = [_bar_to_dict(b) for b in bars]
    df = pd.DataFrame(rows)
    
    if not df.empty:
        df["datetime"] = df["date"].map(_parse_ib_date)
        df = df.drop(columns=["date"])
        df = df.sort_values("datetime").set_index("datetime")
        
        # Convert to numeric
        for col in ["open", "high", "low", "close", "volume", "barCount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        if "wap" in df.columns:
            df["wap"] = pd.to_numeric(df["wap"], errors="coerce")
        
        df.insert(0, "symbol", symbol)
    
    return df


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    # Test the data manager
    print("=== Testing Data Manager ===")
    
    from config import TradingConfig
    config = TradingConfig()
    
    # Test without IB connection
    data_manager = DataManager(config)
    
    # Test indicator calculations
    test_data = pd.DataFrame({
        'symbol': ['TEST'] * 100,
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=pd.date_range('2023-01-01', periods=100))
    
    # Test indicator calculation
    indicators = data_manager.indicator_calc
    result = indicators.add_all_indicators(test_data)
    
    print(f"Added indicators: {[col for col in result.columns if col not in test_data.columns]}")
    print(f"ADX values: {result['ADX'].tail()}")
    
    # Test caching
    data_manager.cache.put('TEST', result)
    cached = data_manager.cache.get('TEST')
    print(f"Cache test: {'PASSED' if cached is not None else 'FAILED'}")
    
    # Test quality metrics
    data_manager._update_quality_metrics('TEST', result)
    quality = data_manager.quality_metrics.get('TEST')
    print(f"Quality score: {quality.quality_score if quality else 'None'}")
    
    # Test data summary
    summary = data_manager.get_data_summary()
    print(f"Data summary: {summary}")
    
    print("\n=== Data Manager Test Complete ===")


# ===== INTEGRATION NOTES =====
"""
To integrate this with your existing code:

1. Replace your data fetching logic with:
   data_manager = DataManager(CONFIG, ib_app)
   df = data_manager.get_latest_data("AAPL")

2. Replace indicator calculations with:
   indicators = IndicatorCalculator(CONFIG)
   df_with_indicators = indicators.add_all_indicators(df)

3. Use bulk fetching for backtesting:
   all_data = data_manager.get_multiple_symbols(CONFIG.SYMBOLS)

4. Monitor data quality:
   issues = data_manager.validate_data_quality()
   if issues['low_quality']:
       logger.warning("Low quality data detected", symbols=issues['low_quality'])
"""