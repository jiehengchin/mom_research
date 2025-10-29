import pandas as pd
import numpy as np
import os
import glob
from typing import Dict, List, Optional
from portwine.loaders.base import MarketDataLoader
from numba import jit, prange

class BinanceDataLoader(MarketDataLoader):
    """
    Data loader for Binance individual parquet files designed for cross-sectional strategies.
    
    This loader reads individual ticker parquet files from Binance API data and creates
    a unified interface compatible with existing momentum strategies.
    
    Parameters:
    -----------
    data_directory : str
        Directory containing individual ticker parquet files (format: SYMBOLUSDT-1d-data.parquet)
    min_records : int, default 252
        Minimum number of records required for a cryptocurrency to be included
    min_volume : float, optional  
        Minimum average volume threshold for inclusion (Binance has no market cap data)
    start_date : str, optional
        Start date for data filtering (YYYY-MM-DD format)
    end_date : str, optional
        End date for data filtering (YYYY-MM-DD format)
    """
    
    def __init__(self, 
                 data_directory: str,
                 min_records: int = 252,
                 min_volume: Optional[float] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None):
        super().__init__()
        self.data_directory = data_directory
        self.min_records = min_records
        self.min_volume = min_volume
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        
        # Data storage (compatible with CMC loader interface)
        self._raw_data = None
        self._crypto_universe = {}  # Maps ticker_id to {'data': DataFrame}
        self._returns_matrix = None  # Precomputed returns matrix
        self._total_returns_rankings_cache = {}  # Cache for different lookback periods
        self._daily_universe_cache = {}  # Cache for daily eligible universe
        
        # Load and process data
        self._load_data()
    
    def _load_data(self):
        """Load all individual parquet files and create unified structure."""
        print(f"Loading Binance data from {self.data_directory}...")
        
        # Find all parquet files
        pattern = os.path.join(self.data_directory, "*USDT-1d-data.parquet")
        parquet_files = glob.glob(pattern)
        
        if not parquet_files:
            raise ValueError(f"No USDT parquet files found in {self.data_directory}")
        
        print(f"Found {len(parquet_files)} USDT trading pairs")
        
        # Track loading statistics
        loaded_count = 0
        filtered_count = 0
        
        for file_path in parquet_files:
            try:
                # Extract symbol from filename
                filename = os.path.basename(file_path)
                symbol = filename.replace('-1d-data.parquet', '').strip()
                
                # Skip futures contracts and weird symbols
                if any(skip in symbol for skip in ['_250926', '_251226', 'BTCDOM']):
                    continue
                
                # Load individual file
                df = pd.read_parquet(file_path)
                
                if df.empty:
                    continue
                
                # Ensure timestamp index
                if 'timestamp' not in df.index.names:
                    if 'timestamp' in df.columns:
                        df = df.set_index('timestamp')
                    else:
                        print(f"Warning: {symbol} has no timestamp index or column")
                        continue
                
                # Apply date filters
                if self.start_date is not None:
                    df = df[df.index >= self.start_date]
                if self.end_date is not None:
                    df = df[df.index <= self.end_date]
                
                # Apply minimum records filter
                if len(df) < self.min_records:
                    filtered_count += 1
                    continue
                
                # Apply minimum volume filter if specified
                if self.min_volume is not None:
                    avg_volume = df['volume'].mean()
                    if avg_volume < self.min_volume:
                        filtered_count += 1
                        continue
                
                # Add required columns for compatibility (no market cap in Binance data)
                df['volume_30d'] = df['volume'].rolling(30, min_periods=1).mean()
                
                # Store in crypto universe with CMC-compatible structure
                self._crypto_universe[symbol] = {
                    'data': df,
                    'name': symbol,
                    'symbol': symbol
                }
                
                loaded_count += 1
                
                # Debug: log BTCUSDT specifically
                if symbol == 'BTCUSDT':
                    print(f"✓ BTCUSDT loaded successfully with {len(df)} records, avg volume: {df['volume'].mean():,.0f}")
                
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                continue
        
        print(f"Loaded {loaded_count} cryptocurrencies")
        print(f"Filtered {filtered_count} cryptocurrencies (insufficient data/volume)")
        
        if loaded_count == 0:
            raise ValueError("No valid cryptocurrency data loaded")
        
        # Build returns matrix for fast cross-sectional analysis
        print("Precomputing returns matrix (FAST numpy version)...")
        self._build_returns_matrix()
        print(f"Precomputed returns matrix shape: {self._returns_matrix.shape}")
        print(f"Date range: {self._returns_matrix.index.min()} to {self._returns_matrix.index.max()}")
    
    def _build_returns_matrix(self):
        """Build unified returns matrix for all cryptocurrencies."""
        if not self._crypto_universe:
            self._returns_matrix = pd.DataFrame()
            return
        
        # Get all unique dates
        all_dates = set()
        for crypto in self._crypto_universe.values():
            all_dates.update(crypto['data'].index)
        all_dates = sorted(all_dates)
        
        tickers = list(self._crypto_universe.keys())
        
        print(f"Building returns matrix for {len(tickers)} tickers over {len(all_dates)} dates...")
        
        # Create price matrix
        price_matrix = pd.DataFrame(index=all_dates, columns=tickers)
        
        for ticker, crypto in self._crypto_universe.items():
            data = crypto['data']
            # Handle duplicate dates by keeping last occurrence
            if data.index.duplicated().any():
                data = data[~data.index.duplicated(keep='last')]
            
            price_matrix[ticker] = data['close'].reindex(all_dates)
        
        # Calculate returns matrix
        self._returns_matrix = price_matrix.pct_change(fill_method=None)
        
        # Store price matrix for later use
        self._price_matrix = price_matrix
    
    def get_universe(self) -> List[str]:
        """Get list of all available tickers."""
        return list(self._crypto_universe.keys())
    
    def get_returns_matrix(self) -> pd.DataFrame:
        """Get the precomputed returns matrix."""
        if self._returns_matrix is None:
            self._build_returns_matrix()
        return self._returns_matrix
    
    def get_price_matrix(self) -> pd.DataFrame:
        """Get the precomputed price matrix."""
        if not hasattr(self, '_price_matrix') or self._price_matrix is None:
            self._build_returns_matrix()
        return self._price_matrix
    
    def get_daily_eligible_universe(self, min_volume_30d: float = 1e6, min_market_cap_30d: float = None) -> Dict[str, List[str]]:
        """
        Get daily eligible universe based on volume thresholds.
        Note: min_market_cap_30d ignored for Binance data (no market cap available).
        Returns dictionary with date strings as keys and ticker lists as values.
        """
        cache_key = f"vol_{min_volume_30d}"
        
        if cache_key in self._daily_universe_cache:
            return self._daily_universe_cache[cache_key]
        
        print("Computing daily eligible universe (CACHED fast vectorized approach)...")
        
        # Get all dates from returns matrix
        all_dates = self.get_returns_matrix().index
        
        # Build eligibility matrix for all tickers and dates
        tickers = self.get_universe()
        print(f"Processing {len(tickers)} tickers across {len(all_dates)} dates...")
        
        # Create matrix for volume only (no market cap in Binance data)
        volume_matrix = pd.DataFrame(index=all_dates, columns=tickers)
        
        for ticker in tickers:
            data = self._crypto_universe[ticker]['data']
            # Handle duplicate dates
            if data.index.duplicated().any():
                data = data[~data.index.duplicated(keep='last')]
            
            volume_matrix[ticker] = data['volume_30d'].reindex(all_dates)
        
        # Vectorized eligibility check (volume only)
        eligible_matrix = (
            (volume_matrix >= min_volume_30d) & 
            volume_matrix.notna()
        )
        
        # Convert to daily dictionary
        daily_eligible = {}
        for date in all_dates:
            date_str = date.strftime('%Y-%m-%d')
            eligible_tickers = eligible_matrix.loc[date]
            daily_eligible[date_str] = eligible_tickers[eligible_tickers].index.tolist()
        
        # Cache results
        self._daily_universe_cache[cache_key] = daily_eligible
        
        # Print summary
        avg_eligible = np.mean([len(tickers) for tickers in daily_eligible.values()])
        print(f"Built eligibility matrix shape: {eligible_matrix.shape}")
        print(f"Completed! Average {avg_eligible:.0f} eligible tickers per day")
        
        return daily_eligible
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics about the loaded data."""
        if not self._crypto_universe:
            return {}
        
        total_tickers = len(self._crypto_universe)
        
        # Calculate date range
        all_dates = []
        for crypto in self._crypto_universe.values():
            all_dates.extend(crypto['data'].index)
        
        if all_dates:
            min_date = min(all_dates)
            max_date = max(all_dates)
            total_days = (max_date - min_date).days + 1
        else:
            min_date = max_date = total_days = 0
        
        return {
            'total_cryptocurrencies': total_tickers,
            'date_range': f"{min_date} to {max_date}",
            'total_days': total_days
        }
    
    def get_daily_data(self, date: pd.Timestamp, tickers: Optional[List[str]] = None) -> Dict[str, pd.Series]:
        """
        Get daily OHLCV data for specified date and tickers.
        Compatible with portwine backtester interface.
        """
        if tickers is None:
            tickers = self.get_universe()
        
        daily_data = {}
        
        for ticker in tickers:
            if ticker in self._crypto_universe:
                data = self._crypto_universe[ticker]['data']
                
                # Handle duplicate dates
                if data.index.duplicated().any():
                    data = data[~data.index.duplicated(keep='last')]
                
                # Find exact or closest date
                if date in data.index:
                    row = data.loc[date]
                else:
                    # Find closest previous date
                    available_dates = data.index[data.index <= date]
                    if len(available_dates) > 0:
                        closest_date = available_dates.max()
                        row = data.loc[closest_date]
                    else:
                        continue
                
                # Convert to dictionary format expected by strategies
                daily_data[ticker] = {
                    'open': row['open'],
                    'high': row['high'], 
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume'],
                    'volume_30d': row.get('volume_30d', row['volume'])
                }
        
        return daily_data
    
    def load_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load data for a specific ticker (required by portwine MarketDataLoader interface)."""
        if ticker not in self._crypto_universe:
            return None
        
        data = self._crypto_universe[ticker]['data'].copy()
        
        # Apply date filters if specified
        if self.start_date is not None:
            data = data[data.index >= self.start_date]
        if self.end_date is not None:
            data = data[data.index <= self.end_date]
        
        # Ensure required columns are present for portwine
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            print(f"Warning: {ticker} missing required columns")
            return None
        
        return data[required_cols]
    
    def get_data_for_ticker(self, ticker: str, start_date: Optional[pd.Timestamp] = None, 
                           end_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Get data for a specific ticker (compatible with portwine interface)."""
        if ticker not in self._crypto_universe:
            return pd.DataFrame()
        
        data = self._crypto_universe[ticker]['data'].copy()
        
        # Apply date filters
        if start_date is not None:
            data = data[data.index >= start_date]
        if end_date is not None:
            data = data[data.index <= end_date]
        
        return data