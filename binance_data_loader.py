import pandas as pd
import numpy as np
import os
import glob
from typing import Dict, List, Optional
from numba import jit, prange


class BinanceDataLoader:
    """
    Data loader for Binance individual parquet files designed for cross-sectional strategies.
    
    This loader reads individual ticker parquet files from Binance API data and creates
    a unified interface compatible with existing momentum strategies.
    Symbols can become eligible dynamically: they only need to exceed the rolling
    volume threshold for `activation_min_days`, rather than relying on a lifetime
    average.
    
    Parameters:
    -----------
    data_directory : str
        Directory containing individual ticker parquet files (format: SYMBOLUSDT-<timeframe>-data.parquet)
    timeframe : str, default "1d"
        Candle interval to load (e.g., "1d", "4h"). Defaults to daily data.
    min_records : int, default 252
        Minimum number of records required for a cryptocurrency to be included
    min_volume : float, optional  
        Rolling 30-day volume threshold (USD) for activation.
    activation_min_days : int, default 30
        Minimum number of days where the rolling volume must exceed min_volume
        for a ticker to enter the universe (allows assets to activate over time).
    start_date : str, optional
        Start date for data filtering (YYYY-MM-DD format)
    end_date : str, optional
        End date for data filtering (YYYY-MM-DD format)
    funding_rate_directory : str, optional
        Directory containing funding rate parquet files. 
        Note: Funding rates are only loaded for symbols that have valid price data 
        loaded first (i.e., they must exist in the price data directory and pass 
        volume/record filters).
    """
    
    def __init__(self, 
                 data_directory: str,
                 timeframe: str = "1d",
                 min_records: int = 252,
                 min_volume: Optional[float] = None,
                 activation_min_days: int = 30,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 funding_rate_directory: Optional[str] = None):
        self.data_directory = data_directory
        self.timeframe = timeframe.lower()
        self.min_records = min_records
        self.min_volume = min_volume
        self.activation_min_days = activation_min_days
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.funding_rate_directory = funding_rate_directory
        self._timeframe_delta = self._parse_timeframe(self.timeframe)
        self._volume_lookback_bars = self._compute_volume_lookback_bars()
        self._activation_min_periods = self._compute_activation_min_periods()
        
        # Data storage (compatible with CMC loader interface)
        self._raw_data = None
        self._crypto_universe = {}  # Maps ticker_id to {'data': DataFrame}
        self._returns_matrix = None  # Precomputed returns matrix
        self._total_returns_rankings_cache = {}  # Cache for different lookback periods
        self._daily_universe_cache = {}  # Cache for daily eligible universe
        self._funding_rate_data = {}  # Funding rate data storage
        
        # Load and process data
        self._load_data()
        if self.funding_rate_directory:
            self._load_funding_rates()

    def _parse_timeframe(self, timeframe: str) -> Optional[pd.Timedelta]:
        """Convert a timeframe string like '1d' or '4h' into a Timedelta."""
        try:
            delta = pd.to_timedelta(timeframe)
        except Exception:
            print(f"Warning: Could not parse timeframe '{timeframe}', falling back to daily assumptions.")
            return None

        if delta <= pd.Timedelta(0):
            print(f"Warning: Non-positive timeframe '{timeframe}', falling back to daily assumptions.")
            return None

        return delta

    def _compute_volume_lookback_bars(self, window_days: int = 30) -> int:
        """Translate a 30-day lookback into bars for the chosen timeframe."""
        if self._timeframe_delta is None:
            return window_days

        periods = int(np.ceil(pd.Timedelta(days=window_days) / self._timeframe_delta))
        return max(periods, 1)

    def _compute_activation_min_periods(self) -> int:
        """
        Convert activation_min_days into the equivalent number of bars so that
        the threshold retains its day-based meaning across timeframes.
        """
        if self._timeframe_delta is None:
            return self.activation_min_days

        periods_per_day = pd.Timedelta(days=1) / self._timeframe_delta
        periods_per_day = max(int(np.floor(periods_per_day)), 1)
        return max(self.activation_min_days * periods_per_day, 1)

    def _find_parquet_files(self) -> List[str]:
        """Locate parquet files for the configured timeframe."""
        patterns = [
            os.path.join(self.data_directory, f"*USDT-{self.timeframe}-data.parquet"),
            os.path.join(self.data_directory, f"*USDT-{self.timeframe}.parquet")
        ]

        if self.timeframe in ("1d", "1day"):
            patterns.append(os.path.join(self.data_directory, "*USDT.parquet"))

        parquet_files = []
        for pattern in patterns:
            parquet_files.extend(glob.glob(pattern))

        return sorted(set(parquet_files))
    
    def _load_data(self):
        """Load all individual parquet files and create unified structure."""
        print(f"Loading Binance data from {self.data_directory} (timeframe={self.timeframe})...")
        
        parquet_files = self._find_parquet_files()

        if not parquet_files:
            raise ValueError(f"No USDT parquet files found in {self.data_directory} for timeframe '{self.timeframe}'")
        
        print(f"Found {len(parquet_files)} USDT trading pairs")
        
        # Track loading statistics
        loaded_count = 0
        filtered_count = 0
        volume_lookback = self._volume_lookback_bars
        print(f"Using a {volume_lookback}-bar rolling window for 30d volume checks")
        
        for file_path in parquet_files:
            try:
                # Extract symbol from filename
                filename = os.path.basename(file_path)

                suffixes = [
                    f"-{self.timeframe}-data.parquet",
                    f"-{self.timeframe}.parquet",
                    ".parquet"
                ]
                symbol = filename
                for suffix in suffixes:
                    if symbol.endswith(suffix):
                        symbol = symbol.replace(suffix, '').strip()
                        break
                
                # Skip futures contracts and weird symbols
                if any(skip in symbol for skip in ['_250926', '_251226', 'BTCDOM', 'USDCUSDT']):
                    continue
                
                # Load individual file
                df = pd.read_parquet(file_path)

                # OPTIMIZATION: Cast to float32 immediately to halve memory usage
                float_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in float_cols:
                    if col in df.columns:
                        df[col] = df[col].astype('float32')
                
                if df.empty:
                    continue
                
                # Ensure timestamp/date index
                index_names = [name for name in df.index.names if name]

                if ('timestamp' not in index_names) and ('date' not in index_names):
                    if 'timestamp' in df.columns:
                        df = df.set_index('timestamp')
                    elif 'date' in df.columns:
                        df = df.set_index('date')
                    else:
                        print(f"Warning: {symbol} has no timestamp/date index or column")
                        continue

                # Normalize index to datetime
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    print(f"Warning: {symbol} index could not be converted to datetime")
                    continue
                
                if df.index.has_duplicates:
                    duplicates = df.index[df.index.duplicated()]
                    print(f"Warning: Found duplicates in {symbol}. Dropping duplicates, keeping last. Duplicates:\n{duplicates}")
                    df = df[~df.index.duplicated(keep='last')]

                # Apply date filters
                if self.start_date is not None:
                    df = df[df.index >= self.start_date]
                if self.end_date is not None:
                    df = df[df.index <= self.end_date]
                
                # Apply minimum records filter
                if len(df) < self.min_records:
                    filtered_count += 1
                    continue
                
                # Convert volume to dollar volume so downstream users always see notional volume
                if 'close' in df.columns:
                    dollar_volume = df['volume'] * df['close']
                else:
                    dollar_volume = df['volume']

                df['volume'] = dollar_volume
                df['volume_30d'] = dollar_volume.rolling(volume_lookback, min_periods=1).mean()

                # Apply minimum volume filter after conversion (works in dollar terms)
                if self.min_volume is not None:
                    active_periods = (df['volume_30d'] >= self.min_volume).sum()
                    if active_periods < self._activation_min_periods:
                        filtered_count += 1
                        continue
                
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
        """
        Optimized matrix builder using concat (aligns indices automatically).
        """
        if not self._crypto_universe:
            self._returns_matrix = pd.DataFrame()
            return
        
        print("Building returns matrix (Memory Optimized)...")
        
        # 1. Collect Series objects (lightweight)
        series_list = {}
        for ticker, info in self._crypto_universe.items():
            # Only keep 'close' to save memory during concat
            s = info['data']['close']
            if s.index.has_duplicates:
                duplicates = s.index[s.index.duplicated()].unique()
                print(f"Warning: {ticker} has duplicate indices in build_returns_matrix. Dropping duplicates.")
                print(f"Duplicate timestamps: {duplicates}")
                s = s[~s.index.duplicated(keep='last')]
            series_list[ticker] = s
        
        # 2. Concat all at once (Fastest way to align dates)
        # This replaces the slow loop + reindex
        self._price_matrix = pd.concat(series_list, axis=1)
        
        # 3. Sort index
        self._price_matrix.sort_index(inplace=True)
        
        # 4. Compute returns (keep as float32)
        self._returns_matrix = self._price_matrix.pct_change(fill_method=None).astype('float32')
        
        print(f"Matrix shape: {self._returns_matrix.shape}")
    
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

    def get_volume_matrix(self, vol_30d: bool = False) -> pd.DataFrame:
        """
        Get the unified volume matrix (dollar volume).
        
        Parameters:
        -----------
        vol_30d : bool, default False
            If True, returns the 30-day rolling average dollar volume.
            If False, returns the raw daily dollar volume.
        """
        if not self._crypto_universe:
            return pd.DataFrame()
            
        col_name = 'volume_30d' if vol_30d else 'volume'
        
        # Get all unique dates
        if self._returns_matrix is not None:
            all_dates = self._returns_matrix.index
        else:
            all_dates = set()
            for crypto in self._crypto_universe.values():
                all_dates.update(crypto['data'].index)
            all_dates = sorted(all_dates)
            
        tickers = list(self._crypto_universe.keys())
        
        print(f"Building volume matrix ({col_name}) for {len(tickers)} tickers over {len(all_dates)} dates...")
        
        # Create volume matrix
        volume_matrix = pd.DataFrame(index=all_dates, columns=tickers)
        
        for ticker, crypto in self._crypto_universe.items():
            data = crypto['data']
            # Handle duplicate dates
            if data.index.duplicated().any():
                data = data[~data.index.duplicated(keep='last')]
                
            volume_matrix[ticker] = data[col_name].reindex(all_dates)
            
        return volume_matrix
    
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
        
        # Convert to daily dictionary - aggregate by date
        # For intraday data, a ticker is eligible if it meets threshold at ANY point
        daily_eligible = {}
        for date in all_dates:
            date_str = date.strftime('%Y-%m-%d')
            eligible_tickers = eligible_matrix.loc[date]
            tickers_list = eligible_tickers[eligible_tickers].index.tolist()
            
            # Merge with existing tickers for this date (for intraday data)
            if date_str in daily_eligible:
                existing = set(daily_eligible[date_str])
                existing.update(tickers_list)
                daily_eligible[date_str] = list(existing)
            else:
                daily_eligible[date_str] = tickers_list
        
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
        Compatible with the existing backtester interface.
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
        """Load data for a specific ticker (legacy MarketDataLoader-compatible signature)."""
        if ticker not in self._crypto_universe:
            return None
        
        data = self._crypto_universe[ticker]['data'].copy()
        
        # Apply date filters if specified
        if self.start_date is not None:
            data = data[data.index >= self.start_date]
        if self.end_date is not None:
            data = data[data.index <= self.end_date]
        
        # Ensure required columns are present for downstream strategies
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            print(f"Warning: {ticker} missing required columns")
            return None
        
        return data[required_cols]
    
    def get_data_for_ticker(self, ticker: str, start_date: Optional[pd.Timestamp] = None, 
                           end_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Get data for a specific ticker (compatible with the legacy loader interface)."""
        if ticker not in self._crypto_universe:
            return pd.DataFrame()
        
        data = self._crypto_universe[ticker]['data'].copy()
        
        # Apply date filters
        if start_date is not None:
            data = data[data.index >= start_date]
        if end_date is not None:
            data = data[data.index <= end_date]
        
        return data
    
    def _load_funding_rates(self):
        """Load funding rate data from parquet files (including interval hours)."""
        print(f"Loading funding rate data from {self.funding_rate_directory}...")
        
        pattern = os.path.join(self.funding_rate_directory, "*USDT-funding-data.parquet")
        funding_files = glob.glob(pattern)
        
        if not funding_files:
            print(f"No funding rate files found in {self.funding_rate_directory}")
            return
        
        print(f"Found {len(funding_files)} funding rate files")
        
        for file_path in funding_files:
            try:
                # Extract symbol from filename (handle spaces in filenames)
                filename = os.path.basename(file_path).strip()
                symbol = filename.replace('-funding-data.parquet', '')
                
                # Only load funding rates for symbols we have price data for
                if symbol not in self._crypto_universe:
                    continue
                
                # Load funding rate data (now includes fundingIntervalHours)
                df = pd.read_parquet(file_path)
                
                if df.empty:
                    continue
                
                # Round to nearest hour to align timestamps (e.g. 08:00:02 -> 08:00:00)
                df.index = df.index.round('h')
                
                # Filter out records with invalid funding rates only
                df_filtered = df[df['fundingRate'].notna()].copy()

                # Ensure interval hours (and rate) are numeric for downstream matrix building
                for col in ['fundingRate', 'fundingIntervalHours']:
                    if col in df_filtered.columns:
                        df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')

                # Apply date filters
                if self.start_date is not None:
                    df_filtered = df_filtered[df_filtered.index >= self.start_date]
                if self.end_date is not None:
                    df_filtered = df_filtered[df_filtered.index <= self.end_date]
                
                if not df_filtered.empty:
                    self._funding_rate_data[symbol] = df_filtered
                    
            except Exception as e:
                print(f"Warning: Failed to load funding rate for {file_path}: {e}")
                continue
        
        print(f"Loaded funding rates for {len(self._funding_rate_data)} symbols")
    
    def _build_funding_matrix(self, column_name: str) -> pd.DataFrame:
        """Helper to build a funding-related matrix for a given column."""
        if not self._funding_rate_data:
            return pd.DataFrame()

        # Keep only symbols that actually have the requested column
        valid_symbols = [s for s, df in self._funding_rate_data.items() if column_name in df.columns]
        if not valid_symbols:
            return pd.DataFrame()
        
        # Get all unique timestamps from funding rate data
        all_timestamps = set()
        for symbol in valid_symbols:
            all_timestamps.update(self._funding_rate_data[symbol].index)
        all_timestamps = sorted(all_timestamps)
        
        print(f"Building {column_name} matrix with {len(all_timestamps)} timestamps and {len(valid_symbols)} tickers")
        
        # Create funding matrix for the requested column
        matrix = pd.DataFrame(index=all_timestamps, columns=valid_symbols)
        
        for ticker in valid_symbols:
            data = self._funding_rate_data[ticker]
            # Handle duplicate timestamps by keeping last occurrence
            if data.index.duplicated().any():
                data = data[~data.index.duplicated(keep='last')]
            
            matrix[ticker] = data[column_name].reindex(all_timestamps)
        
        print(f"{column_name} matrix final shape: {matrix.shape}")
        return matrix

    def get_funding_rate_matrix(self) -> pd.DataFrame:
        """Get unified funding rate matrix."""
        return self._build_funding_matrix('fundingRate')

    def get_funding_interval_matrix(self) -> pd.DataFrame:
        """Get unified funding interval matrix (hours)."""
        return self._build_funding_matrix('fundingIntervalHours')

    def get_funding_long_form(self) -> pd.DataFrame:
        """Return funding data as a MultiIndex (symbol, timestamp) with rate and interval columns."""
        if not self._funding_rate_data:
            return pd.DataFrame(columns=['fundingRate', 'fundingIntervalHours'])

        records = []
        for symbol, df in self._funding_rate_data.items():
            if df.empty:
                continue

            # Remove duplicate timestamps to avoid ambiguous index entries
            data = df[~df.index.duplicated(keep='last')].copy()

            # Ensure both columns exist even if absent in file
            for col in ['fundingRate', 'fundingIntervalHours']:
                if col not in data.columns:
                    data[col] = np.nan

            # Keep only the relevant columns
            subset = data[['fundingRate', 'fundingIntervalHours']].copy()
            subset['symbol'] = symbol
            subset['timestamp'] = subset.index
            records.append(subset)

        if not records:
            return pd.DataFrame(columns=['fundingRate', 'fundingIntervalHours'])

        long_df = pd.concat(records, axis=0)
        long_df = long_df.set_index(['symbol', 'timestamp']).sort_index()
        return long_df[['fundingRate', 'fundingIntervalHours']]
    
    def get_funding_rate_for_date(self, date: pd.Timestamp) -> pd.Series:
        """Get funding rates for all symbols on a specific date."""
        funding_matrix = self.get_funding_rate_matrix()
        
        if funding_matrix.empty:
            return pd.Series(dtype=float)
        
        # Find the most recent funding rate data for the given date
        available_dates = funding_matrix.index[funding_matrix.index <= date]
        if len(available_dates) == 0:
            return pd.Series(dtype=float)
        
        closest_date = available_dates.max()
        return funding_matrix.loc[closest_date].dropna()
