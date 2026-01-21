"""
Lightweight backtest framework with reusable components for cross-sectional strategies.

Key pieces:
- DataBundle: shared precomputes (prices, returns, vol, ID, volume, BTC filter).
- Strategy: produces signals/factors for each date.
- WeightingModel: converts signals + risk inputs into target weights.
- BacktestEngine: runs the daily loop with turnover/TC/IC tracking.

This is designed to swap in different strategies/weighting rules without rewriting
the walk-forward logic. Default helpers mirror the momentum + (-ID) flow from the
1h notebook, using numpy arrays for speed.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-8
# Default annualization factor (override for non-hourly data, e.g., 365.25 for 1d)
PERIODS_PER_YEAR = 365.25 * 24


def compute_id_matrix(price_df: pd.DataFrame, Wd: int) -> pd.DataFrame:
    """
    Information Discreteness (-ID) helper; returns positive values for discrete uptrends.
    Matches the notebook implementation.
    """
    lagged_price = price_df.shift(1)
    p_base = lagged_price.shift(Wd)
    p_recent = lagged_price
    pret = p_recent / p_base - 1.0
    pret_sign = np.sign(pret)

    daily_rets = lagged_price.pct_change(fill_method=None)
    neg_indicator = (daily_rets < 0).astype(float)
    pos_indicator = (daily_rets > 0).astype(float)
    zero_indicator = (daily_rets == 0).astype(float)

    neg_count = neg_indicator.rolling(window=Wd, min_periods=1).sum()
    pos_count = pos_indicator.rolling(window=Wd, min_periods=1).sum()
    zero_count = zero_indicator.rolling(window=Wd, min_periods=1).sum()

    non_zero_count = Wd - zero_count
    pct_neg = neg_count / non_zero_count.replace(0, np.nan)
    pct_pos = pos_count / non_zero_count.replace(0, np.nan)

    pct_neg = pct_neg.fillna(0)
    pct_pos = pct_pos.fillna(0)

    id_matrix = -pret_sign * (pct_neg - pct_pos)
    id_matrix = id_matrix.where(~pret.isna(), np.nan)
    return id_matrix


class DataBundle:
    """
    Container for price-related arrays and reusable precomputes.
    """

    def __init__(
        self,
        price_df: pd.DataFrame,
        rolling_volume_df: Optional[pd.DataFrame] = None,
        btc_ret: Optional[pd.Series] = None,
        funding_df: Optional[pd.DataFrame] = None,
        min_hist_days: int = 30,
    ):
        self.price_df = price_df
        self.rolling_volume_df = rolling_volume_df
        self.btc_ret = btc_ret
        self.funding_df = funding_df
        self.min_hist_days = min_hist_days

        self.price_np = price_df.to_numpy(dtype=float)
        self.dates = price_df.index.to_numpy()
        self.tickers = list(price_df.columns)

        self.forward_returns = (self.price_np[1:] / self.price_np[:-1]) - 1.0
        self.price_pair_mask = np.isfinite(self.price_np[:-1]) & np.isfinite(self.price_np[1:])
        self.valid_roll_counts = price_df.notna().rolling(window=min_hist_days, min_periods=1).sum().to_numpy()

        self.simple_rets: Dict[int, np.ndarray] = {}
        self.id_matrix: Dict[int, np.ndarray] = {}
        self.vol_matrix: Dict[int, np.ndarray] = {}

        self.returns_matrix = price_df.pct_change(fill_method=None)
        self.rolling_volume_np = (
            rolling_volume_df.to_numpy(dtype=float) if rolling_volume_df is not None else None
        )
        self.btc_ret_np = btc_ret.to_numpy(dtype=float) if btc_ret is not None else None
        if funding_df is not None:
            funding_aligned = funding_df.reindex(index=price_df.index, columns=price_df.columns)
            self.funding_np = funding_aligned.to_numpy(dtype=float)
        else:
            self.funding_np = None

    def ensure_simple_returns(self, windows: List[int]) -> None:
        for w in windows:
            if w not in self.simple_rets:
                self.simple_rets[w] = (
                    self.price_df.shift(1).pct_change(w, fill_method=None).to_numpy(dtype=float)
                )

    def ensure_id_matrix(self, windows: List[int]) -> None:
        for w in windows:
            if w not in self.id_matrix:
                self.id_matrix[w] = compute_id_matrix(self.price_df, w).to_numpy(dtype=float)

    def ensure_vol_matrix(self, windows: List[int]) -> None:
        for w in windows:
            if w not in self.vol_matrix:
                self.vol_matrix[w] = (
                    self.returns_matrix.rolling(window=w).std().to_numpy(dtype=float)
                )


class Strategy:
    """
    Base strategy: override prepare() and signals().
    signals() should return a dict with at least 'alpha': np.ndarray (len = n_assets).
    Additional keys (e.g., 'id') can be used by weighting models.
    """

    def prepare(self, bundle: DataBundle) -> None:
        pass

    def signals(self, idx: int, bundle: DataBundle) -> Dict[str, np.ndarray]:
        raise NotImplementedError


@dataclass
class MomentumIDParams:
    simple_window: int
    id_window: int
    vol_window: int
    volume_pct: float = 0.2
    momentum_pct: float = 0.1
    momentum_mode: str = "absolute"  # "absolute" (default) or "relative" (demeaned cross-section)
    weighting_method: str = "vol"
    long_id_threshold: float = 0.0
    short_id_threshold: float = 0.0
    max_positions_per_side: int = 10
    max_position_cap: float = 0.3
    min_weight: float = 0.05
    tc_bps: float = 5.0
    use_btc_filter: bool = True


class MomentumIDStrategy(Strategy):
    """
    Momentum + (-ID) signal provider; weights handled separately.
    """

    def __init__(self, params: MomentumIDParams):
        self.params = params

    def prepare(self, bundle: DataBundle) -> None:
        bundle.ensure_simple_returns([self.params.simple_window])
        bundle.ensure_id_matrix([self.params.id_window])
        bundle.ensure_vol_matrix([self.params.vol_window])

    def signals(self, idx: int, bundle: DataBundle) -> Dict[str, np.ndarray]:
        alpha_row = bundle.simple_rets[self.params.simple_window][idx]
        if (self.params.momentum_mode or "absolute").lower() == "relative":
            finite = np.isfinite(alpha_row)
            if finite.any():
                mean_val = alpha_row[finite].mean()
                alpha = alpha_row - mean_val
            else:
                alpha = alpha_row.copy()
        else:
            alpha = alpha_row
        return {
            "alpha": alpha,
            "id": bundle.id_matrix[self.params.id_window][idx],
        }


class WeightingModel:
    """
    Base weighting model: returns weight vector (len = n_assets) given signals and masks.
    """

    def weights(
        self,
        idx: int,
        signals: Dict[str, np.ndarray],
        bundle: DataBundle,
        universe_mask: np.ndarray,
        params: Any,
    ) -> np.ndarray:
        raise NotImplementedError


class LongShortVolWeighting(WeightingModel):
    """
    Long/short top-k weighting with selectable weighting method and ID filters.
    Supported weighting_method values:
    - "vol": inverse-vol scaling (default, previous behavior)
    - "equal": equal weights within each side
    - "alpha": weights proportional to alpha magnitude within each side
    - "alpha_over_vol": weights proportional to alpha divided by rolling vol
    """

    def weights(
        self,
        idx: int,
        signals: Dict[str, np.ndarray],
        bundle: DataBundle,
        universe_mask: np.ndarray,
        params: MomentumIDParams,
    ) -> np.ndarray:
        method = (getattr(params, "weighting_method", "vol") or "vol").lower()
        alpha = signals["alpha"]
        id_vals = signals.get("id")
        vol_row = bundle.vol_matrix[params.vol_window][idx]

        mask = universe_mask & np.isfinite(alpha) & np.isfinite(vol_row)
        if id_vals is not None:
            mask &= np.isfinite(id_vals)

        available = np.nonzero(mask)[0]
        if available.size == 0:
            return np.zeros_like(alpha)

        alpha_av = alpha[available]
        id_av = id_vals[available] if id_vals is not None else None

        k = max(1, int(alpha_av.size * params.momentum_pct))
        order_long = np.argsort(-alpha_av)
        order_short = np.argsort(alpha_av)

        long_idx = order_long
        short_idx = order_short

        if id_av is not None:
            long_idx = long_idx[id_av[long_idx] >= params.long_id_threshold]
            short_idx = short_idx[id_av[short_idx] >= params.short_id_threshold]

        long_idx = long_idx[:k]
        short_idx = short_idx[:k]

        if long_idx.size > params.max_positions_per_side:
            long_idx = long_idx[: params.max_positions_per_side]
        if short_idx.size > params.max_positions_per_side:
            short_idx = short_idx[: params.max_positions_per_side]

        if long_idx.size == 0 or short_idx.size == 0:
            return np.zeros_like(alpha)

        long_assets = available[long_idx]
        short_assets = available[short_idx]

        def side_base_weights(assets: np.ndarray, side: str) -> np.ndarray:
            if assets.size == 0:
                return np.array([], dtype=float)
            if method == "vol":
                vols = np.where(vol_row[assets] <= 0, EPS, vol_row[assets])
                return 1.0 / vols
            if method == "equal":
                return np.ones(assets.size, dtype=float)
            if method == "alpha":
                if side == "long":
                    return np.maximum(alpha[assets], 0)
                return np.maximum(-alpha[assets], 0)
            if method == "alpha_over_vol":
                vols = np.where(vol_row[assets] <= 0, EPS, vol_row[assets])
                if side == "long":
                    return np.maximum(alpha[assets], 0) / vols
                return np.maximum(-alpha[assets], 0) / vols
            raise ValueError(f"Unsupported weighting_method: {method}")

        def finalize_weights(base_w: np.ndarray, assets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            base_w = np.nan_to_num(base_w, nan=0.0, posinf=0.0, neginf=0.0)
            if base_w.size == 0 or base_w.sum() <= 0:
                return np.array([], dtype=float), np.array([], dtype=int)
            base_w /= base_w.sum()
            base_w = np.minimum(base_w, params.max_position_cap)
            keep = base_w >= params.min_weight
            base_w = base_w[keep]
            assets = assets[keep]
            if base_w.size == 0 or base_w.sum() <= 0:
                return np.array([], dtype=float), np.array([], dtype=int)
            base_w /= base_w.sum()
            return base_w, assets

        lw_raw = side_base_weights(long_assets, "long")
        sw_raw = side_base_weights(short_assets, "short")
        lw, long_assets = finalize_weights(lw_raw, long_assets)
        sw, short_assets = finalize_weights(sw_raw, short_assets)

        if lw.size == 0 or sw.size == 0:
            return np.zeros_like(alpha)

        lw *= 0.5
        sw *= 0.5

        weights = np.zeros_like(alpha, dtype=float)
        weights[long_assets] = lw
        weights[short_assets] = -sw
        return weights


class BacktestEngine:
    """
    Runs the daily loop given a strategy and weighting model.
    """

    def __init__(
        self,
        bundle: DataBundle,
        strategy: Strategy,
        weighting: WeightingModel,
        params: Any,
        volume_pct: float = 0.2,
        min_hist_days: Optional[int] = None,
        tc_bps: float = 5.0,
    ):
        self.bundle = bundle
        self.strategy = strategy
        self.weighting = weighting
        self.params = params
        self.volume_pct = volume_pct
        self.min_hist_days = min_hist_days or bundle.min_hist_days
        self.tc_bps = tc_bps

        self.strategy.prepare(bundle)
        bundle.ensure_vol_matrix([params.vol_window]) if hasattr(params, "vol_window") else None

    def _universe_mask(self, idx: int, vol_row: np.ndarray) -> np.ndarray:
        b = self.bundle
        base = (
            (b.valid_roll_counts[idx] >= self.min_hist_days)
            & b.price_pair_mask[idx]
            & np.isfinite(vol_row)
        )
        if b.rolling_volume_np is None:
            return base

        vol_row_vol = b.rolling_volume_np[idx]
        valid_vol_idx = np.where(np.isfinite(vol_row_vol))[0]
        if valid_vol_idx.size == 0:
            return np.zeros_like(base, dtype=bool)

        n_universe = max(1, int(valid_vol_idx.size * self.volume_pct))
        top_candidates = np.argpartition(-vol_row_vol[valid_vol_idx], max(n_universe - 1, 0))[:n_universe]
        top_idx = valid_vol_idx[top_candidates]

        mask = np.zeros_like(base, dtype=bool)
        mask[top_idx] = True
        mask &= base
        return mask

    def run(
        self,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        collect_ic: bool = False,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, Optional[pd.Series], pd.DataFrame]:
        b = self.bundle
        n_dates, n_assets = b.price_np.shape
        if end_idx is None or end_idx >= n_dates:
            end_idx = n_dates - 1
        max_loop_end = min(end_idx, n_dates - 1)

        equity_path: List[float] = []
        return_path: List[float] = []
        turnover_path: List[float] = []
        date_path: List[np.datetime64] = []
        ic_vals: List[float] = []
        ic_dates: List[np.datetime64] = []
        position_records: List[Dict[str, Any]] = []

        prev_w = np.zeros(n_assets, dtype=float)
        seeded = False
        equity = 1.0

        def record_ic(date, value=np.nan):
            if collect_ic:
                ic_dates.append(date)
                ic_vals.append(value)

        for i in range(start_idx, max_loop_end):
            next_i = i + 1
            vol_row = b.vol_matrix[self.params.vol_window][i]
            uni_mask = self._universe_mask(i, vol_row)

            sigs = self.strategy.signals(i, b)
            alpha = sigs.get("alpha")

            if alpha is None or np.isnan(alpha).all():
                if seeded:
                    equity_path.append(equity)
                    return_path.append(0.0)
                    turnover_path.append(0.0)
                    date_path.append(b.dates[next_i])
                    record_ic(b.dates[next_i])
                continue

            if not seeded:
                equity_path.append(equity)
                return_path.append(0.0)
                turnover_path.append(0.0)
                date_path.append(b.dates[next_i])
                record_ic(b.dates[next_i])
                seeded = True
                continue

            if (
                getattr(self.params, "use_btc_filter", False)
                and b.btc_ret_np is not None
                and i < len(b.btc_ret_np)
                and np.isfinite(b.btc_ret_np[i])
                and b.btc_ret_np[i] < 0
            ):
                prev_w = np.zeros_like(prev_w)
                equity_path.append(equity)
                return_path.append(0.0)
                turnover_path.append(0.0)
                date_path.append(b.dates[next_i])
                record_ic(b.dates[next_i])
                continue

            if (i - start_idx + 1) < self.min_hist_days:
                equity_path.append(equity)
                return_path.append(0.0)
                turnover_path.append(0.0)
                date_path.append(b.dates[next_i])
                record_ic(b.dates[next_i])
                continue

            weights = self.weighting.weights(i, sigs, b, uni_mask, self.params)

            if not np.any(weights):
                equity_path.append(equity)
                return_path.append(0.0)
                turnover_path.append(0.0)
                date_path.append(b.dates[next_i])
                record_ic(b.dates[next_i])
                prev_w = weights
                continue

            turnover = np.abs(weights - prev_w).sum()
            fwd = b.forward_returns[i]
            funding_cost = 0.0
            if getattr(b, "funding_np", None) is not None and next_i < b.funding_np.shape[0]:
                funding_row = b.funding_np[next_i]
                # Funding: longs pay when rate > 0, shorts pay when rate < 0 -> -weight * rate
                funding_cost = -np.nansum(weights * funding_row)
            tc_cost = turnover * (self.params.tc_bps / 10000.0 if hasattr(self.params, "tc_bps") else self.tc_bps / 10000.0)
            daily_ret = np.nansum(weights * fwd) + funding_cost - tc_cost
            equity *= (1.0 + daily_ret)

            equity_path.append(equity)
            return_path.append(daily_ret)
            turnover_path.append(turnover)
            date_path.append(b.dates[next_i])

            if collect_ic:
                traded_idx = np.nonzero(weights)[0]
                if traded_idx.size > 1:
                    ic_val = pd.Series(alpha[traded_idx]).rank().corr(
                        pd.Series(fwd[traded_idx]).rank(), method="pearson"
                    )
                else:
                    ic_val = np.nan
                record_ic(b.dates[next_i], ic_val)

            if weights.any():
                long_idx = np.where(weights > 0)[0]
                short_idx = np.where(weights < 0)[0]
                position_records.append(
                    {
                        "date": b.dates[i],
                        "long_tickers": "|".join([b.tickers[j] for j in long_idx]),
                        "short_tickers": "|".join([b.tickers[j] for j in short_idx]),
                        "long_allocations": "|".join(
                            [f"{b.tickers[j]}:{weights[j]:.6f}" for j in long_idx]
                        ),
                        "short_allocations": "|".join(
                            [f"{b.tickers[j]}:{weights[j]:.6f}" for j in short_idx]
                        ),
                        "long_positions": int(long_idx.size),
                        "short_positions": int(short_idx.size),
                        "total_long_exposure": float(np.maximum(weights, 0).sum()),
                        "total_short_exposure": float(np.maximum(-weights, 0).sum()),
                        "turnover": float(turnover),
                        "daily_return": float(daily_ret),
                    }
                )

            prev_w = weights

        eq_s = pd.Series(equity_path, index=pd.to_datetime(date_path), name="equity") if equity_path else pd.Series(dtype=float)
        ret_s = pd.Series(return_path, index=pd.to_datetime(date_path), name="return") if return_path else pd.Series(dtype=float)
        turn_s = pd.Series(turnover_path, index=pd.to_datetime(date_path), name="turnover") if turnover_path else pd.Series(dtype=float)
        ic_s = pd.Series(ic_vals, index=pd.to_datetime(ic_dates), name="ic") if collect_ic and ic_vals else pd.Series(dtype=float)
        pos_df = pd.DataFrame(position_records) if position_records else pd.DataFrame()
        # ensure timezone-naive datetime index for consistency with notebooks
        for s in (eq_s, ret_s, turn_s, ic_s):
            if hasattr(s, "index") and hasattr(s.index, "tz") and s.index.tz is not None:
                s.index = s.index.tz_localize(None)
        if not pos_df.empty and 'date' in pos_df.columns and hasattr(pos_df['date'], 'dt'):
            if pos_df['date'].dt.tz is not None:
                pos_df['date'] = pos_df['date'].dt.tz_localize(None)
        return eq_s, ret_s, turn_s, (ic_s if collect_ic else None), pos_df


# --- Scoring helpers (mirrors notebook logic) ---


def compute_sharpe(ret_series: pd.Series, periods_per_year: float = PERIODS_PER_YEAR) -> float:
    if ret_series.empty or ret_series.size < 2:
        return float("nan")
    std = ret_series.std(ddof=1)
    if std < 1e-10 or not np.isfinite(std):
        return float("nan")
    return float((ret_series.mean() / std) * np.sqrt(periods_per_year))


def compute_sortino_ratio(ret_series: pd.Series, risk_free_rate: float = 0.0, periods_per_year: float = PERIODS_PER_YEAR) -> float:
    if ret_series.empty or ret_series.size < 2:
        return float("nan")
    daily_rf = risk_free_rate / periods_per_year
    excess_returns = ret_series - daily_rf
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return 100.0 if ret_series.mean() > daily_rf else float("nan")
    downside_std = downside_returns.std(ddof=1)
    if downside_std < 1e-10 or not np.isfinite(downside_std):
        return float("nan")
    return float((ret_series.mean() / downside_std) * np.sqrt(periods_per_year))


def compute_calmar_ratio(ret_series: pd.Series, equity_series: Optional[pd.Series] = None, min_periods: int = 30, periods_per_year: float = PERIODS_PER_YEAR) -> float:
    if ret_series.empty or ret_series.size < min_periods:
        return float("nan")
    if equity_series is None:
        equity_series = (1 + ret_series).cumprod()
    running_max = equity_series.expanding().max()
    drawdown = (equity_series - running_max) / running_max
    max_drawdown = abs(drawdown.min())
    annualized_return = ret_series.mean() * periods_per_year
    if max_drawdown == 0:
        return 100.0 if annualized_return > 0 else float("nan")
    if not np.isfinite(max_drawdown) or not np.isfinite(annualized_return):
        return float("nan")
    return float(annualized_return / max_drawdown)


def compute_composite_score(ret_series: pd.Series, equity_series: Optional[pd.Series] = None, w_sortino: float = 0.4, w_sharpe: float = 0.3, w_calmar: float = 0.3, periods_per_year: float = PERIODS_PER_YEAR) -> float:
    if ret_series.empty or ret_series.size < 2:
        return float("nan")
    sortino = compute_sortino_ratio(ret_series, periods_per_year=periods_per_year)
    sharpe = compute_sharpe(ret_series, periods_per_year=periods_per_year)
    calmar = compute_calmar_ratio(ret_series, equity_series, periods_per_year=periods_per_year)
    if np.isnan(sortino) or np.isnan(sharpe) or np.isnan(calmar):
        return float("nan")
    total_weight = w_sortino + w_sharpe + w_calmar
    w_sortino /= total_weight
    w_sharpe /= total_weight
    w_calmar /= total_weight
    return float(w_sortino * sortino + w_sharpe * sharpe + w_calmar * calmar)


def select_score(ret_series: pd.Series, equity_series: Optional[pd.Series] = None, mode: str = "composite", periods_per_year: float = PERIODS_PER_YEAR) -> float:
    mode = (mode or "composite").lower()
    sharpe = compute_sharpe(ret_series, periods_per_year=periods_per_year)
    sortino = compute_sortino_ratio(ret_series, periods_per_year=periods_per_year)
    calmar = compute_calmar_ratio(ret_series, equity_series, periods_per_year=periods_per_year)
    if mode == "sharpe":
        return sharpe
    if mode == "sortino":
        return sortino
    if mode == "calmar":
        return calmar
    if mode == "sharpe_sortino":
        if np.isnan(sharpe) or np.isnan(sortino):
            return float("nan")
        return 0.5 * sharpe + 0.5 * sortino
    if mode == "sharpe_calmar":
        if np.isnan(sharpe) or np.isnan(calmar):
            return float("nan")
        return 0.5 * sharpe + 0.5 * calmar
    if mode == "composite":
        return compute_composite_score(
            ret_series,
            equity_series,
            periods_per_year=periods_per_year,
        )
    raise ValueError(f"Unsupported score_mode: {mode}")


# --- Walk-forward runner ---


class WalkForwardRunner:
    """
    Generic walk-forward driver. Accepts a strategy factory, weighting model, and parameter grid.

    All spans are in bars (e.g., hours for 1h data).
    """

    def __init__(
        self,
        bundle: DataBundle,
        strategy_factory,
        weighting_model: WeightingModel,
        params_grid: List[Any],
        train_span: int,
        test_span: int,
        step_span: int,
        mode: str = "rolling",
        score_mode: str = "composite",
        n_jobs: int = 1,
        periods_per_year: float = PERIODS_PER_YEAR,
    ):
        self.bundle = bundle
        self.strategy_factory = strategy_factory
        self.weighting_model = weighting_model
        self.params_grid = params_grid
        self.train_span = train_span
        self.test_span = test_span
        self.step_span = step_span
        self.mode = mode
        self.score_mode = score_mode
        self.n_jobs = n_jobs
        self.periods_per_year = periods_per_year or PERIODS_PER_YEAR

    def _eval_param(self, params, train_start: int, train_end: int) -> Tuple[Any, float, float]:
        strategy = self.strategy_factory(params)
        engine = BacktestEngine(
            bundle=self.bundle,
            strategy=strategy,
            weighting=self.weighting_model,
            params=params,
            volume_pct=getattr(params, "volume_pct", 0.2),
            tc_bps=getattr(params, "tc_bps", 5.0),
        )
        _, ret, _, _, _ = engine.run(start_idx=train_start, end_idx=train_end, collect_ic=False)
        score = select_score(ret, None, self.score_mode, periods_per_year=self.periods_per_year)
        sharpe = compute_sharpe(ret, periods_per_year=self.periods_per_year)
        return params, score, sharpe

    def run(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
        all_dates = self.bundle.dates
        total = len(all_dates)
        if self.train_span >= total:
            raise ValueError(f"Not enough data: need at least {self.train_span + 1} bars, have {total}")

        wf_results: List[Dict[str, Any]] = []
        oos_returns: List[float] = []
        oos_dates: List[pd.Timestamp] = []
        all_positions: List[pd.DataFrame] = []

        iteration = 0
        current_end = self.train_span

        while current_end < total:
            iteration += 1
            if self.mode == "rolling":
                train_start = current_end - self.train_span
            else:
                train_start = 0
            # Intentionally include the first OOS day’s return in training (1-day look-ahead)
            train_end = current_end  # inclusive
            test_start = current_end
            test_end = min(current_end + self.test_span, total) - 1  # inclusive
            train_start_ts = pd.to_datetime(all_dates[train_start])
            train_end_ts = pd.to_datetime(all_dates[train_end])
            test_start_ts = pd.to_datetime(all_dates[test_start])
            test_end_ts = pd.to_datetime(all_dates[test_end])

            if test_end <= test_start:
                break

            # Evaluate params (parallel if requested)
            if self.n_jobs != 1:
                try:
                    from joblib import Parallel, delayed
                except ImportError:
                    self.n_jobs = 1
                    Parallel = None

            if self.n_jobs == 1:
                evals = [self._eval_param(p, train_start, train_end) for p in self.params_grid]
            else:
                evals = Parallel(n_jobs=self.n_jobs, backend="threading")(
                    delayed(self._eval_param)(p, train_start, train_end) for p in self.params_grid
                )

            best = None
            for params, score, sharpe in evals:
                if np.isnan(score):
                    continue
                if (best is None) or (score > best["score"]):
                    best = {"params": params, "score": score, "sharpe": sharpe}

            if best is None:
                current_end += self.step_span
                continue

            # Run OOS with best params (use full history up to test_end for warmup)
            strategy = self.strategy_factory(best["params"])
            engine = BacktestEngine(
                bundle=self.bundle,
                strategy=strategy,
                weighting=self.weighting_model,
                params=best["params"],
                volume_pct=getattr(best["params"], "volume_pct", 0.2),
                tc_bps=getattr(best["params"], "tc_bps", 5.0),
            )
            _, full_ret, _, _, pos_df = engine.run(start_idx=0, end_idx=test_end, collect_ic=False)
            test_dates = all_dates[test_start : test_end + 1]
            oos_ret = full_ret.reindex(pd.to_datetime(test_dates)).dropna()
            oos_positions = pd.DataFrame()

            if not oos_ret.empty:
                oos_returns.extend(oos_ret.values.tolist())
                oos_dates.extend(oos_ret.index.tolist())
                oos_sharpe = compute_sharpe(oos_ret, periods_per_year=self.periods_per_year)
                oos_score = select_score(oos_ret, None, self.score_mode, periods_per_year=self.periods_per_year)
                if pos_df is not None and not pos_df.empty and "date" in pos_df.columns:
                    pos_df = pos_df.copy()
                    pos_df["date"] = pd.to_datetime(pos_df["date"])
                    oos_positions = pos_df[
                        (pos_df["date"] >= test_start_ts) & (pos_df["date"] <= test_end_ts)
                    ]
                    oos_positions = oos_positions[oos_positions["date"].isin(oos_ret.index)]
                    if not oos_positions.empty:
                        oos_positions = oos_positions.copy()
                        oos_positions["iteration"] = iteration
                        oos_positions["train_start"] = train_start_ts
                        oos_positions["train_end"] = train_end_ts
                        oos_positions["test_start"] = test_start_ts
                        oos_positions["test_end"] = test_end_ts
                        all_positions.append(oos_positions)
            else:
                oos_sharpe = float("nan")
                oos_score = float("nan")

            wf_results.append(
                {
                    "iteration": iteration,
                    "train_start": train_start_ts,
                    "train_end": train_end_ts,
                    "test_start": test_start_ts,
                    "test_end": test_end_ts,
                    "train_bars": train_end - train_start + 1,
                    "test_bars": test_end - test_start + 1,
                    "best_params": best["params"],
                    "is_score": best["score"],
                    "is_sharpe": best["sharpe"],
                    "oos_score": oos_score,
                    "oos_sharpe": oos_sharpe,
                }
            )

            current_end += self.step_span

        wf_df = pd.DataFrame(wf_results)
        combined_oos_returns = pd.Series(oos_returns, index=pd.to_datetime(oos_dates), name="OOS_Returns")
        combined_oos_returns = combined_oos_returns[~combined_oos_returns.index.duplicated(keep="first")].sort_index()
        combined_equity = (1 + combined_oos_returns).cumprod()
        positions_df = pd.concat(all_positions, ignore_index=True) if all_positions else pd.DataFrame()
        return wf_df, combined_oos_returns, combined_equity, positions_df

    def report(
        self,
        wf_df: pd.DataFrame,
        oos_returns: pd.Series,
        oos_equity: pd.Series,
        plot: bool = True,
        fig_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate walk-forward results, rebuild OOS equity, recompute turnover per
        iteration, and optionally render the key plots that existed in the fast
        notebook (equity/drawdown, parameter selection, IS vs OOS, BTC benchmark,
        turnover).

        Returns a dictionary with aggregated metrics and recomputed series.
        """
        per_year = self.periods_per_year
        fig_path = Path(fig_dir) if fig_dir else None
        if fig_path:
            fig_path.mkdir(parents=True, exist_ok=True)

        out: Dict[str, Any] = {}

        # Aggregate OOS performance
        if oos_returns is not None and not oos_returns.empty:
            combined_equity = (1 + oos_returns).cumprod()
            running_max = combined_equity.cummax()
            drawdown = (combined_equity - running_max) / running_max
            max_dd = drawdown.min()
            agg_sharpe = compute_sharpe(oos_returns, periods_per_year=per_year)
            agg_sortino = compute_sortino_ratio(oos_returns, periods_per_year=per_year)
            agg_calmar = compute_calmar_ratio(oos_returns, combined_equity, periods_per_year=per_year)
            agg_comp = compute_composite_score(oos_returns, combined_equity, periods_per_year=per_year)
            agg_score = select_score(oos_returns, combined_equity, mode=self.score_mode, periods_per_year=per_year)
            agg_total_ret = combined_equity.iloc[-1] - 1
            agg_cagr = (combined_equity.iloc[-1] ** (per_year / len(combined_equity)) - 1) if len(combined_equity) else float("nan")
            
            traded_rets = oos_returns[oos_returns != 0]
            agg_win_rate = (traded_rets > 0).mean() if not traded_rets.empty else float("nan")

            print("=" * 80)
            print("AGGREGATED OUT-OF-SAMPLE PERFORMANCE")
            print("=" * 80)
            print(f"Mode: {self.mode.upper()}")
            print(f"Total OOS Bars: {len(combined_equity)}")
            print(f"Date Range: {combined_equity.index[0].date()} to {combined_equity.index[-1].date()}")
            print("--- Risk-Adjusted ---")
            print(f"Selected Score ({self.score_mode}): {agg_score:.3f}")
            print(f"Sharpe:   {agg_sharpe:.3f}")
            print(f"Sortino:  {agg_sortino:.3f}")
            print(f"Calmar:   {agg_calmar:.3f}")
            print(f"Composite:{agg_comp:.3f}")
            print("--- Absolute ---")
            print(f"Total Return: {agg_total_ret*100:.2f}%")
            print(f"CAGR:         {agg_cagr*100:.2f}%")
            print(f"Win Rate:     {agg_win_rate*100:.2f}%")
            print(f"Max Drawdown: {max_dd*100:.2f}%")

            out.update(
                dict(
                    agg_sharpe=agg_sharpe,
                    agg_sortino=agg_sortino,
                    agg_calmar=agg_calmar,
                    agg_composite=agg_comp,
                    agg_score=agg_score,
                    agg_total_return=agg_total_ret,
                    agg_cagr=agg_cagr,
                    agg_win_rate=agg_win_rate,
                    agg_max_dd=max_dd,
                    combined_equity=combined_equity,
                    combined_drawdown=drawdown,
                )
            )
        else:
            print("No OOS returns available")

        # Parameter selection summary
        if wf_df is not None and not wf_df.empty:
            best_params_df = wf_df.copy()
            best_params_df["best_simple_window"] = best_params_df["best_params"].apply(lambda p: getattr(p, "simple_window", None))
            best_params_df["best_vol_window"] = best_params_df["best_params"].apply(lambda p: getattr(p, "vol_window", None))
            best_params_df["best_volume_pct"] = best_params_df["best_params"].apply(lambda p: getattr(p, "volume_pct", None))
            best_params_df["best_momentum_pct"] = best_params_df["best_params"].apply(lambda p: getattr(p, "momentum_pct", None))
            best_params_df["best_long_id_threshold"] = best_params_df["best_params"].apply(lambda p: getattr(p, "long_id_threshold", None))
            best_params_df["best_short_id_threshold"] = best_params_df["best_params"].apply(lambda p: getattr(p, "short_id_threshold", None))
            best_params_df["best_weighting_method"] = best_params_df["best_params"].apply(lambda p: getattr(p, "weighting_method", None))

            print("Simple window selection:", best_params_df["best_simple_window"].value_counts().sort_index())
            print("Vol window selection:", best_params_df["best_vol_window"].value_counts().sort_index())
            print("Volume pct selection:", best_params_df["best_volume_pct"].value_counts().sort_index())
            print("Momentum pct selection:", best_params_df["best_momentum_pct"].value_counts().sort_index())
            print("Long ID threshold selection:", best_params_df["best_long_id_threshold"].value_counts().sort_index())
            print("Short ID threshold selection:", best_params_df["best_short_id_threshold"].value_counts().sort_index())
            print("Weighting method selection:", best_params_df["best_weighting_method"].value_counts().sort_index())

            out["best_params_df"] = best_params_df
        else:
            print("wf_df is empty; no parameter summary available.")

        # Recompute per-iteration OOS (to capture turnover like the fast notebook)
        iter_stats: List[Dict[str, Any]] = []
        combined_returns: List[float] = []
        combined_dates: List[pd.Timestamp] = []
        combined_turnover: List[float] = []

        dates_idx = pd.Index(self.bundle.dates)
        if wf_df is not None and not wf_df.empty:
            for row in wf_df.itertuples(index=False):
                params = row.best_params
                strategy = self.strategy_factory(params)
                engine = BacktestEngine(
                    bundle=self.bundle,
                    strategy=strategy,
                    weighting=self.weighting_model,
                    params=params,
                    volume_pct=getattr(params, "volume_pct", 0.2),
                    tc_bps=getattr(params, "tc_bps", 5.0),
                )
                test_start = pd.Timestamp(row.test_start)
                test_end = pd.Timestamp(row.test_end)
                if test_start not in dates_idx or test_end not in dates_idx:
                    continue
                test_start_idx = dates_idx.get_loc(test_start)
                test_end_idx = dates_idx.get_loc(test_end)

                _, ret_full, turn_full, _, _ = engine.run(start_idx=0, end_idx=test_end_idx, collect_ic=False)
                oos_ret = ret_full.iloc[test_start_idx : test_end_idx + 1].dropna()
                oos_turn = turn_full.reindex(oos_ret.index).fillna(0)
                if oos_ret.empty:
                    continue
                oos_eq = (1 + oos_ret).cumprod()

                combined_returns.extend(oos_ret.values)
                combined_dates.extend(oos_ret.index)
                combined_turnover.extend(oos_turn.values)

                iter_traded = oos_ret[oos_ret != 0]
                iter_win_rate = (iter_traded > 0).mean() if not iter_traded.empty else float("nan")

                iter_stats.append(
                    dict(
                        iteration=row.iteration,
                        test_start=test_start,
                        test_end=test_end,
                        simple_window=getattr(params, "simple_window", None),
                        vol_window=getattr(params, "vol_window", None),
                        weighting_method=getattr(params, "weighting_method", None),
                        oos_sharpe=compute_sharpe(oos_ret, periods_per_year=per_year),
                        oos_sortino=compute_sortino_ratio(oos_ret, periods_per_year=per_year),
                        oos_calmar=compute_calmar_ratio(oos_ret, oos_eq, periods_per_year=per_year),
                        oos_composite=compute_composite_score(oos_ret, oos_eq, periods_per_year=per_year),
                        oos_total_return=oos_eq.iloc[-1] - 1,
                        oos_win_rate=iter_win_rate,
                        oos_avg_turnover=oos_turn.mean(),
                    )
                )

        combined_returns_series = pd.Series(combined_returns, index=combined_dates, name="OOS_Returns_Recomputed")
        combined_returns_series = combined_returns_series[~combined_returns_series.index.duplicated(keep="first")].sort_index()
        combined_turnover_series = pd.Series(combined_turnover, index=combined_returns_series.index, name="OOS_Turnover") if not combined_returns_series.empty else pd.Series(dtype=float)
        iter_stats_df = pd.DataFrame(iter_stats)

        out.update(
            dict(
                combined_returns_recomputed=combined_returns_series,
                combined_turnover=combined_turnover_series,
                iter_stats=iter_stats_df,
            )
        )

        # Plots (matplotlib versions of the fast notebook visuals)
        if plot:
            try:
                import matplotlib.pyplot as plt

                plt.style.use("seaborn-v0_8")

                # Equity + drawdown
                if oos_equity is not None and not oos_equity.empty:
                    dd = (oos_equity / oos_equity.cummax() - 1.0) * 100
                    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
                    axes[0].plot(oos_equity.index, oos_equity.values, label="OOS Equity", color="tab:blue")
                    axes[0].set_ylabel("Equity")
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)
                    axes[1].plot(dd.index, dd.values, label="Drawdown %", color="tab:red")
                    axes[1].fill_between(dd.index, dd.values, 0, color="tab:red", alpha=0.2)
                    axes[1].set_ylabel("Drawdown (%)")
                    axes[1].set_xlabel("Date")
                    axes[1].legend()
                    axes[1].grid(True, alpha=0.3)
                    plt.tight_layout()
                    if fig_path:
                        fig.savefig(fig_path / "oos_equity_drawdown.png", dpi=150)
                        plt.close(fig)
                    else:
                        plt.show()

                # Parameter selection counts
                if "best_params_df" in out:
                    bp = out["best_params_df"]
                    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
                    plots = [
                        ("best_simple_window", "Simple Window"),
                        ("best_vol_window", "Vol Window"),
                        ("best_volume_pct", "Volume %"),
                        ("best_momentum_pct", "Momentum %"),
                        ("best_long_id_threshold", "Long ID th"),
                        ("best_short_id_threshold", "Short ID th"),
                    ]
                    for ax, (col, title) in zip(axes.flatten(), plots):
                        bp[col].value_counts().sort_index().plot(kind="bar", ax=ax, color="tab:blue")
                        ax.set_title(title)
                        ax.grid(alpha=0.3)
                    plt.tight_layout()
                    if fig_path:
                        fig.savefig(fig_path / "parameter_selection_counts.png", dpi=150)
                        plt.close(fig)
                    else:
                        plt.show()

                # Parameter values over iterations
                if wf_df is not None and not wf_df.empty:
                    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
                    plot_cols = [
                        ("best_simple_window", "Simple Window"),
                        ("best_vol_window", "Vol Window"),
                        ("best_volume_pct", "Volume %"),
                        ("best_momentum_pct", "Momentum %"),
                        ("best_long_id_threshold", "Long ID th"),
                        ("best_short_id_threshold", "Short ID th"),
                    ]
                    # expand params into columns for plotting
                    temp = wf_df.copy()
                    temp["best_simple_window"] = temp["best_params"].apply(lambda p: getattr(p, "simple_window", np.nan))
                    temp["best_vol_window"] = temp["best_params"].apply(lambda p: getattr(p, "vol_window", np.nan))
                    temp["best_volume_pct"] = temp["best_params"].apply(lambda p: getattr(p, "volume_pct", np.nan))
                    temp["best_momentum_pct"] = temp["best_params"].apply(lambda p: getattr(p, "momentum_pct", np.nan))
                    temp["best_long_id_threshold"] = temp["best_params"].apply(lambda p: getattr(p, "long_id_threshold", np.nan))
                    temp["best_short_id_threshold"] = temp["best_params"].apply(lambda p: getattr(p, "short_id_threshold", np.nan))

                    for ax, (col, title) in zip(axes.flatten(), plot_cols):
                        ax.plot(temp["iteration"], temp[col], marker="o", linestyle="-", color="tab:blue")
                        ax.set_title(title)
                        ax.set_xlabel("Iteration")
                        ax.grid(alpha=0.3)
                    plt.tight_layout()
                    if fig_path:
                        fig.savefig(fig_path / "parameter_selection_over_time.png", dpi=150)
                        plt.close(fig)
                    else:
                        plt.show()

                # IS vs OOS Sharpe comparison
                if wf_df is not None and not wf_df.empty:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(wf_df["iteration"], wf_df["is_sharpe"], label="IS Sharpe", marker="o")
                    ax.plot(wf_df["iteration"], wf_df["oos_sharpe"], label="OOS Sharpe", marker="o")
                    ax.set_xlabel("Iteration")
                    ax.set_ylabel("Sharpe")
                    ax.set_title("IS vs OOS Sharpe")
                    ax.grid(alpha=0.3)
                    ax.legend()
                    plt.tight_layout()
                    if fig_path:
                        fig.savefig(fig_path / "is_oos_sharpe.png", dpi=150)
                        plt.close(fig)
                    else:
                        plt.show()

                # BTC benchmark vs strategy equity (if BTC present)
                if oos_equity is not None and not oos_equity.empty and "BTCUSDT" in self.bundle.price_df.columns:
                    btc_prices = self.bundle.price_df["BTCUSDT"].reindex(oos_equity.index)
                    if len(btc_prices) > 0 and not pd.isna(btc_prices.iloc[0]):
                        initial_btc = btc_prices.iloc[0]
                        btc_equity = btc_prices.fillna(method="ffill") / initial_btc
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(oos_equity.index, oos_equity.values, label="Strategy", color="tab:blue")
                        ax.plot(btc_equity.index, btc_equity.values, label="BTC Buy & Hold", color="tab:orange", linestyle="--")
                        ax.set_title("Strategy vs BTC Benchmark (OOS)")
                        ax.set_ylabel("Equity (start=1)")
                        ax.set_xlabel("Date")
                        ax.grid(alpha=0.3)
                        ax.legend()
                        plt.tight_layout()
                        if fig_path:
                            fig.savefig(fig_path / "btc_benchmark.png", dpi=150)
                            plt.close(fig)
                        else:
                            plt.show()

                # Turnover time series + histogram (recomputed)
                if not combined_turnover_series.empty:
                    fig, axes = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [2, 1]})
                    axes[0].plot(combined_turnover_series.index, combined_turnover_series.values * 100, color="purple")
                    axes[0].set_title("Daily Turnover (%)")
                    axes[0].set_ylabel("Turnover (%)")
                    axes[0].grid(alpha=0.3)
                    axes[1].hist(combined_turnover_series.values * 100, bins=40, color="purple", alpha=0.7)
                    axes[1].set_xlabel("Turnover (%)")
                    axes[1].set_ylabel("Frequency")
                    plt.tight_layout()
                    if fig_path:
                        fig.savefig(fig_path / "turnover.png", dpi=150)
                        plt.close(fig)
                    else:
                        plt.show()
            except ImportError:
                print("matplotlib not available; skipping plots.")

        return out
