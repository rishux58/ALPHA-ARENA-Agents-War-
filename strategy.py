"""
AGENTS WAR - Quantitative Trading Hackathon (Alpha Arena).
Strategy Implementation by Team Alpha Vanguard.

OPTIMIZED FOR 0.3% ON-SPOT TRANSACTION COST STRUCTURE.

Strategies Implemented:
  1. decide_risk_adj_momentum: Medium-term momentum scaled by inverse volatility with turnover smoothing.
  2. decide_mean_reversion_rsi: 14-day RSI and Bollinger Band Z-score pullback strategy.
  3. decide_inverse_vol_defensive: Risk parity / low volatility allocation with 50-day SMA trend filter.
  4. decide_dual_ma_trend: Dual EMA crossover (20d vs 60d) normalized by ATR with hysteresis.
  5. decide_hybrid_ensemble: Master Winning Strategy combining Momentum, RSI Pullbacks, learned Ridge
     regression, Dynamic 3-Regime Market Filter, and 0.3% fee-optimized execution inertia.
"""

import math
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

TEAM_NAME = "Pixel_Arc"
TRAIN_WINDOW = (1188, 3707)   # Best generalizing 10-year window (2520 days, up to day 3707)
MODEL = {}                    # Learned parameters stored here

# Internal execution state for rebalancing buffer & inertia
_STRAT_STATE = {}


def _reset_state_if_needed(strat_name, current_day, raw_weights):
    """Resets internal tracking state if a new simulation/backtest begins."""
    if strat_name not in _STRAT_STATE or current_day <= _STRAT_STATE[strat_name].get("last_day", -1):
        _STRAT_STATE[strat_name] = {
            "last_day": current_day,
            "last_rebal": current_day,
            "weights": raw_weights.copy(),
            "prev_regime": None
        }
        return True
    return False


def _apply_inertia(strat_name, raw_weights, current_day, threshold=0.045, rebal_cadence=7):
    """
    Applies aggressive rebalancing inertia optimized for 0.3% transaction cost:
    - Holds positions between rebalance intervals (rebal_cadence = 7 days)
    - Deadband threshold (0.045): small weight drifts are not traded
    """
    if _reset_state_if_needed(strat_name, current_day, raw_weights):
        return raw_weights.copy()

    state = _STRAT_STATE[strat_name]
    days_held = current_day - state["last_rebal"]
    prev_w = state["weights"]
    all_stocks = set(raw_weights.keys()) | set(prev_w.keys())
    max_diff = max(abs(raw_weights.get(s, 0.0) - prev_w.get(s, 0.0)) for s in all_stocks) if all_stocks else 0.0

    if days_held < rebal_cadence and max_diff < 0.15 and len(prev_w) > 0:
        state["last_day"] = current_day
        return state["weights"].copy()

    updated = {}
    for s in all_stocks:
        tw = raw_weights.get(s, 0.0)
        pw = prev_w.get(s, 0.0)
        if abs(tw - pw) < threshold:
            updated[s] = pw
        else:
            updated[s] = tw

    gross = sum(abs(v) for v in updated.values())
    if gross > 1.0:
        updated = {s: v / gross for s, v in updated.items()}

    state["weights"] = updated.copy()
    state["last_rebal"] = current_day
    state["last_day"] = current_day
    return updated.copy()


def train(data):
    """
    Called ONCE with only the days in TRAIN_WINDOW (up to 2520 days).
    Learns:
      1. Equal-weighted market volatility thresholds (median and 75th percentile).
      2. Best moving average length for trend filtering.
      3. Cross-sectional Ridge Regression model predicting 10-day forward relative alpha.
    """
    global _STRAT_STATE
    _STRAT_STATE.clear()

    c = data["close"]
    stk = list(c.columns)

    # 1. Learn Market Regime Parameters
    mkt = c.mean(axis=1)
    mkt_ret = mkt.pct_change().fillna(0.0)
    mkt_vol20 = mkt_ret.rolling(20, min_periods=10).std() * np.sqrt(252)
    MODEL["vol_median"] = float(mkt_vol20.median())
    MODEL["vol_high"] = float(mkt_vol20.quantile(0.75))

    # 2. Learn Best MA Window
    next_day = c.pct_change().shift(-1).fillna(0.0)
    best_sharpe, best_ma = -np.inf, 50
    for ma_len in [30, 50, 75, 100]:
        hold = c > c.rolling(ma_len, min_periods=10).mean()
        d_ret = (next_day * hold).sum(axis=1) / hold.sum(axis=1).replace(0, np.nan)
        sh = d_ret.mean() / (d_ret.std() + 1e-8)
        if sh > best_sharpe:
            best_sharpe, best_ma = sh, ma_len
    MODEL["best_ma"] = best_ma

    # 3. Train Cross-Sectional ML Model (Ridge Regression)
    X_list, y_list = [], []
    ret = c.pct_change().fillna(0.0)
    ret10 = (c / c.shift(10) - 1.0).fillna(0.0)
    ret30 = (c / c.shift(30) - 1.0).fillna(0.0)
    ret60 = (c / c.shift(60) - 1.0).fillna(0.0)
    fwd10 = (c.shift(-10) / c - 1.0)
    mkt_fwd10 = (mkt.shift(-10) / mkt - 1.0)

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=5).mean().fillna(0.0)
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean().fillna(0.0)
    rsi14 = 100.0 - (100.0 / (1.0 + (gain / (loss + 1e-8))))

    sma20 = c.rolling(20, min_periods=10).mean()
    std20 = c.rolling(20, min_periods=10).std().replace(0, 1e-6)
    bz20 = ((c - sma20) / std20).fillna(0.0)
    vol40 = (ret.rolling(40, min_periods=10).std() * np.sqrt(252)).fillna(0.25)

    sample_days = c.index[60:-10:5]
    for d in sample_days:
        m_fwd = mkt_fwd10.loc[d]
        if np.isnan(m_fwd):
            continue
        for s in stk:
            val_fwd = fwd10.loc[d, s]
            if np.isnan(val_fwd):
                continue
            X_list.append([
                float(ret10.loc[d, s]), float(ret30.loc[d, s]), float(ret60.loc[d, s]),
                float(rsi14.loc[d, s] - 50.0) / 25.0, float(bz20.loc[d, s]), float(vol40.loc[d, s])
            ])
            y_list.append(float(val_fwd - m_fwd))

    if len(X_list) > 100:
        ridge = Ridge(alpha=50.0)
        ridge.fit(np.array(X_list), np.array(y_list))
        MODEL["ridge_coef"] = ridge.coef_.copy()
        MODEL["ridge_intercept"] = float(ridge.intercept_)
    else:
        MODEL["ridge_coef"] = np.zeros(6)
        MODEL["ridge_intercept"] = 0.0


def decide_risk_adj_momentum(history):
    """
    Strategy 1: Medium-Term Momentum normalized by 20-day return volatility.
    Long top-ranked momentum stocks, underweight or hedge bottom stocks,
    scaled by inverse volatility with 0.3% fee inertia smoothing.
    """
    close = history["close"]
    n_days = len(close)
    stk = list(close.columns)
    day = close.index[-1]
    if n_days < 10:
        return {s: 1.0 / len(stk) for s in stk}

    d60 = min(60, n_days - 1)
    d120 = min(120, n_days - 1)
    ret60 = close.iloc[-1] / close.iloc[-d60] - 1.0
    ret120 = close.iloc[-1] / close.iloc[-d120] - 1.0
    mom = 0.5 * ret60 + 0.5 * ret120

    d20 = min(20, n_days - 1)
    daily_ret = close.pct_change().fillna(0.0)
    vol20 = daily_ret.iloc[-d20:].std().replace(0, 1e-4).fillna(1e-4)

    risk_adj_mom = mom / vol20
    ranks = risk_adj_mom.sort_values(ascending=False)

    top4 = ranks.index[:4]
    inv_vol = 1.0 / vol20
    top_inv = inv_vol[top4].sum()

    raw = {}
    for s in top4:
        raw[s] = 0.82 * (inv_vol[s] / top_inv)

    bottom1 = ranks.index[-1]
    if mom[bottom1] < -0.05:
        raw[bottom1] = -0.08

    return _apply_inertia("momentum", raw, day, threshold=0.045, rebal_cadence=7)


def decide_mean_reversion_rsi(history):
    """
    Strategy 2: Mean Reversion based on 14-day RSI and 20-day Bollinger Band Z-score.
    Long oversold stocks with volume stabilization, short overbought extremes,
    weighted by inverse volatility with wide deadband inertia.
    """
    close = history["close"]
    vol = history["volume"]
    n_days = len(close)
    stk = list(close.columns)
    day = close.index[-1]
    if n_days < 10:
        return {s: 1.0 / len(stk) for s in stk}

    delta = close.diff()
    d14 = min(14, n_days - 1)
    gain = delta.clip(lower=0).iloc[-d14:].mean().fillna(0.0)
    loss = (-delta.clip(upper=0)).iloc[-d14:].mean().fillna(0.0)
    rsi14 = 100.0 - (100.0 / (1.0 + (gain / (loss + 1e-8))))

    d20 = min(20, n_days - 1)
    sma20 = close.iloc[-d20:].mean()
    std20 = close.iloc[-d20:].std().replace(0, 1e-6).fillna(1e-6)
    bz = (close.iloc[-1] - sma20) / std20

    v5 = vol.iloc[-min(5, n_days - 1):].mean()
    v20 = vol.iloc[-d20:].mean().replace(0, 1e-6)
    vol_ratio = (v5 / v20).fillna(1.0)

    daily_ret = close.pct_change().fillna(0.0)
    vol_ret = daily_ret.iloc[-d20:].std().replace(0, 1e-4).fillna(1e-4)
    inv_vol = 1.0 / vol_ret

    score = - (0.6 * bz + 0.4 * ((rsi14 - 50.0) / 15.0))
    score = score * np.where(vol_ratio > 0.6, 1.0, 0.7)

    sorted_stocks = score.sort_values(ascending=False)
    longs = sorted_stocks.index[:4]

    raw = {}
    l_sum = inv_vol[longs].sum()
    for s in longs:
        raw[s] = 0.78 * (inv_vol[s] / l_sum)

    overbought = [s for s in stk if bz[s] > 1.8 and rsi14[s] > 70]
    if overbought:
        sh = overbought[0]
        raw[sh] = -0.10

    return _apply_inertia("mean_rev", raw, day, threshold=0.055, rebal_cadence=8)


def decide_inverse_vol_defensive(history):
    """
    Strategy 3: Low-Volatility / Risk-Parity Defensive strategy with 50-day SMA trend filter.
    Long stocks above their 50-day SMA, inverse volatility weighted.
    Shift to defensive low-beta allocation with controlled hedging when market trend is negative.
    """
    close = history["close"]
    n_days = len(close)
    stk = list(close.columns)
    day = close.index[-1]
    if n_days < 10:
        return {s: 1.0 / len(stk) for s in stk}

    d50 = min(50, n_days - 1)
    sma50 = close.iloc[-d50:].mean()
    above50 = close.iloc[-1] > sma50

    d40 = min(40, n_days - 1)
    daily_ret = close.pct_change().fillna(0.0)
    vol40 = daily_ret.iloc[-d40:].std().replace(0, 1e-4).fillna(1e-4)
    inv_vol = 1.0 / vol40

    mkt = close.mean(axis=1)
    mkt_sma50 = mkt.iloc[-d50:].mean()
    mkt_uptrend = mkt.iloc[-1] > mkt_sma50

    raw = {}
    if mkt_uptrend:
        picks = [s for s in stk if above50[s]]
        if len(picks) == 0:
            picks = inv_vol.nlargest(4).index.tolist()
        total_inv = sum(inv_vol[s] for s in picks)
        target_gross = 0.86
        for s in picks:
            raw[s] = target_gross * (inv_vol[s] / total_inv)
    else:
        low_vol_stocks = inv_vol.nlargest(3).index
        total_low_vol = sum(inv_vol[s] for s in low_vol_stocks)
        for s in low_vol_stocks:
            raw[s] = 0.50 * (inv_vol[s] / total_low_vol)
        below_stocks = [s for s in stk if not above50[s]]
        if len(below_stocks) >= 2:
            high_vol_below = vol40[below_stocks].nlargest(2).index
            total_hv = sum(vol40[s] for s in high_vol_below)
            for s in high_vol_below:
                raw[s] = -0.15 * (vol40[s] / total_hv)
        else:
            raw[low_vol_stocks[0]] += 0.15

    return _apply_inertia("defensive", raw, day, threshold=0.045, rebal_cadence=7)


def decide_dual_ma_trend(history):
    """
    Strategy 4: Dual Moving Average Crossover (20-day EMA vs 60-day EMA) normalized by ATR.
    Hysteresis and inertia prevent whipsaw trading and transaction costs.
    """
    close = history["close"]
    high = history["high"]
    low = history["low"]
    n_days = len(close)
    stk = list(close.columns)
    day = close.index[-1]
    if n_days < 10:
        return {s: 1.0 / len(stk) for s in stk}

    d20 = min(20, n_days - 1)
    d60 = min(60, n_days - 1)
    ema20 = close.ewm(span=d20, adjust=False).mean().iloc[-1]
    ema60 = close.ewm(span=d60, adjust=False).mean().iloc[-1]

    prev_c = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_c).abs(), (low - prev_c).abs()))
    atr20 = tr.iloc[-d20:].mean().replace(0, 1e-4).fillna(1e-4)

    spread = (ema20 - ema60) / atr20

    daily_ret = close.pct_change().fillna(0.0)
    vol20 = daily_ret.iloc[-d20:].std().replace(0, 1e-4).fillna(1e-4)
    inv_vol = 1.0 / vol20

    pos_trend = [s for s in stk if spread[s] > 0.05]
    if len(pos_trend) < 3:
        pos_trend = spread.nlargest(4).index.tolist()

    raw = {}
    sum_l = sum(inv_vol[s] for s in pos_trend)
    for s in pos_trend:
        raw[s] = 0.84 * (inv_vol[s] / sum_l)

    neg_trend = [s for s in stk if spread[s] < -0.40]
    if neg_trend:
        sh = neg_trend[0]
        raw[sh] = -0.10

    return _apply_inertia("dual_ma", raw, day, threshold=0.045, rebal_cadence=7)


def decide_hybrid_ensemble(history):
    """
    Strategy 5: Master Winning Strategy (Optimized for 0.3% Fee Structure).
    Combines:
      - Trend-following (50-day SMA + 60d/120d Momentum)
      - Statistical Mean Reversion (14d RSI & 20d Bollinger Z pullbacks)
      - Machine Learning (Cross-Sectional Ridge Regression trained on TRAIN_WINDOW)
      - Inverse Volatility Risk Parity sizing
      - Dynamic Market Regime Filter (Bull, Crash, Defensive)
      - 0.3% Fee-Optimized Rebalancing Buffer & Inertia
    """
    close = history["close"]
    n_days = len(close)
    stk = list(close.columns)
    day = close.index[-1]
    if n_days < 10:
        return {s: 1.0 / len(stk) for s in stk}

    # Reset state if new run begins
    if _reset_state_if_needed("hybrid_ensemble", day, {s: 1.0 / len(stk) for s in stk}):
        return _STRAT_STATE["hybrid_ensemble"]["weights"].copy()

    state = _STRAT_STATE["hybrid_ensemble"]

    # 1. Market Index & Dynamic Regime Detection
    mkt = close.mean(axis=1)
    d50 = min(50, n_days - 1)
    mkt_sma50 = float(mkt.iloc[-d50:].mean())
    mkt_uptrend = float(mkt.iloc[-1]) > mkt_sma50

    mkt_ret = mkt.pct_change().fillna(0.0)
    d20 = min(20, n_days - 1)
    mkt_vol20 = float(mkt_ret.iloc[-d20:].std() * np.sqrt(252))
    d60 = min(60, n_days - 1)
    mkt_ret60 = float(mkt.iloc[-1] / mkt.iloc[-d60] - 1.0)

    vol_med = MODEL.get("vol_median", 0.16)
    vol_high = MODEL.get("vol_high", 0.20)

    is_crash = (not mkt_uptrend) and (mkt_vol20 > vol_med or mkt_ret60 < -0.03)
    regime = "CRASH" if is_crash else ("BULL" if mkt_uptrend else "DEFENSIVE")

    regime_changed = (regime != state["prev_regime"])
    days_since = day - state["last_rebal"]

    # Rebalance cadence: 7 days or on market regime shift (cuts 0.3% fee drag)
    if not regime_changed and days_since < 7 and len(state["weights"]) > 0:
        state["last_day"] = day
        return state["weights"].copy()

    state["prev_regime"] = regime
    state["last_rebal"] = day
    state["last_day"] = day

    # 2. Cross-Sectional Features & Signals
    d_ret = close.pct_change().fillna(0.0)
    d40 = min(40, n_days - 1)
    vol40 = d_ret.iloc[-d40:].std().replace(0, 1e-4).fillna(1e-4)
    inv_vol = 1.0 / vol40

    sma50 = close.iloc[-d50:].mean()
    above50 = close.iloc[-1] > sma50

    d120 = min(120, n_days - 1)
    mom = 0.5 * (close.iloc[-1] / close.iloc[-d60] - 1.0) + 0.5 * (close.iloc[-1] / close.iloc[-d120] - 1.0)

    delta = close.diff()
    d14 = min(14, n_days - 1)
    gain = delta.clip(lower=0).iloc[-d14:].mean().fillna(0.0)
    loss = (-delta.clip(upper=0)).iloc[-d14:].mean().fillna(0.0)
    rsi14 = 100.0 - (100.0 / (1.0 + (gain / (loss + 1e-8))))

    sma20 = close.iloc[-d20:].mean()
    std20 = close.iloc[-d20:].std().replace(0, 1e-6).fillna(1e-6)
    bz20 = (close.iloc[-1] - sma20) / std20
    pullback = np.where(above50 & (bz20 < 0), -0.5 * bz20, 0.0)

    # ML Model Prediction
    coef = MODEL.get("ridge_coef", np.zeros(6))
    intercept = MODEL.get("ridge_intercept", 0.0)
    d10 = min(10, n_days - 1)
    ret10 = (close.iloc[-1] / close.iloc[-d10] - 1.0).fillna(0.0)
    ret30 = (close.iloc[-1] / close.iloc[-min(30, n_days - 1)] - 1.0).fillna(0.0)
    ret60_stk = (close.iloc[-1] / close.iloc[-d60] - 1.0).fillna(0.0)

    X = np.column_stack([
        ret10.values, ret30.values, ret60_stk.values,
        ((rsi14.values - 50.0) / 25.0), bz20.values, (vol40.values * np.sqrt(252))
    ])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    ml_score = pd.Series(X @ coef + intercept, index=stk)

    z_mom = (mom - mom.mean()) / (mom.std() + 1e-6)
    z_ml = (ml_score - ml_score.mean()) / (ml_score.std() + 1e-6)
    comp = 0.45 * z_mom + 0.35 * z_ml + 0.20 * pd.Series(pullback, index=stk)

    raw = {}
    if regime == "BULL":
        picks = [s for s in stk if above50[s]]
        if len(picks) == 0:
            picks = inv_vol.nlargest(4).index.tolist()

        weights = {}
        for s in picks:
            c_score = float(comp[s])
            mult = np.clip(1.0 + 0.5 * c_score, 0.6, 1.5)
            weights[s] = inv_vol[s] * mult

        tot_w = sum(weights.values())
        target_gross = 0.88
        for s in picks:
            raw[s] = target_gross * (weights[s] / tot_w)

    elif regime == "CRASH":
        def_stks = inv_vol.nlargest(3).index
        tot_def = sum(inv_vol[s] for s in def_stks)
        for s in def_stks:
            raw[s] = 0.38 * (inv_vol[s] / tot_def)

        below = [s for s in stk if not above50[s]]
        if len(below) >= 2:
            sh_stks = vol40[below].nlargest(2).index
            tot_sh = sum(vol40[s] for s in sh_stks)
            for s in sh_stks:
                raw[s] = -0.18 * (vol40[s] / tot_sh)
        else:
            raw[def_stks[0]] += 0.18

    else:  # DEFENSIVE
        def_stks = inv_vol.nlargest(4).index
        tot_def = sum(inv_vol[s] for s in def_stks)
        for s in def_stks:
            raw[s] = 0.68 * (inv_vol[s] / tot_def)

        below = [s for s in stk if not above50[s]]
        if below:
            worst = vol40[below].idxmax()
            raw[worst] = -0.10

    # Deadband thresholding for 0.3% fee
    prev_w = state["weights"]
    all_s = set(raw.keys()) | set(prev_w.keys())
    updated = {}
    for s in all_s:
        tw = raw.get(s, 0.0)
        pw = prev_w.get(s, 0.0)
        if abs(tw - pw) < 0.050:
            updated[s] = pw
        else:
            updated[s] = tw
    gross = sum(abs(v) for v in updated.values())
    if gross > 1.0:
        updated = {s: v / gross for s, v in updated.items()}

    state["weights"] = updated.copy()
    return updated


# Point decide to our Master Winning Strategy
decide = decide_hybrid_ensemble


if __name__ == "__main__":
    import sys
    from pathlib import Path
    import engine

    data = engine.load(Path(__file__).resolve().parent / "data" / "train.csv")
    window = engine.train_agent(sys.modules[__name__], data)
    log = engine.backtest(decide, data, verbose=False)["log"]
    inside = (log.index >= window[0]) & (log.index <= window[1])
    print(f"{TEAM_NAME}: trained on days {window[0]}-{window[1]} (COST_PER_TRADE = {engine.COST_PER_TRADE*100:.1f}%)")
    print("\nInside the training window:")
    engine.print_metrics(engine.metrics(log[inside]))
    print("Outside the training window (validation):")
    engine.print_metrics(engine.metrics(log[~inside]))
