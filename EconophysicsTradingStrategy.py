import numpy as np
import pandas as pd
import yfinance as yf
import nolds
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

ticker1 = 'BRK-A'
ticker2 = 'BRK-B'
data = yf.download([ticker1, ticker2], start='2010-01-01')['Close']

data = data.dropna()

P1 = data[ticker1]
P2 = data[ticker2]

hurst_window = 252
trading_window = 60

positions = []
hursts = []
zscores = []

lp1 = np.log(P1)
lp2 = np.log(P2)

spread = lp1 - lp2

threshold = 1

for t in range(hurst_window, len(spread)):
    spread_hurst = spread.iloc[t - hurst_window:t]
    spread_diff = spread_hurst.diff().dropna()

    nvals = [8, 16, 32, 64]
    H_t = nolds.dfa(spread_diff.values, nvals=nvals, overlap=False, order=1)
    hursts.append(H_t)

    pos = 0

    if H_t < 0.6 and H_t > 0.15:
        spread_trade = spread.iloc[t - trading_window:t]
        mu = spread_trade.mean()
        sigma = spread_trade.std()

        if sigma == 0:
            zscore = 0
        else:
            zscore = (spread.iloc[t] - mu) / sigma

        zscores.append(zscore)

        if abs(zscore) > threshold:
            pos = (-np.sign(zscore))

    else:
        zscores.append(0)

    positions.append(pos)

idx = spread.index[hurst_window:]
positions = pd.Series(positions, index=idx)
hursts = pd.Series(hursts, index=idx)
zscores = pd.Series(zscores, index=idx)

# 1. Spread changes
spread_change = spread.diff()

# 2. Daily PnL (use yesterday's position → no look-ahead)
pnl = positions.shift(1) * spread_change

# 3. Clean NaNs
pnl = pnl.fillna(0.0)

# 4. Cumulative PnL
cumulative_pnl = pnl.cumsum()

plt.plot(cumulative_pnl)

final_pnl = cumulative_pnl.iloc[-1]
mean_pnl = pnl.mean()
annual_pnl = pnl.mean() * 252
sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252)
downside_std = pnl[pnl < 0].std()
sortino = (pnl.mean() / downside_std) * np.sqrt(252)
win_rate = (pnl >= 0).mean()
avg_win = pnl[pnl>0].mean()
avg_loss = pnl[pnl<0].mean()
profit_factor = avg_win/abs(avg_loss)
var = pnl.quantile(0.05)
cvar = pnl[pnl <= var].mean()

rolling_max = cumulative_pnl.cummax()
drawdown = cumulative_pnl - rolling_max
max_drawdown = drawdown.min()
drawdown_duration = (drawdown < 0).astype(int).groupby(
    (drawdown >= 0).astype(int).cumsum()
).sum().max()
volatility = pnl.std() * np.sqrt(252)
num_trades = (positions.diff().abs() > 0).sum()
time_in_market = (positions != 0).mean()
trade_lengths = positions.ne(positions.shift()).cumsum()
avg_holding = positions.groupby(trade_lengths).apply(
    lambda x: (x != 0).sum()
).mean()
turnover = positions.diff().abs().sum()

print("STRATEGY OVERVIEW")

print("\nPERFORMANCE METRICS:")
print(f"Final Pnl: {final_pnl * 100: .2f}%")
print(f"Mean PnL: {mean_pnl * 100: .2f}%")
print(f"Annual PnL: {annual_pnl * 100: .2f}%")
print(f"Sharpe Ratio: {sharpe: .3f}")
print(f"Downside Standard Deviation: {downside_std * 100: .2f}%")
print(f"Sortino Ratio: {sortino: .3f}")
print(f"Win Rate: {win_rate*100: .2f}%")
print(f"Profit Factor: {profit_factor: .2f}")
print(f"CVaR: {cvar*100:.2f}%")

print("\nRISK METRICS:")
print(f"Max Drawdown: {max_drawdown * 100: .2f}%")
print(f"Drawdown Duration: {drawdown_duration} Days")
print(f"Volatility: {volatility * 100: .2f}%")

print("\nTRADING BEHAVIOR METRICS:")
print(f"Number of Trades: {num_trades}")
print(f"Percent of Time in Market: {time_in_market * 100: .2f}%")
print(f"Average Holding Period: {avg_holding: .2f}")
print(f"Turnover: {turnover: .2f}")

print("STRUCTURAL METRICS")

hursts_series = pd.Series(
    hursts,
    index=spread.index
)

accepted_regime = (
    (hursts_series >= 0.15) &
    (hursts_series <= 0.6)
)

positions_series = pd.Series(
    positions,
    index=spread.index
)

hurst_acceptance_rate = accepted_regime.mean()

trading_mask = positions_series != 0
avg_hurst_in_trades = hursts_series[trading_mask].mean()

zscore_total = ((spread - spread.mean()) / spread.std())

mean_spread_signal = np.abs(zscore_total[trading_mask]).mean()

regime_blocks = accepted_regime.ne(accepted_regime.shift()).cumsum()
avg_regime_length = accepted_regime.groupby(regime_blocks).sum().mean()

spread_vol_regime = spread[accepted_regime].std()
spread_vol_out = spread[~accepted_regime].std()

print(f"Hurst Acceptance Rate: {hurst_acceptance_rate * 100: .2f}%")
print(f"Average Hurst in Trades: {avg_hurst_in_trades: .3f}")
print(f"Mean Spread Signal: {mean_spread_signal: .3f}")
print(f"Average Regime Length: {avg_regime_length: .1f} Days")
print(f"Spread Volatility in Regime: {spread_vol_regime: .6f}% VS Spread Volatility Put of Regime: {spread_vol_out: .6f}%")

