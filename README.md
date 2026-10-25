# Autonomous Deriv Trading Engine v2.0 (AI-Guided Confluence & Dual-Execution)

An autonomous algorithmic trading suite built for Deriv (Synthetic Indices, Volatilities, Jump, Step, Crash/Boom, Crypto, Forex).

Equipped with:
- **Confluence Engine**: Triple EMA trend filter, ADX momentum ($\ge 20$), dynamic RSI bounds, Bollinger %B, and directional MACD. The "72.4%" figure that used to be quoted here came from a handful of demo trades; the measured out-of-sample win rate is **~50%** (see below).
- **Dual-Execution Support**: Multipliers with real-time AI profit locking & loss caps, and Fixed Expiry Rise/Fall (CALL/PUT) options.
- **Gemini AI Integration**: Real-time Pre-Trade Veto filter and automated post-loss forensic root cause analysis.
- **Continuous Learning & Risk Hard-Limits**: SQLite persistent journal with SQLite WAL mode, milestone performance banners, and automatic asset pruning.
- **Fixed Safe Sizing**: Fixed stake of **$1.00** per contract. A **$10 real account cannot trade this configuration** — the risk caps veto every entry. See "The $10 account" below before funding anything.

---

## ⚠️ Measured reality (read before trusting the numbers above)

The "72.4% win rate" claim came from a handful of demo trades, not from a
backtest. The repository now contains an offline replay harness
(`research/`) that runs the **exact** `StrategyEngine` entry logic over
~20,000 real 1-minute candles per symbol (33 symbols, ~14 days) and settles
each contract on the real forward price with the **real quoted payouts**
(`research/payouts.json`).

Result of the current Rise/Fall strategy, after excluding the one measured-defect
symbol `stpRNG4` (see below), 679 simulated trades:

| Slice | Trades | Win rate | Net PnL @ $1 | Expectancy |
|---|---|---|---|---|
| All 14 days | 679 | 50.7% | **-$23.24** | -$0.034 |
| Train (first 65%) | 456 | 53.3% | +$7.50 | +$0.016 |
| Validation (last 35%) | 223 | 45.3% | -$30.74 | -$0.138 |

Before that exclusion the same replay was 725 trades, 49.1% win rate,
**-$45.84**, -$0.063/trade. The exclusion halves the loss; it does not create
an edge.

Breakeven at a 0.88-0.95 payout is **~51.3-53.2%**. The strategy is below it on
the full sample and on validation. Every filter that looked good in-sample
(RSI bands, ADX bands, symbol whitelists, expiry choice) collapsed on
validation — the best fitted rule went 61.9% train -> 49.0% validation, and
walk-forward expectancy was +0.013/trade (statistically zero). Fading the
signals also lost, which is the signature of a fair random walk plus payout
edge, not of a weak signal.

Deriv synthetics are RNG-driven; no indicator combination can create an edge
over the quoted payout. This is not a tuning problem.

### The direct proof: is direction predictable at all? (2026-10-25)

`research/fair_game_test.py` tests the assumption every filter depends on,
instead of testing more filters. On **4,865,389** non-flat 1-minute moves
across 23 symbols and every expiry 1-10m:

* Direction among moves that actually moved: **50.160%** up
  (95% CI 50.116-50.205%). Real, but tiny.
* Required edge to break even: **+1.28 to +3.19 pp** above a coin flip
  (payout 0.95 -> 51.28%, payout 0.88 -> 53.19%).
* Largest directional bias found anywhere, after choosing the best expiry per
  symbol: **1.48 pp** — and it points the wrong way for that symbol's payout
  (1HZ15V fades at 51.48% against a 53.19% breakeven).
* 6 of 23 symbols pass Bonferroni correction for *any* bias, but
  **0 of 230 (symbol, expiry) pairs clear their own breakeven** once ties are
  counted as losses.

**The tie trap.** On step indices the price sits on a 0.1 grid and does not
move at all on **10.3%** of 1-minute expiries (**4.7%** at 6m, **3.3%** at
10m). A tie loses for both Rise and Fall, so it is a pure structural cost that
no signal can avoid. It caps a zero-skill step-index strategy at a **48.3%**
win rate against a **51.28%** breakeven — worse than the continuous indices,
which is why the 0.95 payout on `stpRNG` is an illusion.

Ranked by structural loss (coin-flip call, ties lose), the least-bad
configuration is JD10/JD25/JD50 at short expiry at **-4.00%/trade**; the worst
is `stpRNG4` at 1m at **-12.85%/trade**. The shipped strategy's `stpRNG` 6m
profile is about **-7.1%/trade**. Even the best configuration is negative:
this ranks how fast you lose, not whether you win.

**A candidate that died on full data.** With a truncated 1.7-day sample, JD75
at 8-10m showed a +1.09 pp edge (fade, 53.17% vs 52.08% breakeven). Fetching
the symbol's full 14-day history erased it: 49.7-50.3% at every expiry, edge
**-1.9 to -2.4 pp**. It is kept here as a reminder that a short window plus a
best-of-ten expiry search manufactures edges out of nothing.

**Correction on `stpRNG4`.** The fair-game test finds no directional defect in
`stpRNG4` (p=0.0009 at 7m, does not survive correction), and its tie rate is
the lowest in its family. The exclusion shipped earlier was selected from a
46-trade sample; the 7-day improvement it produced is in-sample. Treat it as a
harmless but unproven filter, not as a discovered defect.

### The accuracy question (why 80% is not a tuning target)

The request was to raise accuracy to ~80%. On 440,000+ 1-minute bars the
instrument behaves like a driftless martingale:

* Autocorrelation of 1m returns is ~0 at every lag 1-10 (median |rho| < 0.006).
* "Next 5 minutes continue the last bar" hits **49.4%**; fade it and it hits
  **49.5%**. A run of 6 same-direction bars fades at **51.2%** — the largest
  real effect found, and still below the 51.3% breakeven at the best payout.
* 822 conditional fade rules were searched across run length, RSI, %B, ADX,
  ATR percentile, body size, hour and symbol family. Nine cleared breakeven on
  both halves; on a **true holdout** (Oct 24 16:00 -> Oct 25 05:00, never used
  in the search) seven failed and the two survivors had 34-42 trades with
  confidence intervals spanning breakeven. That is noise, not edge.
* The same is true of the bot's own signals: no score/ADX/RSI/%B/hour/symbol
  bucket improves both halves, and a fixed 3-minute expiry (the only lever that
  did) holds in just 3/5 time folds and is mixed per symbol.

Win rate and profit are not the same thing. An 80% win rate is trivially
available by taking a far barrier at a low payout, but at a 0.20 payout the
breakeven is **83.3%**, so 80% still loses. Monte Carlo of the exact setup
($10 start, $1 stake, target $20):

| Scenario | Win rate | Payout | P(double) | P(ruin) |
|---|---|---|---|---|
| Measured | 49.1% | 0.908 | 18.4% | **81.6%** |
| Optimistic | 55.0% | 0.880 | 66.1% | 33.9% |
| Breakeven | 52.4% | 0.908 | 47.5% | 52.5% |
| "80% accuracy" | 80.0% | 0.200 | 2.0% | 98.0% |

The realistic ceiling for this family of rules is **50-55%**, and the measured
book is at the bottom of it.

### The live demo session (2026-10-24) and the one losing trade

A demo exploration session ran 11:22-11:56 UTC at a fixed $1 stake and closed
4 trades: **3 wins / 1 loss, +$1.72, 75% win rate** (95% CI 30.1-95.4%;
P(WR >= 75% if truly at breakeven) = 0.357). Four trades cannot distinguish a
real edge from noise, so this is recorded as inconclusive, not as evidence.

The single loss (`1HZ100V` long, trade #319) had the one structural
difference worth testing: price was **below** the higher-timeframe EMA
(1053 vs `ema_higher_tf` 1068) and the engine still went long because
`adx >= 25` bypasses that check. That hypothesis was tested on the full
backtest (`research/htf_test.py`, `research/htf_sweep.py`) and **rejected**:

| Rule | Train expectancy | Validation expectancy |
|---|---|---|
| Baseline | -$0.034 | -$0.200 |
| Require close on the correct side of `ema_higher_tf` | -$0.085 | -$0.242 |
| Distance sweep, 0 to 6 ATR tolerance | monotonically worse | -0.024 to +0.023 (sign flips) |

In the validation sample the *farthest-below-EMA* bucket was actually the
best-performing one (51.3% win rate), the opposite of the hypothesis. The
`adx >= 25` bypass is therefore left in place and the loss is treated as
noise. Requiring higher-timeframe agreement would have made the strategy
worse on both slices.

### What the bot now does about it

* `RF_EDGE_GUARD_ENABLED=true` (default): Rise/Fall entries are only allowed
  while the journal has fewer than `RF_EDGE_MIN_TRADES` closed trades
  (exploration at the fixed $1 stake). After that, the 95% Wilson lower bound
  of the realised win rate must beat the payout breakeven, otherwise entries
  are vetoed with a logged reason. The test is scoped **per asset class**
  (synthetic / forex / commodities), so each class gets its own exploration
  window and cannot be dragged down by another class's record. Per-symbol
  version at `RF_EDGE_SYMBOL_MIN_TRADES`.
  * The synthetic class currently **blocks new Rise/Fall entries**
    (54.4% WR but 95% LB 42.7% < 52.5% breakeven). Forex (10 trades) and
    commodities (3 trades) are still inside their exploration window, so they
    trade at the fixed $1 stake while their own evidence accumulates.
  * To keep collecting demo data on synthetics too, set
    `RF_EDGE_MIN_TRADES=200` (or `RF_EDGE_GUARD_ENABLED=false`).
* `RF_ALLOWED_ASSETS=synthetic,forex,commodities` (default in `.env`):
  forex/metal Rise/Fall is enabled. They pay ~0.79-0.82 (55.5% breakeven), so
  the per-class evidence guard above applies. Set it to `synthetic` to
  exclude them again.
* `CONFIDENCE_STAKING=false` (default): stake stays at `FIXED_STAKE`. Scaling
  stake with a non-predictive confidence score only multiplies variance.
* `GEMINI_VETO_ENABLED=false` (default): the LLM opinion is logged as
  advisory. Its rules were fitted to the same 68-trade sample and added
  latency/timeouts.
* The self-learning loop can no longer ratchet `min_score` or move RSI gates
  on 2-loss noise; it needs >=20-30 trades per decision.

### The $10 account (measured blocker)

A $10 real account at a $1 stake **cannot place a single Rise/Fall trade** with
the current risk caps, and no environment variable can fix it:

| Limit | Value | On $10 | vs $1 stake |
|---|---|---|---|
| `MAX_TOTAL_OPEN_RISK` | 3% (hard ceiling 6%) | $0.30 ($0.60 at ceiling) | **below stake -> every entry vetoed** |
| `MAX_DAILY_DRAWDOWN` | 3% | $0.30 | one loss halts the day |
| `MIN_BALANCE` | $10.00 | exactly at the floor | after one loss, all entries vetoed |

`RiskManager.evaluate()` was run directly against these settings:

```
balance $10.00 -> approved=False  ['total open risk 1.00 > cap 0.30']
balance $20.00 -> approved=False  ['total open risk 1.00 > cap 0.60']
balance $33.34 -> approved=True
```

Minimum funding at the default 3% cap is **~$33.34**; at the 6% hard ceiling it
is **~$16.67**. A $10 account needs a smaller stake (Deriv allows below $1) or
more capital. Startup now **refuses to run** a configuration whose open-risk cap
is below the stake, instead of looking alive while vetoing everything
(`_account_economics` in `main.py`, also printed by `--preflight`).

On a real-money start it additionally refuses to run if the evidence guard is
disabled or left in exploration mode (`RF_EDGE_MIN_TRADES` above the default
40), and warns when the balance is under 20x the stake. The 14-day replay and
the Monte Carlo above are the reason: at the measured win rate, $10 at $1/trade
is ruined before doubling in ~82% of runs.

### The one validated exclusion: `stpRNG4`

`stpRNG4` replayed 46 trades with 12 wins (**26.1%**), binomial p=0.0008
against a fair coin and 1.9% family-wise across the 23 symbols. It was bad in
both halves of the 14 days (28.6%, then 22.2%) and its step size did not change
over the period, so this is not a regime artifact. It is now in
`EXCLUDED_RF_SYMBOLS` in `strategy_engine.py` (mirrored in
`research/rf_backtest.py`, and `engine_mirror_check.py` still passes 460/460).

This is deliberately different from the rejected v11 blacklist, which was based
on **3 live losses** for a symbol that was average in the replay. Excluding
`stpRNG4` halves the replay loss (-$45.84 -> -$23.24) but leaves the book at
-$0.034/trade: it reduces the bleed, it does not make the strategy profitable.

Last 7 days, continuous 1-minute candles, no missing hours
(`research/fill_gap.py` -> `data/candles_7d/`, window 2026-10-18 06:09 ->
2026-10-25 06:09 UTC). Trades attributed to the UTC day they were opened:

| strategy | trades | wins | WR | 95% CI | net PnL | exp/trade | maxDD |
|---|---|---|---|---|---|---|---|
| OLD (v10, no exclusion) | 361 | 168 | 46.5% | 41.5–51.7 | −$40.64 | −$0.113 | $47.58 |
| **NEW (v10 + no stpRNG4)** | **342** | **163** | **47.7%** | 42.4–53.0 | **−$31.39** | **−$0.092** | **$41.71** |
| v11 (reverted, reference) | 196 | 94 | 48.0% | 41.1–54.9 | −$17.19 | −$0.088 | $21.84 |

Change: −19 trades (all `stpRNG4`, 19 trades / 26.3% / −$9.25), +1.1 pp WR,
+$9.25 net, +$0.021 per trade. Every UTC day is flat-to-better; the improvement
is the removed symbol, not a better entry rule. The book is still negative.

### Reporting correctness (Telegram + console audit)

The alerts had four defects that made the numbers wrong; all are fixed:

* A day with no trades used to fall back to *every trade since id 278* and
  present it as "today". Empty days now report zeros.
* Daily figures are now keyed by **trade-open time** (UTC), so "today" means the
  trades actually taken today and matches the journal list. A separate
  **"settled today (cash)"** line appears only when positions opened earlier
  settled today (e.g. a 10-minute-to-midnight entry, or the bot was down at
  expiry). That line is what reconciles with the broker balance and the risk
  manager's daily limits; without it the two views look like a data error.
* Both **today and yesterday** are reported, plus an all-time journal line, so a
  fresh UTC day no longer looks like a blank slate.
* The displayed balance is the **broker's live balance**. The old figure was
  reconstructed from a stored day-start equity that had gone stale after a demo
  reset (it showed $9960.63 while the account held $10001.72). Startup now
  reconciles `day_start_equity` with the broker, which also keeps the daily
  loss/profit limits honest.

Smaller fixes: the loss alert no longer hardcodes "-$1.00 Stake", Bollinger %B
reads the real `bb_pctb` key (it always printed 0.000), the day-PnL marker is
red on losses, and the "Target >=75%" / hardcoded "Calibrated #284+" lines were
replaced with the actual breakeven and the all-time record. Preview the three
message templates offline (no network, no writes to the live DB):

```bash
.venv/bin/python research/preview_telegram.py
```

Re-run the research at any time (read-only, no orders):

```bash
.venv/bin/python research/optimize_rf.py     # baseline, diagnosis, walk-forward
.venv/bin/python research/policy_test.py     # policy / fade / whitelist tests
.venv/bin/python research/htf_test.py        # HTF-agreement gate (rejected)
.venv/bin/python research/analyze_session.py # score a live session vs baseline
.venv/bin/python research/engine_mirror_check.py  # engine == backtest mirror?
.venv/bin/python research/v11_ab.py          # v10 vs v11 rules, train/validation
.venv/bin/python research/v11_rules.py       # marginal effect of each v11 rule
.venv/bin/python research/replay_window.py --start ... --end ...  # replay a window
.venv/bin/python research/edge_study.py      # duration sweep + every signal bucket
.venv/bin/python research/predictability.py  # autocorrelation, momentum/reversal
.venv/bin/python research/fade_search.py     # 822 conditional fade rules, 5 folds
.venv/bin/python research/fade_holdout.py    # survivors on the untouched holdout
.venv/bin/python research/duration_folds.py  # is a fixed expiry real or noise?
.venv/bin/python research/live_vs_backtest.py # mirror vs the real live trades
.venv/bin/python research/ruin_sim.py        # $10 account: P(double) vs P(ruin)
.venv/bin/python research/last7_compare.py   # old vs new rules, last 7 days
.venv/bin/python research/fill_gap.py        # rebuild a gap-free 7-day candle set
.venv/bin/python research/fair_game_test.py  # is direction predictable at all?
.venv/bin/python research/drift_audit.py     # raw up-rate per symbol/expiry
.venv/bin/python research/candle_micro.py    # open/close/gap microstructure
```

### Rejected change: "v11" loss-driven rule tightening (2026-10-25)

A second model rewrote the entry gates from individual losing trades: ADX floor
20→22, a 1HZ15V blacklist, mandatory `close >= EMA200` on longs, `macd_rel`
0.003→0.008, an "exhaustion guard", and a hard-coded 0.50 confidence floor.
Measured on identical candles with a train/validation split:

| variant | trades | train exp | validation exp | all exp |
|---|---|---|---|---|
| v10 (restored) | 725 | −0.018 | −0.157 | **−0.063** |
| v11 (shipped) | 385 | −0.075 | −0.121 | −0.089 |

No individual rule improved **both** halves — the best validation rule
(`score>=3`) degraded training by −0.026, which is the signature of fitting
noise. The 1HZ15V blacklist was based on 3 live losses; in the 14-day replay it
is average (18 trades, 50% WR, −$0.06/trade). All of it was reverted; the v11
path remains in `research/rf_backtest.py` as `variant="v11"` so it can be
re-measured, never re-shipped on faith.

Two structural fixes came out of the same audit:

* **Single-instance lock** (`data/bot.lock`, acquired with `O_EXCL`). Two
  concurrent bots had produced duplicate contracts on the same symbol in the
  same minute (journal: `stpRNG` 2026-10-18 06:03, `1HZ50V` 2026-10-22 15:01)
  and doubled every log line. A stale lock from a dead PID is reclaimed
  automatically.
* **Loss post-mortems no longer blame slippage.** The LLM was given the fill
  price next to signal-bar indicators and "found" rule breaches that never
  happened (trade #322: fill 7328.50 vs EMA9 7328.97, while the signal bar
  closed 7329.10 — above EMA9). It now receives the signal-bar close and the
  fill as separate, labelled values, and the log records the objective
  signal-bar facts next to its opinion.

---

## 🤖 The 3 Bot Modes

The repository provides three dedicated runner scripts and an interactive launcher:

### 1. Only Multiplier (`run_multiplier.py` or `./run.sh multiplier`)
* **Trades exclusively Multiplier contracts** across all supported symbols (Volatilities, Crash/Boom, Crypto, Jump, Step).
* **Real-time Trade Protection**:
  * **Green Shield**: Automatically locks profit at $+0.05$ once a trade hits $+0.150$.
  * **Dynamic Trailing Stop**: Secures peak gains on high-momentum runners.
  * **Loss Floor**: Software emergency cut at $-\$0.28$ (saving capital well before broker stop-out).
  * **Thesis Invalidation**: Immediate exit if candle closes against the Fast EMA.

### 2. Only Fixed Expiry Rise/Fall (`run_rise_fall.py` or `./run.sh rise_fall`)
* **Trades Fixed Expiry (`CALL` / `PUT`) options** on synthetic indices (`R_10`–`R_100`, `1HZ10V`–`1HZ100V`, Jumps `JD10`–`JD100`, Step Indices `stpRNG`–`stpRNG5`) plus forex/metals (the 8 major forex pairs and `frxXAUUSD`/`frxXAGUSD`) when their market is open (`RF_ALLOWED_ASSETS=synthetic,forex,commodities`). Forex/metals are only validated on weekdays, and each asset class must pass its own evidence guard.
* Automatically filters out symbols that do not support Rise/Fall (like Crash/Boom).
* Payout is ~88–95% on synthetics (breakeven ~51–53%) and ~79–82% on forex/metals (breakeven ~55%). Entries are gated by the per-asset-class evidence guard (see "Measured reality" above): a class is vetoed once it has enough trades and its 95% win-rate lower bound fails its own breakeven.
* Automatically settles at expiration without manual exit friction.

### 3. Hybrid Dual-Execution (`run_hybrid.py` or `./run.sh hybrid` - Recommended)
* **Two independent books, one command**: Multiplier and Rise/Fall run at the same time, each with its own entry gates, sizing, expiry/stop management, cooldowns and risk ledger (trades/day, daily drawdown, loss-streak halt). A bad run in one book cannot veto, halt or force-close the other.
* **Both books scan their full symbol sets — no static split**: every symbol that supports Multipliers is evaluated by the Multiplier book, and every symbol that supports Rise/Fall is evaluated by the Rise/Fall book. A dual-capable symbol is watched by both.
* **One symbol, one holder at a time**: when a book opens on a symbol it owns that symbol until the contract closes; the other book waits. This prevents the same symbol being traded by both strategies at once (the old double-entry conflict), while still letting either strategy take the next opportunity on any symbol.
* **Same-bar tie-break**: if both books approve an entry on the same bar and the symbol is free, `HYBRID_SIGNAL_PRIORITY` decides (`multiplier` by default, or `rise_fall`); the other book waits for the close.
* **Optional per-book allow-lists**: `HYBRID_MULT_SYMBOLS` and `HYBRID_RF_SYMBOLS` (comma separated) restrict a book to a subset. Empty means all capable symbols; overlap is allowed because ownership is enforced per entry.
* **Shared direction, separate execution**: direction still comes from the one confluence engine; each book then applies its own contract economics (multiplier trailing/loss-cut vs fixed-expiry CALL/PUT).
* Escape hatches: `HYBRID_ISOLATED_RISK=false` (single shared ledger) and `HYBRID_BLOCK_CROSS_SYMBOL=false` (allow both books to hold the same symbol). Both default to **on**; turning them off restores the old conflicting behaviour.
* **Stake stays at $1.00** per contract. With isolated books each book enforces its own limits, so combined open exposure can reach roughly twice a single-book run.

---

## 📁 File Structure

```
├── config.py                 # Central frozen configuration, symbols, risk parameters, TradeMode
├── deriv_client.py           # Async WebSocket client, PAT auth, candles, buy_multiplier, buy_rise_fall
├── strategy_engine.py        # 72.4% Confluence strategy (EMA, ADX, RSI, MACD, BB %B)
├── risk_manager.py           # Risk gatekeeper, concurrent multi-contract tracking, sizing
├── gemini_advisor.py         # Google Gemini AI: Pre-trade veto filter & forensic loss audit
├── memory_and_learning.py    # SQLite journal, post-mortems, milestone batch reviews
├── main.py                   # Master asyncio orchestrator and candle processor
│
├── run.sh                    # Interactive launcher menu & command helper
├── run_multiplier.py         # Runner: Multiplier-Only Bot
├── run_rise_fall.py          # Runner: Fixed Expiry Rise/Fall Bot
├── run_hybrid.py             # Runner: Hybrid Dual-Execution Bot
│
├── data/
│   ├── bot_memory.sqlite3    # Persistent SQLite database (trades, post-mortems, reviews)
│   └── candles/              # Cached 1m candles used by the offline research harness
├── research/
│   ├── deriv_data.py         # Read-only candle fetcher (authenticated demo WS, no orders)
│   ├── fetch_payouts.py      # Read-only proposal quotes -> payouts.json
│   ├── rf_backtest.py        # Replays StrategyEngine signals on candles + settles at payout
│   ├── optimize_rf.py        # Train/validation selection + walk-forward stability check
│   └── policy_test.py        # Policy, fade and symbol-whitelist out-of-sample tests
└── logs/
    ├── bot.log               # Live execution log with rotating file handler
    └── loss_post_mortems.jsonl # Forensic post-mortem records
```

---

## 🚀 Quick Start

### 1. Interactive Launcher
Launch the interactive menu:
```bash
./run.sh
```
Or run directly:
```bash
./run.sh multiplier    # Run Mode 1
./run.sh rise_fall     # Run Mode 2
./run.sh hybrid        # Run Mode 3
```

### 2. Direct Python Runners
```bash
# Mode 1: Multiplier only
.venv/bin/python run_multiplier.py

# Mode 2: Fixed Expiry 3m Rise/Fall only
.venv/bin/python run_rise_fall.py

# Mode 3: Hybrid Dual Execution
.venv/bin/python run_hybrid.py
```

### 3. Preflight & Dry Run
To validate connection, authentication, and all tradable symbols before sending real orders:
```bash
./run.sh preflight multiplier
./run.sh preflight rise_fall
./run.sh preflight hybrid

# Dry-run (evaluates full pipeline, Gemini AI, without placing broker orders):
./run.sh dry-run hybrid
```

### 4. Emergency Kill Switch
* **Activate Kill Switch**: `./run.sh kill` (or `touch KILL_SWITCH`)
* **Deactivate Kill Switch**: `./run.sh unkill` (or `rm -f KILL_SWITCH`)
* **Stop After Current Trades Close**: `touch STOP_AFTER_TRADE` (bot completes open trades then exits cleanly).

---

## 🛡️ Risk & Capital Protection Ceilings
* **Fixed Stake**: `$1.00` per order (`FIXED_STAKE=1.00`).
* **Hard Daily Loss Limit**: Max daily drawdown capped at `3%` of start equity.
* **Loss-Streak Cooldown**: 4 consecutive losses trigger an automatic 60-minute pause.
* **Real-time Software Loss Cap**: Multiplier losses are cut at `-$0.28` to avoid broker spread slippage.
* **Green Shield**: Automatically secures profit once a position gains `+$0.15`.
