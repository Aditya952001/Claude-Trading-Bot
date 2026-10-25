import sqlite3
import json

conn = sqlite3.connect("data/bot_memory.sqlite3")
conn.row_factory = sqlite3.Row
all_trades = conn.execute("SELECT * FROM trades WHERE status='closed' AND pnl IS NOT NULL ORDER BY id ASC").fetchall()

print(f"Total historical trades: {len(all_trades)}")

def original_filter(t, ind):
    # original logic
    d = t["direction"]
    rsi = ind.get("rsi") or 50.0
    if d == "long":
        return 35.0 <= rsi <= 58.0
    else:
        return 42.0 <= rsi <= 65.0

def patched_filter(t, ind):
    d = t["direction"]
    rsi = ind.get("rsi")
    macdh = ind.get("macd_hist")
    adx = ind.get("adx")
    pctb = ind.get("bb_pctb")
    
    if rsi is None or macdh is None or adx is None or pctb is None:
        return False
        
    # Patch 1: ADX trend strength >= 20.0
    if adx < 20.0:
        return False
        
    if d == "long":
        # Patch 2: MACD Histogram must be positive for long
        if macdh < 0:
            return False
        # Patch 3: Bollinger %B not in extreme zones
        if not (0.20 <= pctb <= 0.68):
            return False
        # Patch 4: Sweet-spot RSI for pullback in uptrend
        if not (42.0 <= rsi <= 60.0):
            return False
    elif d == "short":
        # Patch 2: MACD Histogram must be negative for short
        if macdh > 0:
            return False
        # Patch 3: Bollinger %B not oversold at the floor
        if not (0.32 <= pctb <= 0.80):
            return False
        # Patch 4: Sweet-spot RSI for bounce in downtrend
        if not (40.0 <= rsi <= 58.0):
            return False
            
    return True

def analyze_dataset(trades, filter_fn, name):
    passed = []
    for t in trades:
        ind = json.loads(t["indicators"] or "{}")
        if filter_fn(t, ind):
            passed.append(t)
            
    wins = [t for t in passed if t["pnl"] > 0]
    losses = [t for t in passed if t["pnl"] < 0]
    be = [t for t in passed if t["pnl"] == 0]
    wr = (len(wins) / len(passed) * 100) if passed else 0
    net_pnl = sum(t["pnl"] for t in passed)
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    pf = (gross_win / gross_loss) if gross_loss > 0 else 999.0
    
    # Calculate max consecutive losses
    max_consec_losses = 0
    curr_consec = 0
    consec_distribution = {}
    for t in passed:
        if t["pnl"] < 0:
            curr_consec += 1
            if curr_consec > max_consec_losses:
                max_consec_losses = curr_consec
        else:
            if curr_consec > 0:
                consec_distribution[curr_consec] = consec_distribution.get(curr_consec, 0) + 1
            curr_consec = 0
    if curr_consec > 0:
        consec_distribution[curr_consec] = consec_distribution.get(curr_consec, 0) + 1
        
    print(f"\n=== {name} ===")
    print(f"Trades Taken: {len(passed)} (filtered out {len(trades) - len(passed)} bad entries)")
    print(f"Wins: {len(wins)} | Losses: {len(losses)} | Win Rate: {wr:.1f}%")
    print(f"Net PnL: ${net_pnl:+.2f} | Gross Win: ${gross_win:.2f} | Gross Loss: -${gross_loss:.2f} | PF: {pf:.2f}")
    print(f"Max Consecutive Losses: {max_consec_losses}")
    print(f"Loss Streak Distribution: {dict(sorted(consec_distribution.items()))}")
    return passed

analyze_dataset(all_trades, original_filter, "ORIGINAL STRATEGY LOGIC")
analyze_dataset(all_trades, patched_filter, "PATCHED STRATEGY ENGINE")
