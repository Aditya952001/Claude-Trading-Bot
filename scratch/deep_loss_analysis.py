import sqlite3, json

conn = sqlite3.connect('data/bot_memory.sqlite3')
cur = conn.cursor()

trades = cur.execute('SELECT id, symbol, direction, stake, multiplier, sl_amount, tp_amount, pnl, exit_price, entry_price, (exit_time - entry_time), indicators FROM trades WHERE id >= 114').fetchall()

wins = [t for t in trades if t[7] is not None and t[7] > 0]
losses = [t for t in trades if t[7] is not None and t[7] < 0]
be = [t for t in trades if t[7] is not None and t[7] == 0]

print(f"Total trades analyzed: {len(trades)}")
print(f"Wins: {len(wins)} | Losses: {len(losses)} | Breakeven: {len(be)}")
win_rate = len(wins) / (len(wins) + len(losses)) * 100 if (wins or losses) else 0.0
print(f"Win Rate: {win_rate:.1f}%")

avg_win = sum(t[7] for t in wins) / len(wins) if wins else 0
avg_loss = abs(sum(t[7] for t in losses)) / len(losses) if losses else 0
print(f"Average Win:  +${avg_win:.3f}")
print(f"Average Loss: -${avg_loss:.3f}")
if avg_loss > 0:
    print(f"Win/Loss Payoff Ratio: {avg_win / avg_loss:.2f}")

print("\n--- ALL LOSSES BREAKDOWN ---")
for t in losses:
    ind = json.loads(t[11] or '{}')
    rsi = ind.get('rsi', 0)
    dur = t[10] or 0
    print(f"ID {t[0]:3d} | Symbol: {t[1]:10s} | Dir: {t[2]:5s} | PnL: ${t[7]:+.2f} | Stake: {t[3]} | Multiplier: {t[4]} | Duration: {int(dur):3d}s | RSI: {rsi:.1f}")

print("\n--- PERFORMANCE BY SYMBOL ---")
symbols = {}
for t in trades:
    sym = t[1]
    if sym not in symbols:
        symbols[sym] = {"w": 0, "l": 0, "pnl": 0.0}
    pnl = t[7] or 0.0
    symbols[sym]["pnl"] += pnl
    if pnl > 0: symbols[sym]["w"] += 1
    elif pnl < 0: symbols[sym]["l"] += 1

sorted_syms = sorted(symbols.items(), key=lambda x: x[1]["pnl"])
for sym, data in sorted_syms:
    tot = data["w"] + data["l"]
    wr = (data["w"] / tot * 100) if tot > 0 else 0
    print(f"{sym:10s} -> Total: {tot:2d} | Wins: {data['w']:2d} | Losses: {data['l']:2d} | WinRate: {wr:5.1f}% | Net PnL: ${data['pnl']:+.2f}")
