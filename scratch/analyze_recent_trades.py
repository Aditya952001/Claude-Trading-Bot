import sqlite3
import json

con = sqlite3.connect('data/bot_memory.sqlite3')
con.row_factory = sqlite3.Row
cur = con.cursor()

# Check total closed trades
cur.execute('SELECT count(*), sum(case when pnl > 0 then 1 else 0 end), sum(case when pnl < 0 then 1 else 0 end), sum(pnl) FROM trades WHERE status="closed"')
tot, wins, losses, net = cur.fetchone()
print(f"Total Closed: {tot} | Wins: {wins} | Losses: {losses} | Net PnL: ${net:.2f} | WinRate: {wins/tot*100:.1f}%\n")

# Calculate max consecutive losses across the entire history
cur.execute('SELECT id, contract_id, symbol, direction, pnl, exit_price, exit_time FROM trades WHERE status="closed" ORDER BY id ASC')
all_trades = cur.fetchall()

curr_loss_streak = 0
max_loss_streak = 0
loss_streaks = []

for t in all_trades:
    pnl = t['pnl'] or 0.0
    if pnl < 0:
        curr_loss_streak += 1
        if curr_loss_streak > max_loss_streak:
            max_loss_streak = curr_loss_streak
    else:
        if curr_loss_streak > 0:
            loss_streaks.append(curr_loss_streak)
        curr_loss_streak = 0
if curr_loss_streak > 0:
    loss_streaks.append(curr_loss_streak)

print(f"Max Consecutive Losses: {max_loss_streak}")
print(f"Distribution of loss streaks: {sorted(loss_streaks)}")

# Inspect last 15 trades
print("\n--- Last 15 Trades ---")
cur.execute('SELECT id, contract_id, symbol, direction, pnl, exit_price FROM trades WHERE status="closed" ORDER BY id DESC LIMIT 15')
last_15 = cur.fetchall()
for t in reversed(last_15):
    print(f"Trade #{t['id']:3d} | {t['symbol']:12} | {t['direction']:5} | PnL: ${t['pnl']:+.2f} | exit: {t['exit_price']}")

# Inspect loss post mortems table
cur.execute('SELECT count(*) FROM loss_post_mortems')
pm_count = cur.fetchone()[0]
print(f"\nTotal Loss Post-Mortems recorded: {pm_count}")

cur.execute('SELECT id, contract_id, symbol, direction, pnl, failure_category, root_cause, recommended_patch FROM loss_post_mortems ORDER BY id DESC LIMIT 5')
for r in cur.fetchall():
    print(f"\nPost-Mortem #{r['id']} ({r['symbol']} {r['direction']}, ${r['pnl']:+.2f}):")
    print(f"  Category: {r['failure_category']}")
    print(f"  Root Cause: {r['root_cause']}")
    print(f"  Patch Rec: {r['recommended_patch']}")
