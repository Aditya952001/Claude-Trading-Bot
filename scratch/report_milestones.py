#!/usr/bin/env python3
import sqlite3, json, datetime

conn = sqlite3.connect('data/bot_memory.sqlite3')
cur = conn.cursor()

print("=" * 74)
print("             🏆 10-TRADE AUTOMATED MILESTONE & PnL REPORT 🏆")
print("=" * 74)

# 1. Fetch saved milestones
rows = cur.execute('SELECT batch_index, batch_trades, batch_wins, batch_losses, batch_win_rate, batch_gross_profit, batch_gross_loss, batch_net_pnl, batch_profit_factor, overall_trades, overall_wins, overall_losses, overall_win_rate, overall_net_pnl, overall_profit_factor, notes, datetime(ts, "unixepoch", "localtime") FROM milestones ORDER BY batch_index ASC').fetchall()

for r in rows:
    notes = json.loads(r[15] or "[]")
    notes_str = " | ".join(notes) if notes else "Standard execution"
    print(f"\n📦 BATCH #{r[0]} ({r[1]} Trades | Recorded: {r[16]}):")
    print(f"   * Win Rate:      {r[4]*100:.1f}% ({r[2]} Wins, {r[3]} Losses)")
    print(f"   * Gross Profit:  +${r[5]:.2f}")
    print(f"   * Gross Loss:    -${r[6]:.2f}")
    print(f"   * Batch Net PnL: {'+' if r[7] >= 0 else ''}${r[7]:.2f}")
    print(f"   * Profit Factor: {r[8]:.2f}")
    print(f"   * Cumulative:    {r[9]} Trades ({r[12]*100:.1f}% Win Rate) | Net: {'+' if r[13] >= 0 else ''}${r[13]:.2f} | PF: {r[14]:.2f}")
    print(f"   * AI Learning:   {notes_str}")

# 2. Current In-Progress Batch
closed_since = int(cur.execute("SELECT value FROM kv WHERE key = 'closed_since_milestone'").fetchone()[0] or 0)
batch_idx = int(cur.execute("SELECT value FROM kv WHERE key = 'milestone_batch_index'").fetchone()[0] or 0) + 1

if closed_since > 0:
    cur_batch = cur.execute('SELECT id, symbol, direction, pnl, datetime(entry_time, "unixepoch", "localtime") FROM trades ORDER BY id DESC LIMIT ?', (closed_since,)).fetchall()
    cur_batch.reverse()
else:
    cur_batch = []

cb_wins = [t for t in cur_batch if t[3] is not None and t[3] > 0]
cb_loss = [t for t in cur_batch if t[3] is not None and t[3] < 0]
cb_be = [t for t in cur_batch if t[3] is not None and t[3] == 0]
cb_prof = sum(t[3] for t in cb_wins)
cb_loss_amt = abs(sum(t[3] for t in cb_loss))
cb_net = cb_prof - cb_loss_amt
cb_wr = len(cb_wins) / (len(cb_wins) + len(cb_loss)) * 100 if (cb_wins or cb_loss) else 0.0

print(f"\n⏳ CURRENT BATCH #{batch_idx} (In Progress: {closed_since}/10 Trades Completed):")
print(f"   * Current Progress: {closed_since} of 10 trades closed (triggers next report in {10 - closed_since} trades)")
print(f"   * Win Rate:         {cb_wr:.1f}% ({len(cb_wins)} Wins, {len(cb_loss)} Losses, {len(cb_be)} Breakeven)")
print(f"   * Gross Profit:     +${cb_prof:.2f}")
print(f"   * Gross Loss:       -${cb_loss_amt:.2f}")
print(f"   * Current Batch Net: {'+' if cb_net >= 0 else ''}${cb_net:.2f}")

print("\n" + "=" * 74)
